from pprint import pprint
import tensorflow as tf
import tflearn
import numpy as np

from .utils import Network
from .base_networks import VTN, VoxelMorph, VTNAffineStem
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer
from .trilinear_sampler import TrilinearSampler


def mask_metrics(seg1, seg2):
    ''' Given two segmentation seg1, seg2, 0 for background 255 for foreground.
    Calculate the Dice score 
    $ 2 * | seg1 \cap seg2 | / (|seg1| + |seg2|) $
    and the Jacc score
    $ | seg1 \cap seg2 | / (|seg1 \cup seg2|) $
    '''
    sizes = np.prod(seg1.shape.as_list()[1:])
    seg1 = tf.reshape(seg1, [-1, sizes])
    seg2 = tf.reshape(seg2, [-1, sizes])
    seg1 = tf.cast(seg1 > 128, tf.float32)
    seg2 = tf.cast(seg2 > 128, tf.float32)
    dice_score = 2.0 * tf.reduce_sum(seg1 * seg2, axis=-1) / (
        tf.reduce_sum(seg1, axis=-1) + tf.reduce_sum(seg2, axis=-1))
    union = tf.reduce_sum(tf.maximum(seg1, seg2), axis=-1)
    return (dice_score, tf.reduce_sum(tf.minimum(seg1, seg2), axis=-1) / tf.maximum(0.01, union))


class RecursiveCascadedNetworks(Network):
    default_params = {
        'weight': 1,
        'raw_weight': 1,
        'reg_weight': 1,
    }

    def __init__(self, name, framework,
                 base_network, n_cascades, rep=1,
                 det_factor=0.1, ortho_factor=0.1, reg_factor=1.0,
                 extra_losses={}, warp_gradient=True,
                 fast_reconstruction=False, warp_padding=False, affine=True,
                 **kwargs):
        super().__init__(name)
        self.det_factor = det_factor
        self.ortho_factor = ortho_factor
        self.reg_factor = reg_factor

        self.base_network = eval(base_network)
        self.stems = [(VTNAffineStem('affine_stem', trainable=affine), {'raw_weight': 0, 'reg_weight': 0})] + sum([
            [(self.base_network("deform_stem_" + str(i),
                                flow_multiplier=1.0 / n_cascades), {'raw_weight': 0})] * rep
            for i in range(n_cascades)], [])
        self.stems[-1][1]['raw_weight'] = 1

        for _, param in self.stems:
            for k, v in self.default_params.items():
                if k not in param:
                    param[k] = v
        # print(self.stems)

        self.framework = framework
        self.warp_gradient = warp_gradient
        self.fast_reconstruction = fast_reconstruction

        self.reconstruction = Fast3DTransformer(
            warp_padding) if fast_reconstruction else Dense3DSpatialTransformer(warp_padding)
        self.trilinear_sampler = TrilinearSampler()

    @property
    def trainable_variables(self):
        return list(set(sum([stem.trainable_variables for stem, params in self.stems], [])))

    @property
    def data_args(self):
        return dict()

    def build(self, img1, img2, seg1, seg2, pt1, pt2, affine_matrix=None):
        stem_results = []

        stem_result = self.stems[0][0](img1, img2, affine_matrix=affine_matrix)
        stem_result['warped'] = self.reconstruction(
            [img2, stem_result['flow']])
        stem_result['agg_flow'] = stem_result['flow']
        # only to get tumor
        seg = seg2>1.5
        seg = tf.cast(seg, tf.float32)
        stem_result['mask'] = self.reconstruction([seg, stem_result['agg_flow']]) > 0.5
        organ_seg = tf.cast(seg2>0.5, tf.uint8)
        flow1_org = tf.cast(self.reconstruction([organ_seg, stem_result['agg_flow']])>0.5, tf.uint8)
        # ratio for each image
        stem_result['ratio']= tf.reduce_sum(flow1_org, axis=[1,2,3,4]) / tf.reduce_sum(organ_seg, axis=[1,2,3,4])
        stem_results.append(stem_result)

        for stem, params in self.stems[1:]:
            if self.warp_gradient:
                stem_result = stem(img1, stem_results[-1]['warped'])
            else:
                stem_result = stem(img1, tf.stop_gradient(
                    stem_results[-1]['warped']))

            if len(stem_results) == 1 and 'W' in stem_results[-1]:
                I = tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1],
                                tf.float32, [1, 3, 3])
                stem_result['agg_flow'] = tf.einsum(
                    'bij,bxyzj->bxyzi', stem_results[-1]['W'] + I, stem_result['flow']) + stem_results[-1]['flow']
            else:
                stem_result['agg_flow'] = self.reconstruction(
                    [stem_results[-1]['agg_flow'], stem_result['flow']]) + stem_result['flow']
            stem_result['warped'] = self.reconstruction(
                [img2, stem_result['agg_flow']])
            # Add mask in stem
            stem_result['mask'] = self.reconstruction([seg, stem_result['agg_flow']]) > 0.5
            flow2_org = tf.cast(self.reconstruction([organ_seg, stem_result['agg_flow']])>0.5, tf.uint8)
            stem_result['ratio']= tf.reduce_sum(flow2_org, axis=[1,2,3,4])/tf.reduce_sum(organ_seg, axis=[1,2,3,4])
            stem_results.append(stem_result)
        # pprint(stem_results)
        if tflearn.get_training_mode():
            for i, stem_result in enumerate(stem_results):
                tf.summary.image(f"Layer {i}/warped seg2", tf.cast(stem_result['mask'][:1,..., 64,:], tf.float32))
                tf.summary.image(f"Layer {i}/warped", stem_result['warped'][:1,..., 64,:])
                tf.summary.image(f"Layer {i}/mask", tf.cast(tf.logical_or(stem_result['mask'],(seg1>1.5))[:1,..., 64,:], tf.float32))
            tf.summary.image("seg1", seg1[:1,..., 64,:])
            tf.summary.image("seg2", seg2[:1,..., 64,:])
            tf.summary.image("img1", img1[:1,..., 64,:])
            tf.summary.image("img2", img2[:1,..., 64,:])
            tf.summary.scalar('organ_ratio', stem_result['ratio'][0])

        print('if masked:', self.framework.masked)
        # unsupervised learning with simlarity loss and regularization loss
        for stem_result, (stem, params) in zip(stem_results, self.stems):
            if 'W' in stem_result:
                stem_result['loss'] = stem_result['det_loss'] * \
                    self.det_factor + \
                    stem_result['ortho_loss'] * self.ortho_factor
                if params['raw_weight'] > 0:
                    if self.framework.masked:
                        stem_result['raw_loss'] = self.similarity_loss(
                            img1, stem_result['warped'], tf.logical_or(stem_result['mask'],(seg1>1.5))
                        )
                    else:
                        stem_result['raw_loss'] = self.similarity_loss(
                            img1, stem_result['warped']
                        )
                    stem_result['loss'] = stem_result['loss'] + \
                        stem_result['raw_loss'] * params['raw_weight']
            else:
                if params['raw_weight'] > 0:
                    if self.framework.masked:
                        stem_result['raw_loss'] = self.similarity_loss(
                            img1, stem_result['warped'], tf.logical_or(stem_result['mask'],(seg1>1.5)))
                    else:
                        stem_result['raw_loss'] = self.similarity_loss(
                            img1, stem_result['warped']
                        )
                if params['reg_weight'] > 0:
                    stem_result['reg_loss'] = self.regularize_loss(
                        stem_result['flow']) * self.reg_factor
                stem_result['loss'] = sum(
                    [stem_result[k] * params[k.replace('loss', 'weight')] for k in stem_result if k.endswith('loss')])

        ret = {}

        flow = stem_results[-1]['agg_flow']
        warped = stem_results[-1]['warped']

        jacobian_det = self.jacobian_det(flow)

        loss = sum([r['loss'] * params['weight']
                    for r, (stem, params) in zip(stem_results, self.stems)])

        pt_mask1 = tf.reduce_any(tf.reduce_any(pt1 >= 0, -1), -1)
        pt_mask2 = tf.reduce_any(tf.reduce_any(pt2 >= 0, -1), -1)
        pt1 = tf.maximum(pt1, 0.0)

        moving_pt1 = pt1 + self.trilinear_sampler([flow, pt1])

        pt_mask = tf.cast(pt_mask1, tf.float32) * tf.cast(pt_mask2, tf.float32)
        landmark_dists = tf.sqrt(tf.reduce_sum(
            (moving_pt1 - pt2) ** 2, axis=-1)) * tf.expand_dims(pt_mask, axis=-1)
        landmark_dist = tf.reduce_mean(landmark_dists, axis=-1)

        if self.framework.segmentation_class_value is None:
            seg_fixed = seg1
            warped_seg_moving = self.reconstruction([seg2, flow])
            # seg mask value should be 255
            dice_score, jacc_score = mask_metrics(seg_fixed, warped_seg_moving)
            jaccs = [jacc_score]
            dices = [dice_score]
        else:
            def mask_class(seg, value):
                return tf.cast(tf.abs(seg - value) < 0.5, tf.float32) * 255
            jaccs = []
            dices = []
            fixed_segs = []
            warped_segs = []
            for k, v in self.framework.segmentation_class_value.items():
                #print('Segmentation {}, {}'.format(k, v))
                fixed_seg_class = mask_class(seg1, v)
                warped_seg_class = self.reconstruction(
                    [mask_class(seg2, v), flow])
                class_dice, class_jacc = mask_metrics(
                    fixed_seg_class, warped_seg_class)
                ret['jacc_{}'.format(k)] = class_jacc
                jaccs.append(class_jacc)
                dices.append(class_dice)
                fixed_segs.append(fixed_seg_class)
                warped_segs.append(warped_seg_class)
                if not tflearn.get_training_mode():
                    tf.Print(class_dice, ['Segmentation {}:{}'.format(k, v), class_dice], summarize=10)
            seg_fixed = tf.stack(fixed_segs, axis=-1)
            warped_seg_moving = tf.stack(warped_segs, axis=-1)
            dice_score, jacc_score = tf.add_n(
                dices) / len(dices), tf.add_n(jaccs) / len(jaccs)
            dice_dct = {'dice_'+k: v for k, v in zip(self.framework.segmentation_class_value.keys(), dices)}
            ret.update(dice_dct)

        ret.update({'loss': tf.reshape(loss, (1, )),
                    'dice_score': dice_score,
                    'jacc_score': jacc_score,
                    'dices': tf.stack(dices, axis=-1),
                    'jaccs': tf.stack(jaccs, axis=-1),
                    'landmark_dist': landmark_dist,
                    'landmark_dists': landmark_dists,
                    'real_flow': flow,
                    'pt_mask': pt_mask,
                    'reconstruction': warped * 255.0,
                    'image_reconstruct': warped,
                    'warped_moving': warped * 255.0,
                    'seg_fixed': seg_fixed,
                    'warped_seg_moving': warped_seg_moving,
                    'image_fixed': img1,
                    'moving_pt': moving_pt1,
                    'jacobian_det': jacobian_det})

        for i, r in enumerate(stem_results):
            for k in r:
                if k.endswith('loss'):
                    ret['{}_{}'.format(i, k)] = r[k]
            ret['warped_seg_moving_%d' %
                i] = self.reconstruction([seg2, r['agg_flow']])
            ret['warped_moving_%d' % i] = r['warped']
            ret['flow_%d' % i] = r['flow']
            ret['real_flow_%d' % i] = r['agg_flow']

        return ret

    def similarity_loss(self, img1, warped_img2, mask=None):
        # mask: 1 means keeping
        sizes = np.prod(img1.shape.as_list()[1:])            
        flatten1 = tf.reshape(img1, [-1, sizes])
        flatten2 = tf.reshape(warped_img2, [-1, sizes])

        if self.fast_reconstruction:
            _, pearson_r, _ = tf.user_ops.linear_similarity(flatten1, flatten2)
        else:
            if mask is not None:
                mask = tf.logical_not(mask)
                mflatten = tf.reshape(tf.cast(mask, tf.float32), [-1, sizes])
                mask = tf.reshape(mask, [-1, sizes])
                mean1 = tf.reduce_sum(flatten1 * mflatten, axis=-1, keep_dims=True) / tf.reduce_sum(mflatten, axis=-1, keep_dims=True)
                mean2 = tf.reduce_sum(flatten2 * mflatten, axis=-1, keep_dims=True) / tf.reduce_sum(mflatten, axis=-1, keep_dims=True)
                # repeat mean to flatten
                flatten1 = tf.where(mask, flatten1, tf.tile(mean1, [1, sizes]))
                flatten2 = tf.where(mask, flatten2, tf.tile(mean2, [1, sizes]))
            else:
                mean1 = tf.reshape(tf.reduce_mean(flatten1, axis=-1), [-1, 1])
                mean2 = tf.reshape(tf.reduce_mean(flatten2, axis=-1), [-1, 1])
            var1 = tf.reduce_mean(tf.square(flatten1 - mean1), axis=-1)
            var2 = tf.reduce_mean(tf.square(flatten2 - mean2), axis=-1)
            cov12 = tf.reduce_mean(
                (flatten1 - mean1) * (flatten2 - mean2), axis=-1)
            pearson_r = cov12 / tf.sqrt((var1 + 1e-6) * (var2 + 1e-6))

        raw_loss = 1 - pearson_r
        raw_loss = tf.reduce_sum(raw_loss)
        return raw_loss

    def regularize_loss(self, flow):
        ret = ((tf.nn.l2_loss(flow[:, 1:, :, :] - flow[:, :-1, :, :]) +
                tf.nn.l2_loss(flow[:, :, 1:, :] - flow[:, :, :-1, :]) +
                tf.nn.l2_loss(flow[:, :, :, 1:] - flow[:, :, :, :-1])) / np.prod(flow.shape.as_list()[1:5]))
        return ret

    def jacobian_det(self, flow):
        _, var = tf.nn.moments(tf.linalg.det(tf.stack([
            flow[:, 1:, :-1, :-1] - flow[:, :-1, :-1, :-1] +
            tf.constant([1, 0, 0], dtype=tf.float32),
            flow[:, :-1, 1:, :-1] - flow[:, :-1, :-1, :-1] +
            tf.constant([0, 1, 0], dtype=tf.float32),
            flow[:, :-1, :-1, 1:] - flow[:, :-1, :-1, :-1] +
            tf.constant([0, 0, 1], dtype=tf.float32)
        ], axis=-1)), axes=[1, 2, 3])
        return tf.sqrt(var)
