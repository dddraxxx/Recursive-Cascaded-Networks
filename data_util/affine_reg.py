import itk
import numpy as np
from scipy.ndimage import map_coordinates

def itk_affine(arr1, arr2):
    # arr dtype float
    fixed_image = itk.GetImageViewFromArray(np.array(arr1).astype(np.float32))
    moving_image = itk.GetImageViewFromArray(np.array(arr2).astype(np.float32))
    parameter_object = itk.ParameterObject.New()
    affine_parameter_map = parameter_object.GetDefaultParameterMap('affine', 4)
    # affine_parameter_map['CenterOfRotationPoint'] = ['0', '0', '0']
    parameter_object.AddParameterMap(affine_parameter_map)

    # Call registration function
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image, parameter_object=parameter_object)
    res = itk.GetArrayFromImage(result_image)
    # normalize to 0-255
    res = (res - res.min())/(res.max()-res.min())*255
    return res, result_transform_parameters

def itk_seg(seg2, params):
    seg2 = itk.GetImageViewFromArray(np.array(seg2).astype(np.float32))
    parameter_map = params.GetParameterMap(0)
    # parameter_map['CenterOfRotationPoint'] = ['0', '0', '0']
    # parameter_map['DefaultPixelValue'] = '2'
    parameter_map['FinalBSplineInterpolationOrder'] = '0'
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(parameter_map)
    m_seg2 = itk.transformix_filter(seg2, parameter_object)
    m_seg2 = itk.GetArrayFromImage(m_seg2).astype(np.uint8)
    return m_seg2, parameter_map

def get_pair_affine(arr1, arr2):
    _, params = itk_affine(arr1, arr2)
    tx = params.GetParameterMap(0)
    params = np.array(tx['TransformParameters'], dtype=np.float32).reshape(4,3)
    W, b = params[:3], params[-1]
    o = np.array(tx['CenterOfRotationPoint'], dtype=np.float32)
    # print(W,b,o)

    def get_total_matrix(W, b, o):
        center = np.eye(4)
        center[:3, 3] = -o
        center_inv = center.copy()
        center_inv[:3, 3] = o
        W = (W.T)[::-1,::-1].T
        b = b[::-1]
        T = np.append(W, b[None].T, axis=1)
        T = np.append(T, [[0, 0, 0, 1]], axis=0)
        # print(center,'\n', T)
        return center_inv@T@center
    total_matrix = get_total_matrix(W, b, o)
    return total_matrix

def affine_img_seg(total_matrix, img, seg=None):
    def affine_flow(W, b, len1, len2, len3):
        b = np.reshape(b, [1, 1, 1, 3])
        xr = np.arange(0, len1, 1.0, np.float32) 
        xr = np.reshape(xr, [-1, 1, 1, 1])
        yr = np.arange(0, len2, 1.0, np.float32) 
        yr = np.reshape(yr, [1, -1, 1, 1])
        zr = np.arange(0, len3, 1.0, np.float32) 
        zr = np.reshape(zr, [1, 1, -1, 1])
        wx = W[:, 0]
        wx = np.reshape(wx, [ 1, 1, 1, 3])
        wy = W[:, 1]
        wy = np.reshape(wy, [ 1, 1, 1, 3])
        wz = W[:, 2]
        wz = np.reshape(wz, [ 1, 1, 1, 3])
        return (xr * wx + yr * wy) + (zr * wz + b)
    tW, tb = total_matrix[:3,:3], total_matrix[:3,3]
    coord = affine_flow(tW,tb,*img.shape)

    coord = coord.transpose(-1, 0, 1, 2)
    res = map_coordinates(img, coord, order=3, mode='constant', cval=0)
    if seg is not None:
        res_seg = map_coordinates(seg, coord, order=0, mode='constant', cval=0)
        return res, res_seg
    return res

if __name__=='__main__':
    from liver import Dataset
    import tqdm
    dataset = Dataset('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/liver_cust.json')
    # 1 is split_train
    generator = dataset.generator(2, batch_size=1, loop=False)
    affine_dct = {}
    save_path = '/home/hynx/regis/Recursive-Cascaded-Networks/datasets/slits_val_affine_dct.npy'
    cnt = 0
    for k in (tqdm.tqdm(generator)):
        cnt += 1
        id1 = k['id1'][0]
        id2 = k['id2'][0]
        # print(id1, id2)
        arr1 = np.array(k['voxel1'], dtype=np.float32)[0,...,0]
        arr2 = np.array(k['voxel2'], dtype=np.float32)[0,...,0]
        mtrix = get_pair_affine(arr1, arr2)
        d1 = affine_dct.setdefault(id1, {})
        d1.update({id2: mtrix})
        if cnt%50==0:
            np.save(save_path, affine_dct)
    if cnt%50!=0:
        np.save(save_path, affine_dct)
    #%%
    # import numpy as np
    # a = np.load('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/slits_train_affine_dct.npy', allow_pickle=True).item()
    # print(len(a['lits_24']))
