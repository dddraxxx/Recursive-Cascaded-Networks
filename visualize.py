# import pickle as pkl

# # read pkl file from evaluate folder
# dct = pkl.load(open('evaluate/Jul08-1538-model-1500.pkl', 'rb'))
# print(dct.keys())
# print(dct['seg2'][-1].shape)
#%%
from pprint import pprint
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle as pkl
from adet.utils.visualize_niigz import *
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path as pa


# fname = '/home/hynx/regis/Recursive-Cascaded-Networks/evaluate/Jul26-1358-model-7000.pkl'
# fname = '/home/hynx/regis/Recursive-Cascaded-Networks/evaluate/Aug03-0139-model-7000.pkl'
# fname = '/home/hynx/regis/Recursive-Cascaded-Networks/evaluate/Aug08-1440-model-5560.pkl'
fname = '/home/hynx/regis/Recursive-Cascaded-Networks/evaluate/Aug23-1306-model-99500.pkl'
fname = '/home/hynx/regis/Recursive-Cascaded-Networks/evaluate/Aug23-1307-model-99500.pkl'
fname = '/home/hynx/regis/Recursive-Cascaded-Networks/evaluate/Aug23-1307-model-99500-slits-val.pkl'

dct = pkl.load(open(fname, 'rb'))
print(dct.keys())
print(dct['dices'])

idx = 10
mode = 'mask'
dir_name = f'{mode}_pair{idx}'
(pa('images')/dir_name).mkdir(exist_ok=True)
# include img1,2 seg1,2 warped_moving
keys = ['img1', 'img2', 'seg1', 'seg2', 'warped_moving']

k = 'warped_seg_moving'
p = np.maximum((dct[k][...,0,1]>0.5*255)*2,(dct[k][...,0,0]>0.5*255))
print('overlap in warped_seg_moving: {}'.format(np.logical_and(dct[k][idx,...,0,1]>0.5*255,dct[k][idx,...,0,0]>0.5*255).sum()))

seg_pairs = zip(*[dct[k][...,0] for k in keys], p)
keys.append(k)


def basic_vis(idx):
    s = list(seg_pairs)[idx]
    print('id1:{} id2:{}'.format(dct['id1'][idx], dct['id2'][idx]))
    # for i in range(len(s)):
        # print(s[i].shape)
        # print(np.unique(s[i]))
        # visulize_3d(s[i], inter_dst=3, save_name=f'images/{dir_name}/{keys[i]}.jpg')
    # draw seg on img
    pairs = [('im1', s[0], s[2]), ('im2', s[1], s[3]), ('warped', s[4], s[5])]
    for k,i,s in pairs:
        img = draw_seg_on_vol(np.expand_dims(i, 0), np.expand_dims(s, 0)==2, alpha=0.8, colors=["red"])
        visulize_3d(img, inter_dst=3, save_name=f'images/{dir_name}/{k}.jpg')  

    # calculate area change after each flow, VTN-3
    for i in range(4):
        k = f'warped_seg_moving_{i}'
        seg = dct[k][idx, ..., 0]
        # print(np.unique(seg))
        print('flow tumour area{}:'.format(i), (seg>1.5).sum())
        print('flow organ area{}:'.format(i), (seg>0.5).sum())
        print()

# visualize each flow in plt subplot
def plot_flow(flows, width, topk, gt):
    idx = np.argsort((gt>1.5).sum((1,2)))[-topk:]
    print(idx)
    le = len(flows[idx])
    r = (le+width-1)//width
    fig, axes = plt.subplots(r*2, width, figsize=(width*7, 2*r*5))
    x, y, z = np.meshgrid(np.arange(128), np.arange(128), np.arange(128), indexing='ij')

    u, v, w = np.transpose(flows, (-1, 0, 1, 2))
    fx, fy, fz = u, y+v, z+w
    fff = np.stack((fx, fy, fz), axis=1)[idx]
    gt = gt[idx]
    for flow, ax, g in zip(fff, axes.flat[:r*width], gt):
        cp = plot_single_flow(flow, fig, ax, g)
        fig.colorbar(cp, ax=ax)
    for ax, g in zip(axes.flat[r*width:], gt):
        cmap1 = matplotlib.colors.ListedColormap(['none', 'green', 'red'])
        ax.imshow(g, cmap=cmap1)
    fig.tight_layout(pad=0)
    return idx
    

def plot_single_flow(flow, fig, ax, gt=None):
    fx, fy, fz = flow
    plot_grid(fz, fy, ax, colors='black', alpha=0.3)
    if gt is not None:
        ax.invert_yaxis()
        px, py, pz = fx[gt==1], fy[gt==1], fz[gt==1]
        ax.scatter(pz, py, alpha=0.5, c='g')
        px, py, pz = fx[gt==2], fy[gt==2], fz[gt==2]
        ax.scatter(pz, py, alpha=0.5, c='r')
        # cmap1 = matplotlib.colors.ListedColormap(['none', 'green', 'red'])
        # eps = 1e-2
        # ax.contourf(gt.transpose(-1,-2), [0.5, 1.5], cmap=cmap1, alpha=0.8)
    cp = ax.contourf(fz, fy, fx, cmap='bwr', alpha=0.3)
    # ax.axis('off')
    return cp

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    return ax

def cal_revflow(flow):
    import SimpleITK as sitk
    displacement_image = sitk.GetImageFromArray(flow, isVector=True)
    # use sitk.InverseDisplacementField
    inverse_displacement_field = sitk.InverseDisplacementField(displacement_image, size=displacement_image.GetSize())
    # inverse_displacement_field = sitk.InvertDisplacementField(displacement_image)
    inv_np = sitk.GetArrayFromImage(inverse_displacement_field)
    return inv_np

def elastix_affine(arr1, arr2):
    '''
    arr1 the fixed
    arr2 the moving'''
    import SimpleITK as sitk
    img1 = sitk.GetImageFromArray(arr1)
    img2 = sitk.GetImageFromArray(arr2)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(img1)
    elastixImageFilter.SetMovingImage(img2)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.Execute()
    # return elastixImageFilter.GetResultImage()
    # return the transform
    return elastixImageFilter.GetTransformParameterMap()

#%%
img1 = dct['img1'][idx,...,0]
img2 = dct['img2'][idx,...,0]
# m_img2 = sitk.GetArrayFromImage(elastix_affine(img1, img2))
# visulize_3d(m_img2, inter_dst=3, save_name=f'images/{dir_name}/m_img2_tx.jpg')
tx = elastix_affine(img1, img2)
seg2 = dct['seg2'][idx,...,0]
tx[0]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
m_seg2 = sitk.GetArrayFromImage(sitk.Transformix(sitk.GetImageFromArray(seg2), tx))
visulize_3d(m_seg2, inter_dst=3, save_name=f'images/{dir_name}/m_seg2_tx.jpg')
#%%
print('elastix affine results: tumour area: {}, organ area {}'.format((m_seg2==2).sum(), (m_seg2==1).sum()))
#%%
flow = dct['real_flow'][idx]
origin_seg1 = dct['seg1'][idx, ..., 0]
gt = dct['seg2'][idx, ..., 0]
print('organ area   {}:{}:{}'.format((origin_seg1==1).sum(), (gt==1).sum(), (p[idx]==1).sum()))
print('tumour area  {}:{}:{}'.format((origin_seg1==2).sum(), (gt==2).sum(), (p[idx]==2).sum()))
#%%
basic_vis(idx)
rev_flow = cal_revflow(flow)
plot_flow(rev_flow, 2, 2, gt=gt)
plt.savefig(f'images/{dir_name}/grid_vis.png')

    