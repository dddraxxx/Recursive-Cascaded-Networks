# %%
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import numpy as np
from pathlib import Path as pa
import pygem as pg

from adet.utils.visualize_niigz import *

def show_img(res):
    import torchvision.transforms as T
    res = tt(res)
    if res.ndim>=3:
        return T.ToPILImage()(visulize_3d(res))
    # normalize res
    res = (res-res.min())/(res.max()-res.min())
    return T.ToPILImage()(res)

def plt_grid3d(pts, ax, **kwargs):
    l = np.round(len(pts)**(1/3)).astype(int)
    x, y, z = pts.T.reshape(-1, l, l, l)
    grid1 = np.stack((x,y,z), axis=-1)
    ax.add_collection3d(Line3DCollection(grid1.reshape(-1, l, 3), **kwargs))
    ax.add_collection3d(Line3DCollection(grid1.transpose(0,2,1,3).reshape(-1, l, 3),  **kwargs))
    ax.add_collection3d(Line3DCollection(grid1.transpose(1,2,0,3).reshape(-1, l, 3),  **kwargs))

from PIL import Image
def combine_pil_img(*ims):
    widths, heights = zip(*(i.size for i in ims))
    total_width = sum(widths) + 5*(len(ims)-1)
    total_height = max(heights)
    im = Image.new('RGB', (total_width, total_height), color='white')
    for i, im_ in enumerate(ims):
        im.paste(im_, (sum(widths[:i])+5*i, 0))
    return im

def save_niigz(arr, path):
    import itk
    img = itk.GetImageFromArray(arr)
    itk.imwrite(img, path)

# calculate centroid of seg2
def cal_centroid(seg2):
    obj = seg2==2
    obj = obj.astype(np.float32)
    x, y, z = np.meshgrid(np.arange(obj.shape[0]), np.arange(obj.shape[1]), np.arange(obj.shape[2]), indexing='ij')
    c_x = np.sum(x*obj)/np.sum(obj)
    c_y = np.sum(y*obj)/np.sum(obj)
    c_z = np.sum(z*obj)/np.sum(obj)
    print(c_x, c_y, c_z)
    # show_img(img2[:,:, int(c_z)])

# %% part 1: simulate pressure of the tumor

# find a location to paste tumour
def find_loc_to_paste_tumor(seg1, kseg2, bbox):
    '''
    1: organ, 2:tumor'''
    box_length = bbox[1] - bbox[0]
    area =  seg1.shape - box_length
    seg1_area = seg1[:area[0], :area[1], :area[2]]
    pts = (seg1_area == 1).nonzero()
    pts = np.stack(pts, axis=-1)
    pts = np.random.permutation(pts)
    # print(len(pts))
    for i in range(1000):
        pt = pts[i]
        s1 = seg1[pt[0]:pt[0]+box_length[0], pt[1]:pt[1]+box_length[1], pt[2]:pt[2]+box_length[2]]==1
        s2 = kseg2[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]==2
        if (s1&s2 == s2).all():
            return np.array([pt, pt+box_length])
    raise ValueError("cannot find a location to paste tumor")

def find_bbox(seg2):
    from monai.transforms import KeepLargestConnectedComponent
    from monai.transforms.utils import generate_spatial_bounding_box
    k = KeepLargestConnectedComponent(2)
    kseg2 = np.maximum(k(seg2[None].copy()), seg2>0)
    bbox = generate_spatial_bounding_box(kseg2, lambda x:x==2)
    kseg2 = kseg2[0]
    o_bbox = np.array(bbox)
    print("origin bbox", o_bbox)
    f_bbox = find_loc_to_paste_tumor(seg1, kseg2, o_bbox)
    box_length = f_bbox[1]-f_bbox[0]
    ext_l = np.random.rand(3)*(box_length/2)
    print('box length', box_length, '\nextenede length', ext_l)
    bbox = np.stack([np.maximum(f_bbox[0]-ext_l, 0), np.minimum(f_bbox[1]+ext_l, 128-1)], axis=0)
    # print('extended bbox', bbox[0], bbox[1])
    # print("find bbox", f_bbox)
    # print("new center", np.array([c_x, c_y, c_z]) - o_bbox[0] + f_bbox[0])
    return kseg2, o_bbox, f_bbox, bbox

def draw_3dbox(res, l, r, val=255):
    res = res.copy()   
    res[l[0]:r[0], l[1], l[2]:r[2]] = val
    res[l[0]:r[0], r[1], l[2]:r[2]] = val
    res[l[0]:r[0], l[1]:r[1], l[2]] = val
    res[l[0]:r[0], l[1]:r[1], r[2]] = val
    return res
    
from scipy.ndimage import map_coordinates
def presure_ffd(o_bbox, f_bbox, bbox, kseg2):
    l = 8
    ffd = pg.FFD([l, l, l])
    ffd.box_origin = bbox[0]
    ffd.box_length = bbox[1]-bbox[0]
    control_pts = ffd.control_points(False).reshape(l,l,l,3)[1:-1,1:-1,1:-1]
    # calculate tumor map for each control point
    paste_seg2 = np.zeros_like(kseg2)
    paste_seg2[f_bbox[0][0]:f_bbox[1][0], f_bbox[0][1]:f_bbox[1][1], f_bbox[0][2]:f_bbox[1][2]] = kseg2[o_bbox[0][0]:o_bbox[1][0], o_bbox[0][1]:o_bbox[1][1], o_bbox[0][2]:o_bbox[1][2]]
    seg_pts = map_coordinates(paste_seg2==2, control_pts.reshape(-1,3).T, order=0, mode='nearest')
    seg_pts = seg_pts.reshape(*control_pts.shape[:3])
    # calculate the pressure
    dsp_matrix = control_pts.reshape(-1, 3) - control_pts.reshape(-1, 3)[:, None]
    dsp_matrix /= ffd.box_length
    # diagonal entries are the same point, and no need to calculate the distance
    dst_matrix = (dsp_matrix**2).sum(axis=-1)
    np.fill_diagonal(dst_matrix, 1)
    fct = np.random.rand()*0.4+0.2
    print('factor', fct)
    prs_matrix = fct*dsp_matrix/dst_matrix[...,None]
    prs_pts = (prs_matrix*seg_pts.reshape(-1,1)).sum(axis=1)
    dx, dy, dz = prs_pts.reshape(*control_pts.shape).transpose(-1,0,1,2)/((l-1)**3)
    # dx, dy, dz = prs_pts.reshape(*control_pts.shape).transpose(-1,0,1,2)/(l-1)/ffd.box_length[:,None,None,None]
    ffd.array_mu_x[1:-1,1:-1,1:-1] = dx
    ffd.array_mu_y[1:-1,1:-1,1:-1] = dy
    ffd.array_mu_z[1:-1,1:-1,1:-1] = dz

    ## %% plot deform
    # %matplotlib widget
    # fig = plt.figure(figsize=(4, 4))
    # ax = Axes3D(fig)
    # ax.scatter(*control_pts[seg_pts].T, s=20, c='green')
    # # ax.scatter(*control_pts[~seg_pts].T, s=20)
    # ax.scatter(*ffd.control_points().T, s=5, c='red')
    # # plt_grid3d(ffd.control_points(False), ax, colors='b', linewidths=1)
    # plt_grid3d(ffd.control_points(), ax, colors='r', linewidths=0.5)
    # # show x,y,z axis
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.view_init(3, 90)
    return ffd, paste_seg2

def do_ffd(ffd, img):
    x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]), indexing='ij')
    mesh = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    n_mesh = ffd(mesh)
    nr_mesh = n_mesh.reshape(128,128,128,3)
    # this is backward pass, not correct
    res = map_coordinates(img, nr_mesh.transpose(-1,0,1,2), order=3, mode='constant', cval=0)
    return res
#%% paste tumor
def roll_pad(data, shift, padding=0):
    res = np.roll(data, shift, axis=(0,1,2))
    off_index = lambda x: np.s_[int(x):] if x<0 else np.s_[:int(x)]
    offset = [off_index(i) for i in shift]
    res[offset[0], ...] = padding
    res[:, offset[1]] = padding
    res[..., offset[2]] = padding
    return res

from scipy.ndimage import distance_transform_edt

# signed distance map
def sdm(kseg2, feather_len):
    global dist_pts, weight_kseg2, dst_to_weight
    dist_pts = distance_transform_edt(kseg2!=2) - distance_transform_edt(kseg2==2)
    weight_kseg2 = np.zeros_like(kseg2, dtype=np.float32)
    dst_to_weight = lambda x, l: 0.5-x/2/l
    idx = abs(dist_pts)<feather_len
    weight_kseg2[idx] = dst_to_weight(dist_pts[idx], feather_len)
    weight_kseg2[dist_pts<=-feather_len] = 1
    # feathering area should be inside organ
    inside_organ = kseg2>0
    weight_kseg2[~inside_organ] = 0
    return weight_kseg2

# distance map
def dm(kseg2, feather_len):
    global dist_pts, weight_kseg2, dst_to_weight
    dist_pts = distance_transform_edt(kseg2!=2)
    weight_kseg2 = np.zeros_like(kseg2, dtype=np.float32)
    dst_to_weight = lambda x, l: 1-x/l
    idx = dist_pts<feather_len
    weight_kseg2[idx] = dst_to_weight(dist_pts[idx], feather_len)
    inside_organ = kseg2>0
    weight_kseg2[~inside_organ] = 0
    return weight_kseg2

def direct_paste(res, img2, kseg2, f_bbox, o_bbox):
    weight_kseg2 = kseg2.copy()
    weight_seg1 = roll_pad(1-weight_kseg2, f_bbox[0]-o_bbox[0], padding=1)
    roll_img2 = roll_pad(img2, f_bbox[0]-o_bbox[0], padding=0)
    res1 = res*weight_seg1 + roll_img2*(1-weight_seg1)
    # res1 = res.copy()
    # res1[paste_seg2==2] = img2[kseg2==2]
    # direct paste tumor with borders
    # res1 = res.copy()
    # roll_dist = roll_pad(dist_pts, f_bbox[0]-o_bbox[0], padding=feather_len+1)
    # res1[roll_dist<feather_len]=img2[dist_pts<feather_len]
    return res1

def feathering_paste(feather_len, res, img2, kseg2, f_bbox, o_bbox):
    weight_kseg2 = sdm(kseg2, feather_len)
    weight_seg1 = roll_pad(1-weight_kseg2, f_bbox[0]-o_bbox[0], padding=1)
    roll_img2 = roll_pad(img2, f_bbox[0]-o_bbox[0], padding=0)
    res1 = res*weight_seg1 + roll_img2*(1-weight_seg1)
    return res1


# %% part 2 of the deform: random deformation
def random_ffd():
    l = 5
    ffd = pg.FFD([l, l, l])
    ffd.box_length = np.full(3, 128-1)
    ffd.box_origin = np.zeros(3)
    # get random field
    x_field, y_field, z_field = (np.random.rand(3, l, l, l)-0.5) /(l-1)
    ffd.array_mu_x += x_field
    ffd.array_mu_y += y_field
    ffd.array_mu_z += z_field
    return ffd

def plt_ffd(ffd):
    fig = plt.figure(figsize=(4, 4))
    ax = Axes3D(fig)
    ax.scatter(*ffd.control_points(False).T, s=10)
    ax.scatter(*ffd.control_points().T, s=10, c='red')
    # plt_grid3d(ffd.control_points(False), ax, colors='blue')
    plt_grid3d(ffd.control_points(), ax, colors='red')
    plt.show()

#%%
# use h5
import h5py
fname = '/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5'
dct = h5py.File(fname, 'r')
id1 = '32'
# id2 = '129'
id2 = '129'
# id2 = '64'
img1 = dct[id1]['volume'][...]
img2 = dct[id2]['volume'][...]
seg1 = dct[id1]['segmentation'][...]
seg2 = dct[id2]['segmentation'][...]

kseg2, o_bbox, f_bbox, bbox = find_bbox(seg2)
p_ffd, paste_seg2 = presure_ffd(o_bbox, f_bbox, bbox, seg2)
res = do_ffd(p_ffd, img1)
feather_len = 5
# paste tumor
# res1 = direct_paste()
res1 = feathering_paste(feather_len, res, img2, kseg2, f_bbox, o_bbox)

dres = draw_3dbox(res1, f_bbox[0], f_bbox[1])
dimg2 = draw_3dbox(img2, o_bbox[0], o_bbox[1])
combine_pil_img(show_img(dres), show_img(dimg2), show_img(res1))
## %%
# show_img(distance_transform_edt(kseg2!=2) - distance_transform_edt(kseg2==2))
# show_img(sdm(kseg2, feather_len))
# show_img(weight_kseg2*img2)
#%%
r_ffd = random_ffd()
res2 = do_ffd(r_ffd, res1)
# plt_ffd(ffd)

# res_box = ffd.box_origin, ffd.box_origin+ffd.box_length
# d_res = draw_3dbox(res, *res_box, val=255)
# d_img2 = draw_3dbox(img2, *bbox, val=255)
# im = combine_pil_img(show_img(d_img2), show_img(d_res))
im = combine_pil_img(show_img(res2), show_img(res1))
im.save('ffd.png')
im
