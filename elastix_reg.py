#%%
import numpy as np
import pickle as pkl
import SimpleITK as sitk
from skimage.transform import AffineTransform
from skimage.transform import warp
from adet.utils.visualize_niigz import *
import itk

fname = '/home/hynx/regis/Recursive-Cascaded-Networks/evaluate/Aug23-1307-model-99500-slits-val.pkl'
dct = pkl.load(open(fname, 'rb'))

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
    return elastixImageFilter
def itk_affine(arr1, arr2):
    import itk
    fixed_image = itk.GetImageViewFromArray(arr1)
    moving_image = itk.GetImageViewFromArray(arr2)
    parameter_object = itk.ParameterObject.New()
    affine_parameter_map = parameter_object.GetDefaultParameterMap('affine', 4)
    # affine_parameter_map['CenterOfRotationPoint'] = ['0', '0', '0']
    parameter_object.AddParameterMap(affine_parameter_map)

    # Call registration function
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image, parameter_object=parameter_object)
    return result_transform_parameters

def show_img(res):
    import torchvision.transforms as T
    res = tt(res)
    if res.ndim>=3:
        return T.ToPILImage()(visulize_3d(res))
    return T.ToPILImage()(res)

idx = 10
mode = 'mask'
dir_name = f'{mode}_pair{idx}'
(pa('images')/dir_name).mkdir(exist_ok=True)
img1 = dct['img1'][idx,...,0]
img2 = dct['img2'][idx,...,0]
seg2 = dct['seg2'][idx,...,0]
#%%
def elastix():
    filter = elastix_affine(img1, img2)
    tx = filter.GetTransformParameterMap()
    sitk.PrintParameterMap(tx)
    # tx[0]['FinalBSplineInterpolationOrder'] = ['0']
    params = np.array(tx[0]['TransformParameters'], dtype=np.float32).reshape(4,3)
    m_seg2 = sitk.GetArrayFromImage(sitk.Transformix(sitk.GetImageFromArray(img2), tx))
    return m_seg2, tx[0]

def itke():
    params = itk_affine(img1, img2)
    parameter_map = params.GetParameterMap(0)
    # parameter_map['CenterOfRotationPoint'] = ['0', '0', '0']
    # parameter_map['DefaultPixelValue'] = '2'
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(parameter_map)
    print(parameter_object)
    m_seg2 = itk.transformix_filter(seg2, parameter_object)
    return m_seg2, parameter_map
    
m_seg2, tx = itke()
visulize_3d(m_seg2, save_name=f'images/{dir_name}/m_img2_itk2.jpg')
#%%
# [(k,tx[k]) for k in tx.keys()]
#%%
###### Try to use affine in python ########
params = np.array(tx['TransformParameters'], dtype=np.float32).reshape(4,3)
W, b = params[:3], params[-1]
o = np.array(tx['CenterOfRotationPoint'], dtype=np.float32)
print(W,b,o)

def get_total_matrix(W, b, o):
    center = np.eye(4)
    center[:3, 3] = -o
    center_inv = center.copy()
    center_inv[:3, 3] = o
    W = (W.T)[::-1,::-1].T
    b = b[::-1]
    T = np.append(W, b[None].T, axis=1)
    T = np.append(T, [[0, 0, 0, 1]], axis=0)
    print(center,'\n', T)
    return center_inv@T@center
total_matrix = get_total_matrix(W, b, o)
def scipy_affine(seg2, total_matrix):
    from scipy.ndimage import affine_transform
    res = affine_transform(seg2, total_matrix, order=3, mode='constant', cval=0)
    return res
def monai_affine(seg2, total_matrix):
    from monai.transforms import Affine
    affine_tr = Affine(affine=total_matrix, mode='bilinear', image_only=False, padding_mode='border')
    res, aff = affine_tr(seg2[None])
    print(aff)
    return res[0]
# monai_res = monai_affine(seg2, total_matrix)
# scipy_res = scipy_affine(seg2, total_matrix)
# show_img(scipy_res)
#%%
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
coord = affine_flow(tW,tb,*seg2.shape)

coord = coord.transpose(-1, 0, 1, 2)
from scipy.ndimage import map_coordinates
res = map_coordinates(seg2, coord, order=1, mode='constant', cval=0)>0.5
show_img(res).save(f'images/{dir_name}/m_img2_itk2_affine.jpg')
