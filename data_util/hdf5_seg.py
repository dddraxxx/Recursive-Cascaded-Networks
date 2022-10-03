#%%
from nnunet.preprocessing.preprocessing import resample_data_or_seg
from adet.utils.visualize_niigz import *
import h5py
import SimpleITK as sitk
import numpy as np
import pickle as pkl
import pydicom

# read npz file
def get_npz(path):
    return np.load(path)['data']

# read nii file
def read_nii(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr

def read_dicom_itk(path):
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path))
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    img = reader.Execute()
    return img

def read_dicom(path):
    plots = []
    slices = [pydicom.dcmread(f) for f in pa(path).glob("*.dcm")]
    for ds in sorted(slices, key = lambda x: int(x.InstanceNumber), reverse = True):
        pix = ds.pixel_array
        pix = pix*1+(-1024)
        plots.append(pix)

    y = np.dstack(plots)
    return y
#%%
def normalize_ct(arr, property_f = '/mnt/sdb/nnUNet/nnUNet_cropped_data/Task029_LITS/dataset_properties.pkl'):
    # borrow from nnunet preprocessed stats; read pkl
    with open(property_f, 'rb') as f:
        it_prop = pkl.load(f)['intensityproperties'][0]
    # clip by it_prop bounds
    arr = np.clip(arr, it_prop['percentile_00_5'], it_prop['percentile_99_5'])
    # z-norm by it_prop mean and std
    arr = (arr - it_prop['mean']) / it_prop['sd']
    return arr

# get id and find seg file
def get_seg(id, root_dir = '/mnt/sdc/lits/train'):
    seg_id = f'segmentation-{id}.nii'
    seg_path = root_dir + '/' + seg_id
    return read_nii(seg_path)

# get id and find volume file
def get_data(id, root_dir = '/mnt/sdc/lits/train'):
    ct_id = f'volume-{id}.nii'
    ct_path = root_dir + '/' + ct_id
    return read_nii(ct_path)

# crop seg by the bounding box where value is larger than 0
def bbox_seg(seg):
    bbox = seg.nonzero()
    bbox = np.array([np.min(bbox[0]), np.min(bbox[1]), np.min(bbox[2]), np.max(bbox[0]), np.max(bbox[1]), np.max(bbox[2])])
    return bbox
def crop(seg, bbox, pad=5):
    # increase bbox by pad
    bbox[0] = max(bbox[0] - pad, 0)
    bbox[1] = max(bbox[1] - pad, 0)
    bbox[2] = max(bbox[2] - pad, 0)
    bbox[3] += pad
    bbox[4] += pad
    bbox[5] += pad
    return seg[bbox[0]:bbox[3]+1, bbox[1]:bbox[4]+1, bbox[2]:bbox[5]+1]

# resize seg
def resize_data(data, size, is_seg=False):
    # resize 
    data = resample_data_or_seg(data[None], [s for s in size], is_seg=is_seg)[0]
    # pad seg by 5 each side
    # data = np.pad(data, ((5, 5), (5, 5), (5, 5)), 'constant', constant_values=0)
    return data

def find_no_tumor_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        tumour_sum = []
        keys = list(f.keys())
        for id in sorted(keys, key=lambda x: int(x)):
            tumour_sum.append(np.sum(np.array(f[id]['segmentation'])==2))
            if np.max(f[id]['segmentation'])<2:
                print(id)
    print(np.argsort(tumour_sum))
    print(tumour_sum)

def visual_h5(h5_path):
    ids = range(130)
    dir = pa('images') / pa(h5_path).stem
    # mkdir
    dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, 'r') as f:
        for id in f:
                img = np.array(f[id]['volume'])
                visulize_3d(img, save_name = dir / f'{id}_img.jpg')
                seg = np.array(f[id]['segmentation'])
                visulize_3d(seg, save_name = dir / f'{id}_seg.jpg')

# visualize original ct data
def visual_nii(id, path = '/mnt/sdc/lits/train'):
    seg = get_seg(id, path)
    data = get_data(id, path)
    data = normalize_ct(data)
    # visulize_3d(data, save_name = f'images/{id}_img.jpg')
    # visulize_3d(seg, save_name = f'images/{id}_seg.jpg')
    # calculate centroid of mass
    obj = seg==2
    import scipy.ndimage as ndi
    centroid = ndi.measurements.center_of_mass(obj)
    centroid = np.array(centroid).astype(int)
    print(centroid)
    print(data.shape)

    radius = 96
    # crop
    c_data = data[int(centroid[0])-radius:int(centroid[0])+radius, int(centroid[1])-radius:int(centroid[1])+radius, int(centroid[2])-radius:int(centroid[2])+radius]
    # normalize
    c_data = (c_data - np.min(c_data)) / (np.max(c_data) - np.min(c_data))*255
    c_data = c_data.astype(np.uint8)
    c_seg = seg[int(centroid[0])-radius:int(centroid[0])+radius, int(centroid[1])-radius:int(centroid[1])+radius, int(centroid[2])-radius:int(centroid[2])+radius]
    # visulize_3d(c_data, save_name = f'images/{id}_c_img.jpg')
    from PIL import Image
    img_x = Image.fromarray(c_data[radius, :, :]).convert('L')
    img_y = Image.fromarray(c_data[:, radius, :]).convert('L')
    img_z = Image.fromarray(c_data[:, :, radius]).convert('L')
    img_x.save(f'images/{id}_c_img_x.jpg')
    img_y.save(f'images/{id}_c_img_y.jpg')
    img_z.save(f'images/{id}_c_img_z.jpg')

    cl_data = draw_seg_on_vol(c_data[None], (c_seg==2)[None]).permute(0,2,3,1).numpy()
    cl_data = (cl_data*255).astype(np.uint8)
    print(cl_data.shape)
    cl_img_x = Image.fromarray(cl_data[radius, :, :])
    cl_img_y = Image.fromarray(cl_data[:, radius, :])
    cl_img_z = Image.fromarray(cl_data[:, :, radius])
    cl_img_x.save(f'images/{id}_cl_img_x.jpg')
    cl_img_y.save(f'images/{id}_cl_img_y.jpg')
    cl_img_z.save(f'images/{id}_cl_img_z.jpg')

    print('make grid')
    # put imgs in grid together
    def make_grid_pil(*ims, nrow=3, padding=2):
        ncol = (nrow-1+len(ims)) // nrow
        ims = [im.convert('RGB') for im in ims]
        widths, heights = zip(*(i.size for i in ims))
        total_width = ncol * max(widths) + (ncol-1) * padding
        max_height = nrow * max(heights) + (nrow-1) * padding
        new_im = Image.new('RGB', (total_width, max_height), color='white')
        for i, im in enumerate(ims):
            col = i // nrow
            row = i % nrow
            x_offset = col * (max(widths) + padding)
            y_offset = row * (max(heights) + padding)
            new_im.paste(im, (x_offset, y_offset))
        return new_im
    im = make_grid_pil(cl_img_x, cl_img_y, cl_img_z, img_x, img_y, img_z, padding=5)
    im.save(f'images/{id}_grid.jpg')

# assign seg according to group id/segmentation in hdf5 file
def store_segs(h5_path):
    ids = range(131)
    # create h5 and put data and seg in it
    with h5py.File(h5_path, 'w') as f:
        for id in ids:
            print('id', id)
            seg = get_seg(id)
            data = get_data(id)
            data = normalize_ct(data)
            
            bbox = bbox_seg(seg)
            seg = crop(seg, bbox, pad=5)
            data = crop(data, bbox, pad=5)

            seg = resize_data(seg, (128, 128, 128), is_seg=True)
            data = resize_data(data, (128, 128, 128), is_seg=False)
            seg = seg.transpose((2, 1, 0))[::-1].copy()
            data = data.transpose((2, 1, 0))[::-1].copy()
            # visulize_3d(data, inter_dst=3, save_name = 'img.jpg')

            group = f'{id}'
            # create group
            f.create_group(group)
            # normalize volume to 0-255
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            data = data.astype(np.uint8)
            seg = seg.astype(np.uint8)
            # store data and seg as 'volume' and 'segmentation' dataset
            f[group].create_dataset('volume', data=data, dtype='uint8')
            f[group].create_dataset('segmentation', data=seg, dtype='uint8')

def visual_dicom(path = '/mnt/sdc/qhd/nbia/Duke-Breast-Cancer-MRI'):
    img_dir = 'images/breast'
    pa(img_dir).mkdir(parents=True, exist_ok=True)
    for id in range(1,922):
        dir = f'{path}/Breast_MRI_{id:03d}/'
        print(dir)
        try:
            dir = pa(dir).glob('*/*ax t1*').__next__()
        except Exception as e:
            print(e)
            continue
        img = read_dicom(dir).transpose(0,1,2)
        print(np.unique(img), img.shape)
        # img = normalize_ct(img)
        img = resize_data(img, (128, 128, 128), is_seg=False)
        visulize_3d(img, save_name = f'{img_dir}/{id}_img.jpg')
        print(img.shape)
        break

def store_dicom_h5(h5_path, path = '/mnt/sdc/qhd/nbia/Duke-Breast-Cancer-MRI'):
    def get_data_by_id(id):
        dir = f'{path}/Breast_MRI_{id:03d}/'
        try:
            dir = pa(dir).glob('*/*ax t1*').__next__()
        except Exception as e:
            print(e)
            return
        img = read_dicom(dir)
        return img
    ids = range(1,922)
    with h5py.File(h5_path, 'w') as f:
        for id in ids:
            print('id', id)
            data = get_data_by_id(id)
            if data is None:
                continue
            data = resize_data(data, (128, 128, 128), is_seg=False)
            data = data.transpose(2, 1, 0)[::-1].copy()
            if id%100==1:
                visulize_3d(data, inter_dst=2, save_name = 'img.jpg')

            group = f'{id}'
            # create group
            f.create_group(group)
            # normalize volume to 0-255
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            data = data.astype(np.uint8)
            # store data and seg as 'volume' and 'segmentation' dataset
            f[group].create_dataset('volume', data=data, dtype='uint8')
            f[group].create_dataset('segmentation', data=np.zeros((128,128,128)), dtype='uint8')

if __name__=='__main__':
    # store_segs('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5')
    # visual_h5('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits_bkp.h5')
    find_no_tumor_h5('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5')
    # visual_dicom()
    # store_dicom_h5('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/breast.h5')
    # visual_nii(118)