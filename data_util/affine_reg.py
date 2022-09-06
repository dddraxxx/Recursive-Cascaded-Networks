import itk
import numpy as np

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

if __name__=='__main__':
    print()