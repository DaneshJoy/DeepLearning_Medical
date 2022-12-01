import numpy as np
import SimpleITK as sitk


def window_intensities(image, _min=0, _max=255):
    out_image = sitk.IntensityWindowing(image, _min, _max)
    return out_image


def rescale(image, minimum=0, maximum=1):
    out_image = sitk.RescaleIntensity(image, minimum, maximum)
    return out_image


def ROI_from_mask(data_image, data_liver):
    _liverArray = sitk.GetArrayFromImage(data_liver)
    _imageArray = sitk.GetArrayFromImage(data_image)
    x, y, z = np.nonzero(_liverArray)
    pad = 10
    minX = max(1, min(x)-pad)
    minY = max(1, min(y)-pad)
    minZ = max(1, min(z)-pad)
    maxX = min(_liverArray.shape[0], max(x)+pad)
    maxY = min(_liverArray.shape[1], max(y)+pad)
    maxZ = min(_liverArray.shape[2], max(z)+pad)
    ROI_liver = _liverArray[minX:maxX, minY:maxY, minZ:maxZ]
    _liver = sitk.GetImageFromArray(ROI_liver)
    ROI_image = _imageArray[minX:maxX, minY:maxY, minZ:maxZ]
    _image = sitk.GetImageFromArray(ROI_image)

    return _image, _liver


def resize_image(img, newSize):
    inSpacing = np.round(img.GetSpacing(), 3)
    outSpacing = np.round(inSpacing * img.GetSize() / newSize, 3)

    filterResamp = sitk.ResampleImageFilter()
    filterResamp.SetSize(newSize)
    filterResamp.SetDefaultPixelValue(0)
    filterResamp.SetOutputDirection(img.GetDirection())
    filterResamp.SetOutputSpacing(outSpacing)
    filterResamp.SetOutputOrigin(img.GetOrigin())
    filterResamp.SetOutputPixelType(sitk.sitkInt16)
    filterResamp.SetInterpolator(sitk.sitkNearestNeighbor)
    img_resized = filterResamp.Execute(img)

    return img_resized


def write_image(image_data, image_path):
    sitk.WriteImage(image_data, image_path)
