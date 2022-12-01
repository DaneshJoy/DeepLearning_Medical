import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os

from prep_funcs import window_intensities, rescale, ROI_from_mask
from prep_funcs import resize_image, write_image


base_path = r"D:\DeepLearning_IACT\medical_images\3Dircadb_20"
out_path = r"D:\DeepLearning_IACT\medical_images\3Dircadb_20_preprocessed"

if not os.path.exists(out_path):
    os.mkdir(out_path)

# Get file paths
images_paths = glob.glob(os.path.join(base_path, '*patient*'))
masks_paths = glob.glob(os.path.join(base_path, '*-liver*'))


# Read images
for img_path, msk_path in tqdm(zip(images_paths, masks_paths)):
    img = sitk.ReadImage(img_path)
    msk = sitk.ReadImage(msk_path)

    # preprocess
    newSize = (128, 128, 128)

    img_1 = window_intensities(img)
    msk_1 = rescale(msk)
    img_2, msk_2 = ROI_from_mask(img_1, msk_1)

    img_3 = resize_image(img_2, newSize)
    msk_3 = resize_image(msk_2, newSize)

    img_3 = sitk.GetImageFromArray(sitk.GetArrayFromImage(img_3))
    msk_3 = sitk.GetImageFromArray(sitk.GetArrayFromImage(msk_3))

    # # Visualization
    # img_arr = sitk.GetArrayFromImage(img)
    # img_1_arr = sitk.GetArrayFromImage(img_3)

    # fig, axes = plt.subplots(1,2)
    # axes[0].imshow(img_arr[70, : ,:], cmap='gray')
    # axes[0].set_axis_off()
    # axes[1].imshow(img_1_arr[70, : ,:], cmap='gray')
    # axes[1].set_axis_off()

    # Save

    img_path_new = os.path.basename(img_path).split(
        '.')[0] + '_prep.' + '.'.join(os.path.basename(img_path).split('.')[1:])
    msk_path_new = os.path.basename(msk_path).split(
        '.')[0] + '_prep.' + '.'.join(os.path.basename(msk_path).split('.')[1:])

    image_path = os.path.join(out_path, img_path_new)
    mask_path = os.path.join(out_path, msk_path_new)

    write_image(img_3, image_path)
    write_image(msk_3, mask_path)
