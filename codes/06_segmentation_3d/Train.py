import os
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from UNet_3D import UNet_3D


# Load Data
print('Loading Dataset...')
dataset_path = r'D:\DeepLearning_IACT\medical_images\3Dircadb_20_preprocessed'
img_files = glob(os.path.join(dataset_path, '*patient*'))
msk_files = glob(os.path.join(dataset_path, '*-liver*'))

images = []
masks = []
for img_f, msk_f in tqdm(zip(img_files, msk_files)):
    img = sitk.ReadImage(img_f)
    msk = sitk.ReadImage(msk_f)
    images.append(sitk.GetArrayFromImage(img))
    masks.append(sitk.GetArrayFromImage(msk))

images = np.array(images, dtype='int32')
masks = np.array(masks, dtype='float32')

# Create Model
model = UNet_3D(img_size=images[0].shape)
model.save('model.h5')

# Callbacks

# Train
os.environ['CUDA_VISIBLE_DEVICES'] = -1
history = model.fit(images, masks, epochs=3)
