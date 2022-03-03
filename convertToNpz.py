#-*- coding:utf-8 -*-

import numpy as np
import nibabel as nib
import os

img = nib.load('./data/test_mr_raw/image_mr_1007.nii.gz')
label = nib.load('./data/test_mr_raw/gth_mr_1007.nii.gz')
print(img.shape)
print(label.shape)
#Convert them to numpy format,
data = img.get_fdata()
label_data = label.get_fdata()

np.savez('./data/test_mr/mr_1007.npz',data,label_data)
