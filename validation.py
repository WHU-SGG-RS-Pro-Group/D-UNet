from __future__ import print_function
import torch
import numpy as np
import scipy.io as sio
# import scipy.misc
import os
import h5py
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from PIL import Image
# from math import sqrt
import scipy.io as scio
import torch.nn.init
from common import *
from quality import *
from datasetload import *
if torch.cuda.is_available():
    torch.cuda.set_device(3)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sam = samLoss()
ssim=SSIM()
if __name__ == '__main__':
    # truth, ms_data = dataset('dc')
    data = my_dataset(r'.\Test_CAVE_10\hsi',
                      r'.\Test_CAVE_10\msi',
                      r'.\Test_CAVE_10\GT')
    data=DataLoader(data,batch_size=1,shuffle=True)
    upsampler = nn.Upsample(scale_factor=16, mode='bicubic')
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 
    if torch.cuda.is_available():
        MFN_3D = UNet(31, 3)

        model=MFN_3D.cuda()
   
        PATH = r'.\path\unet_2_epoch_199.pkl'
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['net'])
        for i, dataset in enumerate(data):
            hsi, msi, label = dataset
            lhs=upsampler(hsi)

            if torch.cuda.is_available():
                label=label.cuda()
                msi=msi.cuda()
                lhs=lhs.cuda()

            fake1,fake = MFN_3D(lhs, msi)
            print(sam(fake, label))

            pre = torch.squeeze(fake)
            hs = torch.squeeze(label)

            np_pre = pre.data.cpu().numpy()
            np_hs = hs.data.cpu().numpy()
            psnr(np_hs, np_pre)
            ssimloss(np_hs, np_pre)
            ergas(np_pre, np_hs)
            