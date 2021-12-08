
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normalize
])


def default_loader(path,key):
    # img_pil = Image.open(path)
    # img_pil = img_pil.resize((224, 224))
    # img_tensor = preprocess(img_pil)
    img_pil = h5py.File(path)
    img_pil = np.array(img_pil[key])
    # plt.imshow(img_pil[1,:,:])
    # plt.show()
    img_tensor = torch.tensor(img_pil).type(torch.float)

    return img_tensor

# 当然出来的时候已经全都变成了tensor
# class trainset(Dataset):
#     def __init__(self, data1,data2,data3,loader=default_loader):
#         # 定义好 image 的路径
#         self.images = data1
#         self.aux = data2
#         self.target = data3
#         self.loader = loader
#
#     def __getitem__(self, index):
#         # fn = self.images[index]
#         img = self.loader(self.images,'hsi')
#         # target = self.target[index]
#         target = self.loader(self.target,'gt')
#         # aux = self.aux[index]
#         aux = self.loader(self.aux,'msi')
#         return img, aux, target
#
#     def __len__(self):
#         return len(self.images)


def trainset(data1,data2,data3,loader=default_loader):


    # fn = self.images[index]
    img = loader(data1,'hsi')
    # target = self.target[index]
    target = loader(data3,'gt')
    # aux = self.aux[index]
    aux = loader(data2,'msi')
    return img, aux, target



class my_dataset(Dataset):
    def __init__(self,img_path, mask_path, truth_path, data_transform=None):
        self.img_path = img_path  # 文件目录
        self.images = os.listdir(self.img_path)
        self.mask_path = mask_path  # 文件目录
        self.mask = os.listdir(self.mask_path)
        self.truth_path = truth_path  # 文件目录
        self.truth = os.listdir(self.truth_path)


    def __getitem__(self, item):
        image_index = self.images[item]  # 根据索引index获取该图片
        mask_index = self.mask[item]
        label_index = self.truth[item]
        img_path = os.path.join(self.img_path, image_index)
        mask_path = os.path.join(self.mask_path, mask_index)
        label_path = os.path.join(self.truth_path, label_index)
        # sample = {'image':img_path,'mask':mask_path, 'label':label_path}
        hsi,msi,label = trainset(img_path,mask_path,label_path)
        return hsi,msi,label
    def __len__(self):
        return len(self.images)

def filename(file_dir):
    data1 = []
    aux = []
    label = []
    J = 0
    for root, dir, files in os.walk(file_dir):
        J = J + 1
        if J==1:continue
        print(root)
        for file in files:
            if J==3:
                label.append(os.path.join(root, file))
            elif J==2:
                data1.append(os.path.join(root, file))
            else:aux.append(os.path.join(root, file))
    return data1,aux,label
