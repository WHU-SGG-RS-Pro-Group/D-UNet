import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from sklearn.metrics import mean_squared_error
import math

def psnr(x_true, x_pred):

    n_bands = x_true.shape[0]

    PSNR = np.zeros(n_bands)
    PSNR_ = np.zeros(n_bands+1)
    MSE = np.zeros(n_bands)
    MSE_ = np.zeros(n_bands+1)
    mask = np.ones(n_bands)

    x_true=x_true[:,:,:]

    for k in range(n_bands):

        x_true_k = x_true[k, :, :].reshape([-1])

        x_pred_k = x_pred[k, :, :].reshape([-1])

        MSE[k] = mean_squared_error(x_true_k, x_pred_k)

        x_true_k = np.array(x_true_k)
        MAX_k = np.max(x_true_k)

        if MAX_k != 0 :

            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])

            #print ('P', PSNR[k])

        else:

            mask[k] = 0

    psnr = PSNR.sum() / mask.sum()

    mse = MSE.mean()
    PSNR_[0] = psnr
    PSNR_[1:] = PSNR
    print('psnr', psnr)
    MSE_[0] = mse
    MSE_[1:] = MSE
    print('mse', mse)

    return PSNR_, MSE_

def ssimloss(x_true,x_pre):

    num=x_true.shape[0]

    ssimm=np.zeros(num)
    ssim_ = np.zeros(num+1)
    c1=0.0001

    c2=0.0009

    n=0

    for x in range(x_true.shape[0]):
        z = np.reshape(x_pre[x, :, :], [-1])

        sa = np.reshape(x_true[x, :, :], [-1])

        y=[z,sa]
        y = np.array(y)

        cov=np.cov(y)


        oz=cov[0,0]

        osa=cov[1,1]

        ozsa=cov[0,1]

        ez=np.mean(z)

        esa=np.mean(sa)

        ssimm[n]=((2*ez*esa+c1)*(2*ozsa+c2))/((ez*ez+esa*esa+c1)*(oz+osa+c2))

        n = n+1

    SSIM = np.mean(ssimm)
    ssim_[0] = SSIM
    ssim_[1:] = ssimm
    print('SSIM',SSIM)
    return ssim_

def sam(x_true,x_pre):

    print(x_pre.shape)

    print(x_true.shape)

    num = (x_true.shape[1]) * (x_true.shape[2])

    samm = np.zeros(num)

    n = 0

    for x in range(x_true.shape[1]):

        for y in range(x_true.shape[2]):

            z = np.reshape(x_pre[:, x, y], [-1])

            sa = np.reshape(x_true[:,x, y], [-1])

            tem1=np.dot(z,sa)

            tem2=(np.linalg.norm(z))*(np.linalg.norm(sa))

            samm[n]=np.arccos(tem1/tem2)

            n=n+1

    SAM=(np.mean(samm))*180/np.pi

    print('SAM',SAM)

def ergas(sr_img, gt_img, resize_factor=8):

    """Error relative global dimension de synthesis (ERGAS)

    reference: https://github.com/amteodoro/SA-PnP-GMM/blob/9e8dffab223d88642545d1760669e2326efe0973/Misc/ERGAS.m

    """

    sr_img = sr_img.astype(np.float64)

    gt_img = gt_img.astype(np.float64)

    err = sr_img - gt_img

    ergas = 0

    for i in range(err.shape[0]):

        ergas += np.mean(err[i, :, :]**2) / (np.mean(gt_img[i, :, :]))**2
    ergas = (100 / float(resize_factor)) * np.sqrt(1 / err.shape[0] * ergas)
    print(ergas)
    return ergas


def spatial(x, y):
    spatial1 = x[:,:,1:,:]-x[:,:,:-1,:]
    truth1 = y[:,:,1:,:]-y[:,:,:-1,:]
    spatial2 = x[:,:,:,1:]-x[:,:,:,:-1]
    truth2 = y[:,:,:,1:]-y[:,:,:,:-1]
    return spatial1,truth1,spatial2,truth2


def spectral(x, y):
    spectral1 = x[:,:,:,1:]-x[:,:,:,:-1]
    truth1 = y[:,:,:,1:]-y[:,:,:,:-1]
    return spectral1,truth1
def tv_loss(x, beta=0.5):
    '''Calculates TV loss for an image `x`.



    Args:

        x: image, torch.Variable of torch.Tensor

        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`

    '''

    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)

    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(torch.pow(dh[:, :, :, :-1] + dw[:, :, :, :-1], beta)/dh.size(0))




class samLoss(nn.Module):

    def __init__(self, tv_loss_weight=1):

        super(samLoss, self).__init__()

        self.tv_loss_weight = tv_loss_weight


    def forward(self, x, y):
        sam_ = 0

        for i in range(x.size()[0]):
            image1 = torch.squeeze(x[i, :, :, :])
            image2 = torch.squeeze(y[i, :, :, :])
            image1 = image1.view(-1, x.size()[1])

            image2 = image2.view(-1, x.size()[1])

            mole = torch.sum(image1.mul(image2), dim=1)

            image1_norm = torch.sqrt(torch.sum(image1.pow(2), dim=1))

            image2_norm = torch.sqrt(torch.sum(image2.pow(2), dim=1))

            deno = torch.mul(image1_norm, image2_norm)

            sam = (torch.acos((mole + 10e-12) / (deno + 10e-12))) * 180 / np.pi
            sam_ = torch.mean(sam) + sam_
        sam = sam_ / x.size()[0]
        # print(sam)
        return sam

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


#soble算子
def functional_conv2d(im):
    # im=torch.unsqueeze(im,dim=2)
    sobel_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]]).type(torch.float)#
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = sobel_kernel.cuda()
    for i in range(im.size()[1]):
        im[:,i:i+1,:,:] = F.conv2d(im[:,i:i+1,:,:], weight,padding=1)
    return im
# if __name__ == '__main__':
#     a = torch.rand(12, 31, 64, 64)
    # b = (torch.rand(12, 31, 64, 64) - torch.rand(12, 31, 64, 64))*0.1
    # c = sam_loss(a, b)
