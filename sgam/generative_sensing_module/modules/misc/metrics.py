import os
import sys
import math
import torch
import numpy as np
import cv2


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, visibiltiy_mask=None):
        mse = np.mean((img1 - img2) ** 2)
        if visibiltiy_mask is not None:
            visible_mse = ((img1 - img2) ** 2 * visibiltiy_mask).sum()/visibiltiy_mask.sum()
            return 20 * np.log10(255.0 / np.sqrt(mse)), 20 * np.log10(255.0 / np.sqrt(visible_mse)),
        else:
            return 20 * np.log10(255.0 / np.sqrt(mse))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2, visibility_mask=None):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return SSIM._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                if visibility_mask is not None:
                    visible_ssims = []
                for i in range(3):
                    if visibility_mask is not None:
                        ssim, visible_ssim = SSIM._ssim(img1[:, :, i], img2[:, :, i], visibility_mask[..., i])
                        ssims.append(ssim)
                        visible_ssims.append(visible_ssim)
                    else:
                        ssims.append(SSIM._ssim(img1, img2))
                if visibility_mask is not None:
                    return np.array(ssims).mean(), np.array(visible_ssims).mean()
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return SSIM._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2, visibility_mask=None):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if visibility_mask is not None:
            visibility_mask = visibility_mask[5:-5, 5:-5]
            return ssim_map.mean(), (ssim_map * visibility_mask).sum() / visibility_mask.sum()
        else:
            return ssim_map.mean()