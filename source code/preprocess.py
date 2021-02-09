import os
from tqdm import tqdm
import matplotlib.image as mpimg
import cv2
import numpy as np

from utils import comp_tsallis_entropy

def preprocess_fundus(fundus_dir, manual_dir, mask_dir, all_filenames) :
    """
    Load fundus images and return the followings:
    inputImages = raw green channel of the fundus images
    histEnhancedImages = green channel enhanced by CLAHE
    stdImages = local standard deviation images
    entropyTsallisImages = local tsallis images
    outputImages = ground truth images
    maskImages = binary mask for the valid region in fundus images
    """
    for file_index in tqdm(range(all_filenames.shape[1])):
        fundus_image_path = os.path.join(fundus_dir, all_filenames[0,file_index])
        manual_image_path =  os.path.join(manual_dir, all_filenames[1,file_index])
        mask_image_path =  os.path.join(mask_dir, all_filenames[2,file_index])
        
        fundus_img = cv2.imread(fundus_image_path, 1)
        green_channel = fundus_img[:,:,1]
        
        green_channel[green_channel>180] = np.mean(green_channel[75:450,75:450])
        
        gt_img = mpimg.imread(manual_image_path)
        gt_img = gt_img[np.newaxis,:,:,np.newaxis]/255.0
        
        mask_img = mpimg.imread(mask_image_path)
        mask_img = mask_img[np.newaxis,:,:,np.newaxis]/255.0
        
        # This is for the std and entropy calculation
        paddingWidth = 2 
        # This is to remove the circular artifact around valid region
        maskPaddingWidth = 4
        paddedImg = np.pad(green_channel,paddingWidth,'edge')
        paddedMask = np.pad(mask_img[0,:,:,0],maskPaddingWidth,'edge')
    
        entropyImg = np.zeros((green_channel.shape[0], green_channel.shape[1]), dtype=float)
        entropyTsallisImg = np.zeros((green_channel.shape[0], green_channel.shape[1]), dtype=float)
        stdImg = np.zeros((green_channel.shape[0], green_channel.shape[1]), dtype=float)
        
        mask_index = np.zeros((green_channel.shape[0], green_channel.shape[1]), dtype=bool)
        
        for colIndex in range(paddingWidth,paddingWidth+green_channel.shape[0]):
            for rowIndex in range(paddingWidth,paddingWidth+green_channel.shape[1]):
                mask_region = paddedMask[colIndex-maskPaddingWidth:colIndex+maskPaddingWidth, \
                                         rowIndex-maskPaddingWidth:rowIndex+maskPaddingWidth]
                if (mask_region<0.5).any():
                    mask_index[colIndex-paddingWidth, rowIndex-paddingWidth] = True
                    entropyTsallisImg[colIndex-paddingWidth, rowIndex-paddingWidth] = 1000000
                    stdImg[colIndex-paddingWidth, rowIndex-paddingWidth] = 0.0
                else:
                    region = paddedImg[colIndex-paddingWidth:colIndex+paddingWidth, rowIndex-paddingWidth:rowIndex+paddingWidth]
                    entropyTsallisImg[colIndex-paddingWidth, rowIndex-paddingWidth] = comp_tsallis_entropy(region, -0.1)
                    if fundus_img[colIndex-paddingWidth, rowIndex-paddingWidth,1]>180 :
                        stdImg[colIndex-paddingWidth, rowIndex-paddingWidth] = 0.0
                    else :
                        stdImg[colIndex-paddingWidth, rowIndex-paddingWidth] = np.std(region)
        
        entropyTsallisImg[mask_index] = np.min(entropyTsallisImg)
        
        stdImg = stdImg - np.min(stdImg)
        stdImg = stdImg/np.max(stdImg[100:500, 200:300])
        stdImg[np.bitwise_and(stdImg>0.3*np.max(stdImg),stdImg>1.2)] = 0
        stdImg[stdImg>1.0] = 1.0
        stdImg = stdImg[np.newaxis,:,:,np.newaxis]
        
        entropyTsallisImg = entropyTsallisImg - np.min(entropyTsallisImg)
        entropyTsallisImg = entropyTsallisImg/np.max(entropyTsallisImg)
        entropyTsallisImg = entropyTsallisImg[np.newaxis,:,:,np.newaxis]
        
        clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(40,40))
        enhanced1 = clahe.apply(green_channel)
        enhanced1 = enhanced1 - np.min(enhanced1)
        enhanced1 = 1 - enhanced1/np.max(enhanced1)
        
        clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(80,80))
        enhanced2 = clahe.apply(green_channel)
        enhanced2 = enhanced2 - np.min(enhanced2)
        enhanced2 = 1 - enhanced2/np.max(enhanced2)
        
        enhanced3 = enhanced1*enhanced2
        enhanced3 = enhanced3[np.newaxis,:,:,np.newaxis]
        
        try:
            fundusImages = np.concatenate((fundusImages, fundus_img[np.newaxis,:,:,1,np.newaxis]),0)
            histEnhancedImages = np.concatenate((histEnhancedImages, enhanced3),0)
            stdImages = np.concatenate((stdImages, stdImg), 0)
            entropyTsallisImages = np.concatenate((entropyTsallisImages, entropyTsallisImg), 0)
            outputImages = np.concatenate((outputImages, gt_img), 0)
            maskImages = np.concatenate((maskImages, mask_img), 0)
        except:
            fundusImages = fundus_img[np.newaxis,:,:,1,np.newaxis]
            histEnhancedImages = enhanced3
            stdImages = stdImg
            entropyTsallisImages = entropyTsallisImg
            outputImages = gt_img
            maskImages = mask_img
            
    return fundusImages, histEnhancedImages, stdImages, entropyTsallisImages, outputImages, maskImages
