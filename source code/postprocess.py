import numpy as np
import cv2
import copy


def postprocess_probability(binarizationThreshold, medianWindow, predictedImages, maskImages) :
    """
    Processes the probability based prediction images with median filtering and binary thresholding. Return the following-
    finalImages = Median filtered images
    binaryImages = Binary thresholed images
    """
    
    nImages, nCols, nRows = predictedImages.shape
    finalImages = np.zeros(predictedImages.shape, dtype=np.float64)
    binaryImages = np.zeros(predictedImages.shape, dtype=np.float64)
    
    for index in range(nImages):
        tempImg = np.float32(copy.deepcopy(predictedImages[index,:,:]))
        # Discard non-valid regions
        if len(maskImages.shape)==4 :
            validRegion = maskImages[index,:,:,0]>0.5
        elif len(maskImages.shape)==5 :
            validRegion = maskImages[index,:,:,0,0]>0.5
        else :
            raise ValueError("Un-supported dimension of mask images")
        tempImg = tempImg*np.float32(validRegion)
        # Apply median filtering
        medianImage = np.float64(cv2.medianBlur(tempImg,medianWindow))
        filteredImg = medianImage*tempImg
        finalImages[index] = filteredImg
        # Apply binary thresholding
        binaryImages[index, filteredImg>=binarizationThreshold] = 1.0
        
    return finalImages, binaryImages
