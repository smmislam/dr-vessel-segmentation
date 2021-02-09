import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import postprocess


def comp_entropy(region) :
    """
    Computes shannon entropy
    """
    region = np.reshape(region, (1,-1))
    hist, hist_edges = np.histogram(region, np.arange(0,256,5))
    hist = hist + 1e-15
    hist = hist/max(hist)
    return -sum(hist*np.log(hist))


def comp_tsallis_entropy(region, q) :
    """
    Computes tsallis entropy with constants, K=1, q=-0.05
    """
    region = np.reshape(region, (1,-1))
    hist, hist_edges = np.histogram(region, np.arange(0,256,5))
    hist = hist + 1e-15
    hist = hist/max(hist)
    hist = hist**q
    return 1*(1-sum(hist))


def train_data_generator(windowWidth, stdImages, entropyTsallisImages, histEnhancedImages, outputImages, maskImages) :
    """
    Takes the features images and returns the followings-
    blockedInputs = windowWidth x windowWidth x 3 vectors around each pixel (17x17 blocks are taken from each feature images)
    blockedOutputs = 1x2 vectors for each pixel, boolean representation of the type of each pixel [background pixel, vessel pixel]
    """
    
    nImages, nCols, nRows, nChannels = entropyTsallisImages.shape
    # Only original version is used. But, rotation, translation, zoom-in, zoom-out, random-crop could've been used
    nVariations = 1
    paddingWidth = np.int(np.floor(windowWidth/2.0))
    
    # Approximate number of blocks
    nBlocks = np.int(4.1*np.sum(outputImages)*nVariations)
    # Approximate memory allocation for blocked input and output
    blockedInputs = np.zeros((nBlocks,windowWidth,windowWidth,3), dtype=np.float16) # 3 different types of info (shannon, tsallis, edge)
    blockedOutputs = np.zeros((nBlocks,2))
    
    blockIndex = 0
    bgPixelCounter = 0
    vesselPixelCounter = 0
    
    for imageIndex in tqdm(range(nImages)):
        paddedStd = np.pad(stdImages[imageIndex,:,:,0],paddingWidth,'edge')
        paddedTsallis = np.pad(entropyTsallisImages[imageIndex,:,:,0],paddingWidth,'edge')
        paddedEnhanced = np.pad(histEnhancedImages[imageIndex,:,:,0],paddingWidth,'edge')
        for colIndex in range(0,nCols):
            for rowIndex in range(0,nRows):
                
                # Skip non-valid regions
                if maskImages[imageIndex,colIndex,rowIndex,:] < 0.5:
                    continue
                
                # Binarize the output pixel
                currentOutputPixel = outputImages[imageIndex, colIndex, rowIndex, :]
                if currentOutputPixel<0.5:
                    # Background Pixel
                    currentOutputBlock = np.array([1.0, 0.0])
                    bgPixelCounter += 1 
                    bgPixel = True
                else:
                    # Vessel Pixel
                    currentOutputBlock = np.array([0.0, 1.0])
                    vesselPixelCounter += 1
                    bgPixel = False
                    
                currentOutputBlock = currentOutputBlock[np.newaxis,:]
                
                # Maintain a 3:1 ratio of background pixel and vessel pixel
                if bgPixel and (bgPixelCounter>3*vesselPixelCounter):
                    bgPixelCounter -= 1
                    continue
                
                originalVersion = np.empty([])
                rotatedVersion = np.empty([])
                # Standard deviation block
                currentStdBlock = paddedStd[colIndex:colIndex+windowWidth, rowIndex:rowIndex+windowWidth]
                originalVersion = currentStdBlock[np.newaxis,:,:,np.newaxis]
                # Tsallis block
                currentTsallisBlock = paddedTsallis[colIndex:colIndex+windowWidth, rowIndex:rowIndex+windowWidth]
                originalVersion = np.concatenate((originalVersion, currentTsallisBlock[np.newaxis,:,:,np.newaxis]), 3)
                # CLAHE block
                currentEnhancedBlock = paddedEnhanced[colIndex:colIndex+windowWidth, rowIndex:rowIndex+windowWidth]
                originalVersion = np.concatenate((originalVersion, currentEnhancedBlock[np.newaxis,:,:,np.newaxis]), 3)
                
                # Append blocks
                blockedInputs[blockIndex,:,:,:] = np.float16(originalVersion)
                blockedOutputs[blockIndex,:] = np.float16(currentOutputBlock)
                blockIndex = blockIndex+1
    
    # Discard un-used last part
    blockedInputs = blockedInputs[0:blockIndex,:,:,:]
    blockedOutputs = blockedOutputs[0:blockIndex,:,]
    
    return blockedInputs, blockedOutputs


def predict_vessel(segCNN, windowWidth, stdImages, entropyTsallisImages, histEnhancedImages) :
    """
    Predict probability of each pixel being associated to background or vessel, using the trained model 
    """
    
    paddingWidth = np.int(np.floor(windowWidth/2.0))

    nImages, nCols, nRows, nChannels = entropyTsallisImages.shape
    predictedImages = np.zeros(entropyTsallisImages.shape[:-1], dtype=np.float64)
    
    for imageIndex in tqdm(range(nImages)):
        blockIndex = 0
        currentBlockedImage = np.zeros((entropyTsallisImages.shape[1]*entropyTsallisImages.shape[2],windowWidth,windowWidth,3), dtype=np.float16)
        paddedStd = np.pad(stdImages[imageIndex,:,:,0],paddingWidth,'edge')
        paddedTsallis = np.pad(entropyTsallisImages[imageIndex,:,:,0],paddingWidth,'edge')
        paddedEnhanced = np.pad(histEnhancedImages[imageIndex,:,:,0],paddingWidth,'edge')
        for colIndex in range(nCols):
            for rowIndex in range(nRows):
                currentBlockedImage[blockIndex,:,:,0] = paddedStd[colIndex:colIndex+windowWidth, rowIndex:rowIndex+windowWidth]
                currentBlockedImage[blockIndex,:,:,1] = paddedTsallis[colIndex:colIndex+windowWidth, rowIndex:rowIndex+windowWidth]
                currentBlockedImage[blockIndex,:,:,2] = paddedEnhanced[colIndex:colIndex+windowWidth, rowIndex:rowIndex+windowWidth]
                blockIndex+=1
        predictedImages[imageIndex,:,:] = np.reshape(np.float64(segCNN.predict(currentBlockedImage))[:,1], (entropyTsallisImages.shape[1],entropyTsallisImages.shape[2]))
        
    return predictedImages


def roc_calc(filteredImg, validRegion, outputImg):
    """
    Computes accuracy, sensitivity, specificity
    """
    predictedPos = np.logical_and(filteredImg>=0.5, validRegion)
    predictedNeg = np.logical_and(filteredImg<0.5, validRegion)
    actualPos = np.logical_and(outputImg>=0.5, validRegion)
    actualNeg = np.logical_and(outputImg<0.5, validRegion)

    truePos = np.sum(np.logical_and(predictedPos, actualPos))
    trueNeg = np.sum(np.logical_and(predictedNeg, actualNeg))
    falsePos = np.sum(np.logical_and(predictedPos, actualNeg))
    falseNeg = np.sum(np.logical_and(predictedNeg, actualPos))

    accuracy = (truePos+trueNeg)/(truePos+trueNeg+falsePos+falseNeg)
    sensitivity = truePos/(truePos+falseNeg)
    specificity = trueNeg/(trueNeg+falsePos)
    
    return accuracy, sensitivity, specificity


def roc_curve_generator(medianWindow, predicted_images, outputImages, maskImages, thresholdRanges) :
    """
    Computes the ROC metrics of the predicted images. Return the following-
    finalImages = Median filtered images
    binaryImages = Binary thresholed images
    """

    nImages = outputImages.shape[0]
    
    accuracy = np.zeros((nImages,1),dtype=np.float32)
    sensitivity = np.zeros((nImages,1),dtype=np.float32)
    specificity = np.zeros((nImages,1),dtype=np.float32)
    
    roc_accuracy = np.zeros((thresholdRanges.shape[0],1),dtype=np.float32)
    roc_sensitivity = np.zeros((thresholdRanges.shape[0],1),dtype=np.float32)
    roc_specificity = np.zeros((thresholdRanges.shape[0],1),dtype=np.float32)
    
    for rocIndex, threshold in enumerate(thresholdRanges) :
        final_segmented_images, final_segmented_images_binary = \
            postprocess.postprocess_probability(threshold, medianWindow, predicted_images, maskImages)
        for index in range(nImages):
            if len(outputImages.shape)==4 :
                outputImg = outputImages[index,:,:,0]
            elif len(outputImages.shape)==5 :
                outputImg = outputImages[index,:,:,0,1]
            else :
                raise ValueError("Un-supported dimension of output images")
                
            if len(maskImages.shape)==4 :
                validRegion = maskImages[index,:,:,0]>0.5
            elif len(maskImages.shape)==5 :
                validRegion = maskImages[index,:,:,0,0]>0.5
            else :
                raise ValueError("Un-supported dimension of mask images")
            
            filteredImg = final_segmented_images_binary[index,:,:]
            accuracy[index], sensitivity[index], specificity[index] = roc_calc(filteredImg, validRegion, outputImg)
        roc_accuracy[rocIndex] = np.mean(accuracy)   
        roc_sensitivity[rocIndex] = np.mean(sensitivity)
        roc_specificity[rocIndex] = np.mean(specificity)
    
    return roc_accuracy, roc_sensitivity, roc_specificity

    
def show_images(list_of_images) :
    for index, img in enumerate(list_of_images) :
        img = np.float64(img/np.max(img))
        try :
            img = img[:,:,0]
        except:
            pass
        imgTitle = 'image_'+str(index)
        cv2.imshow(imgTitle, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_ROC(roc_sensitivity, roc_specificity, xValLineDet=None, yValLineDet=None) :
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111)

    # Plot proposed models ROC
    ax1.plot(1-roc_specificity, roc_sensitivity, label='Multi-Scale CNN', color='blue')   
    
    # Plot ROC for line detection
    if (xValLineDet is not None) and (yValLineDet is not None) :
        ax1.plot(xValLineDet, yValLineDet, label='Line Detector', color='red', linestyle='-.')

    plt.xticks(np.arange(0, 1.05, step=0.1), fontsize = 12)
    plt.yticks(np.arange(0, 1.05, step=0.1), fontsize = 12)
    plt.xlabel('1-Specificity (False Postive Rate)', fontsize = 24)
    plt.ylabel('Sensitivity (True Positve Rate)', fontsize = 24)
    plt.legend(loc='lower right', fontsize = 24)
    plt.grid(linestyle='-.')

    plt.show()

    return True
