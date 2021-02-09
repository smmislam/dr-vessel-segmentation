import numpy as np
import os

import preprocess
import postprocess
from keras.models import load_model

from utils import predict_vessel, roc_curve_generator, plot_ROC
from hparams import *

# Test Dataset Path
if TEST_DATASET=='DRIVE':
    test_fundus_dir = '../data/DRIVE/test/images'
    test_manual_dir = '../data/DRIVE/test/2nd_manual/'
    test_mask_dir = '../data/DRIVE/test/mask/'
elif TEST_DATASET=='HRF':
    test_fundus_dir = '../data/HRF/h_images'
    test_manual_dir = '../data/DRIVE/test/h_manual1/'
    test_mask_dir = '../data/DRIVE/test/h_mask/'
else:
    raise NameError('Unspecified DataSet!')

test_filenames = sorted(np.array(os.listdir(test_fundus_dir)))
manual_filenames = sorted(np.array(os.listdir(test_manual_dir)))
mask_filenames = sorted(np.array(os.listdir(test_mask_dir)))

all_filenames = np.vstack([test_filenames,manual_filenames, mask_filenames])

fundusImagesTest, histEnhancedImagesTest, entropyTsallisImagesTest, stdImagesTest, outputImagesTest, maskImagesTest = \
            preprocess.preprocess_fundus(test_fundus_dir, test_manual_dir, test_mask_dir, all_filenames)


# with open('segCNN_model.json', 'r') as f :
#     segCNN = model_from_json(f.read())
# segCNN.load_weights('segCNN_weights.h5')
model_name = 'multi_scale_segCNN'
model_path = '../pretrained/'+ model_name + '.h5'
segCNN = load_model(model_path)
predicted_images = predict_vessel(segCNN, WINDOW_WIDTH, stdImagesTest, entropyTsallisImagesTest, histEnhancedImagesTest)


# 0.45 for hrf, 0.5 for drive
binarizationThreshold = 0.5 if TEST_DATASET == 'DRIVE' else 0.45
medianWindow = 3

final_segmented_images, final_segmented_images_binary = \
        postprocess.postprocess_probability(binarizationThreshold, medianWindow, predicted_images, maskImagesTest)


thresholdRanges = np.hstack((np.array([0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]), \
                            np.arange(0.01,1.01,0.01)))
#thresholdRange = np.array([0.5])
roc_accuracy, roc_sensitivity, roc_specificity = \
            roc_curve_generator(medianWindow, predicted_images, outputImagesTest, maskImagesTest, thresholdRanges)


# ROC Curve points for line detection
if TEST_DATASET == 'DRIVE':
    xValLineDet = [0, 1e-7, 1e-6, 5e-3, 0.01, 0.02, 0.030, 0.04, 0.055, 0.07, 0.085, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    yValLineDet = [1e-4, 0.3, 0.4, 0.5, 0.6,  0.705,  0.776, 0.811, 0.85, 0.878, 0.895, 0.91, 0.935, 0.95, 0.97, 0.981, 0.987, 0.991, 0.993, 0.997, 1.0, 1.0]
elif TEST_DATASET == 'HRF':
    xValLineDet = [0, 1e-7, 1e-6, 5e-3, 0.01, 0.02, 0.037, 0.045, 0.06, 0.07, 0.085, 0.10, 0.125, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    yValLineDet = [1e-4, 0.3, 0.37, 0.42, 0.5,  0.6,  0.72, 0.75, 0.80, 0.82, 0.842, 0.863, 0.89, 0.905, 0.928, 0.96, 0.981, 0.987, 0.991, 0.993, 0.997, 1.0, 1.0]
else:
    raise NameError('Un-specified dataset for prediction!')

# Plotting the ROC curves
plot_ROC(roc_sensitivity, roc_specificity, xValLineDet, yValLineDet)
