import numpy as np
import os

import preprocess
from modelSegCNN import multi_scale_model
from utils import train_data_generator
from hparams import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model


train_fundus_dir = '../data/DRIVE/training/images'
train_manual_dir = '../data/DRIVE/training/1st_manual/'
train_mask_dir = '../data/DRIVE/training/mask/'

train_filenames = sorted(np.array(os.listdir(train_fundus_dir)))
manual_filenames = sorted(np.array(os.listdir(train_manual_dir)))
mask_filenames = sorted(np.array(os.listdir(train_mask_dir)))

all_filenames = np.vstack([train_filenames,manual_filenames, mask_filenames])

# Generate preprocessed dataset
print('Running Preprocess Routine...')
fundusImagesTrain, histEnhancedImagesTrain, entropyTsallisImagesTrain, stdImagesTrain, outputImagesTrain, maskImagesTrain = \
            preprocess.preprocess_fundus(train_fundus_dir, train_manual_dir, train_mask_dir, all_filenames)

# Generate training data
print('Generating training samples...')
windowWidth = WINDOW_WIDTH
blockedInputs, blockedOutputs = \
    train_data_generator(windowWidth, stdImagesTrain, entropyTsallisImagesTrain, histEnhancedImagesTrain, outputImagesTrain, maskImagesTrain)
    
# Define model
model_name = 'multi_scale_segCNN'
model_path = '../pretrained/'+ model_name + '.h5'
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', verbose=1,
                                 save_best_only=True, save_weights_only=False)
earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=10)

callback_list = [checkpoint, earlyStop]

# LOAD/COMPILE MODEL
if MODEL_MODE == 'USE_PRE_TRAINED' and os.path.isfile(model_path):
    model = load_model(model_path)

elif MODEL_MODE == 'RESUME_TRAINING' and os.path.isfile(model_path):
    model = load_model(model_path)
    # TRAIN
    model.fit(x=blockedInputs, y=blockedOutputs, callbacks=callback_list, epochs=N_EPOCHS, shuffle=True,
              batch_size=BATCH_SIZE, verbose=1, validation_split=VALIDATION_SPLIT)
    model.fit(x=blockedInputs, y=blockedOutputs, callbacks=callback_list, epochs=N_EPOCHS // 5, shuffle=True,
              batch_size=BATCH_SIZE * 10, verbose=1, validation_split=VALIDATION_SPLIT)

elif MODEL_MODE == 'RESET_MODEL' or not os.path.isfile(model_path):
    model = multi_scale_model(model_name, blockedInputs[0].shape)
    model.compile(Adam(lr=2e-3, decay=1e-8), loss='mse', metrics=['accuracy'])
    # TRAIN
    model.fit(x=blockedInputs, y=blockedOutputs, callbacks=callback_list, epochs=N_EPOCHS, shuffle=True,
              batch_size=BATCH_SIZE, verbose=1, validation_split=VALIDATION_SPLIT)
    model.fit(x=blockedInputs, y=blockedOutputs, callbacks=callback_list, epochs=N_EPOCHS // 5, shuffle=True,
              batch_size=BATCH_SIZE * 10, verbose=1, validation_split=VALIDATION_SPLIT)

else:
    raise NameError('Un-specified mode for model!')

# segCNN = multi_scale_model(model_name, blockedInputs[0].shape)
# segCNN.compile(Adam(lr=2e-3, decay=1e-8), loss='mse', metrics=['accuracy'])
#
# segCNN.fit(x=blockedInputs, y=blockedOutputs, epochs=100, shuffle=True, batch_size=1000, verbose=1)
# segCNN.fit(x=blockedInputs, y=blockedOutputs, epochs=20, shuffle=True, batch_size=10000, verbose=1)
#
# json_string = segCNN.to_json()
# with open('segCNN_model.json', 'w') as f:
#     f.write(json_string)
# segCNN.save_weights('segCNN_weights.h5')
