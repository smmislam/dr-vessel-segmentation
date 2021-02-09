# AVAILABLE MODES
# USE_PRE_TRAINED: LOAD PREVIOUSLY SAVED MODELS, NO NEW TRAINING
# RESUME_TRAINING: LOAD PREVIOUSLY SAVED MODELS, AND RESUME TRAINING
# RESET_MODEL: RESTART TRAINING FROM SCRATCH
MODEL_MODE = 'USE_PRE_TRAINED'

# PARAMETERS
RAND_SEED = 123


# TRAINING DATA GENERATOR PARAMETERS
WINDOW_WIDTH = 17  # Recommended window width is 17
N_EPOCHS = 100
BATCH_SIZE = 1000
VALIDATION_SPLIT = 0.3

# PREDICTION PARAMETERS
TEST_DATASET = 'DRIVE'  # Either 'DRIVE' or 'HRF'