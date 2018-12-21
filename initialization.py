from configobj import ConfigObj
import numpy as np
def init_params():
    
    '''
    tc : training_configuration
    '''
    # USER-DEFINED GLOBAL PARAMETERS
    tc = ConfigObj()
    tc.APPLICATION = 'classification'
    tc.WLENGTH, tc.ridx = 0.75e-3, 1.72499
    tc.M, tc.N = 512, 512
    tc.DX, tc.DY = 0.4e-3, 0.4e-3
    tc.MASK_WX, tc.MASK_WY = 8e-2, 8e-2
    tc.DATA_ROW, tc.DATA_COL = 28, 28
    tc.OBJ_WX, tc.OBJ_WY = 3.36e-2, 3.36e-2
    tc.SENSOR_WX, tc.SENSOR_WY = 4e-2, 4e-2
    tc.BEAM_WX, tc.BEAM_WY = 8e-2, 8e-2
    tc.OBJECT_PHASE_INPUT, tc.OBJECT_AMPLITUDE_INPUT = True, False  
    tc.MASK_PHASE_MODULATION, tc.MASK_AMPLITUDE_MODULATION = True, True
    tc.MASK_INIT_TYPE = 'const'
    tc.MASK_NUMBER = 5
    tc.OBJECT_MASK_DISTANCE, tc.MASK_MASK_DISTANCE, tc.MASK_SENSOR_DISTANCE = 3e-3, 3e-3, 3e-3
    tc.THETA0 = 75.0
    tc.DATA_ROW, tc.DATA_COL, tc.NUM_CLASS = 28, 28, 10
    tc.CLASS_SIZE_X, tc.CLASS_SIZE_Y, tc.CLASS_NUM_X, tc.CLASS_NUM_Y = 4.8e-3, 4.8e-3, 10, 10
    tc.NUM_HOLES = 10
    tc.TRAINING_DATA_TYPE, tc.TESTING_DATA_TYPE = 'fashion-mnist', 'fashion-mnist'
    tc.NUMBER_TRAINING_ELEMENTS, tc.NUMBER_TEST_ELEMENTS = 60000, 10000 
    tc.BATCH_SIZE, tc.TEST_BATCH_SIZE = 64, 20
    tc.LEARNING_RATE, tc.OPTIMIZER, tc.TV_LOSS_PARAM = 1e-3, 'adam', 0.0
    tc.MAX_EPOCH = 50
    tc.TFBOARD_PATH, tc.MASK_SAVING_PATH, tc.MASK_TESTING_PATH, tc.OUTPUT_PATH = '.\TFBOARD', \
    '.\MODEL\MASKS', '.\BEST_MODEL', '.\OUTPUT'
    tc.DATA_PATH = 'D:\Deniz\Python\Datasets'
    tc.validation_ratio = 0.5 # Ratio of number of elements in validation set to test set
    
    # COMPUTED GLOBAL PARAMETERS
    tc.WX = tc.M*tc.DX
    tc.WY = tc.N*tc.DY
    tc.OBJECT_ROW = int(np.round(tc.OBJ_WX/tc.DX))
    tc.OBJECT_COL = int(np.round(tc.OBJ_WY/tc.DY))
    tc.MASK_ROW = int(np.round(tc.MASK_WX/tc.DX))
    tc.MASK_COL = int(np.round(tc.MASK_WY/tc.DY))
    tc.SENSOR_ROW = int(np.round(tc.SENSOR_WX/tc.DX))
    tc.SENSOR_COL = int(np.round(tc.SENSOR_WY/tc.DY))
    tc.BEAM_ROW = int(np.round(tc.BEAM_WX/tc.DX))
    tc.BEAM_COL = int(np.round(tc.BEAM_WY/tc.DY))
    
    return tc