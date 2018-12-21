# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 11:04:13 2018

@author: deepLearning505
"""

#import clc
#clc.clear_all()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mask_modulation_model as mmm
import initialization as init
import data_generation as dtg

dict = init.import_parameters('./PARAMETERS.txt')
APPLICATION = (dict['application'])
WLENGTH = float(dict['wavelength'])
REF_IDX = float(dict['ref_idx'])
M = int(dict['window_size_X'])
N = int(dict['window_size_Y'])
DX = float(dict['pixel_size_X'])
DY = float(dict['pixel_size_Y'])
MASK_WX = float(dict['mask_width_X'])
MASK_WY = float(dict['mask_width_Y'])
OBJECT_WX = float(dict['object_width_X'])
OBJECT_WY = float(dict['object_width_Y'])
SENSOR_WX = float(dict['sensor_width_X'])
SENSOR_WY = float(dict['sensor_width_Y'])
SENSOR_DX = float(dict['sensor_pixelsize_X'])
SENSOR_DY = float(dict['sensor_pixelsize_Y'])
OBJECT_PHASE_INPUT = bool(int(dict['object_phase_input']))
OBJECT_AMPLITUDE_INPUT = bool(int(dict['object_amplitude_input']))
MASK_PHASE_MODULATION = bool(int(dict['mask_phase_modulation']))
MASK_AMPLITUDE_MODULATION = bool(int(dict['mask_amplitude_modulation']))
MASK_INIT_TYPE = (dict['mask_init_type'])
MASK_PHASE_INIT_VALUE = float(dict['mask_phase_init_value'])
MASK_AMP_INIT_VALUE = float(dict['mask_amp_init_value'])
MASK_NUMBER = int(float(dict['mask_number']))
OBJECT_MASK_DISTANCE = float(dict['object_mask_distance'])
MASK_MASK_DISTANCE = float(dict['mask_mask_distance'])
MASK_SENSOR_DISTANCE = float(dict['mask_sensor_distance'])
NUM_CLASS = int(dict['number_of_classes'])
LABEL_CELL_to_SENSOR_X = float(dict['sensorplane_cell2sensor_X'])
LABEL_CELL_to_SENSOR_Y = float(dict['sensorplane_cell2sensor_Y'])
DATA_ROW = int(dict['number_of_rows_data'])
DATA_COL = int(dict['number_of_cols_data'])
NUM_HOLES = int(dict['number_of_holes'])

TRAINING_DATA_TYPE = (dict['training_data_type'])
TESTING_DATA_TYPE = (dict['testing_data_type'])
OBJECT_DATA_PATH = (dict['object_data_path'])
LEARNING_RATE = float(dict['learning_rate'])
MAX_STEPS = int(dict['max_steps'])
TV_WEIGHT = float(dict['total_variation_loss_param'])
BATCH_SIZE = int(dict['batch_size'])
TEST_STEPS = int(dict['test_steps'])
OPTIMIZER = (dict['optimizer'])
TENSORBOARD_PATH = (dict['tensorboard_path'])
MASK_SAVING_PATH = (dict['mask_saving_path'])
MASK_TESTING_PATH = (dict['mask_testing_path']) #!!!!! THE ONLY DIFFERENCE from TRAINING PARAMS
SENSOR_SAVING_PATH = (dict['sensor_saving_path'])
INPUT_SAVING_PATH = (dict['input_saving_path'])

WX = M*DX
WY = N*DY
WINDOW_PIXEL_NUM = M*N
OBJECT_ROW = int(np.floor(OBJECT_WX/DX))
OBJECT_COL = int(np.floor(OBJECT_WY/DY))
OBJECT_PIXEL_NUM = OBJECT_ROW*OBJECT_COL
DATA_PIXEL_NUM = DATA_ROW*DATA_COL
MASK_ROW = int(np.floor(MASK_WX/DX))
MASK_COL = int(np.floor(MASK_WY/DY))
MASK_PIXEL_NUM = MASK_ROW * MASK_COL
SENSOR_ROW = int(np.floor(SENSOR_WX/SENSOR_DX))
SENSOR_COL = int(np.floor(SENSOR_WY/SENSOR_DY))
SENSOR_PIXEL_NUM = SENSOR_ROW*SENSOR_COL
BATCH_SHAPE = [BATCH_SIZE, M, N]

cell_locs,celltype1,celltype2 = dtg.sensorplane_geometry()
    
if __name__ == '__main__':

    save_mask_phase, save_mask_amp, save_mask_holes = mmm.read_mask(MASK_TESTING_PATH)
    #save_mask_amp[save_mask_amp == 0] = 0.031

    placeholder_input = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, WINDOW_PIXEL_NUM))

    onn_measurement = mmm.inference_testing(placeholder_input, save_mask_phase, save_mask_amp, save_mask_holes)

#    tf.gfile.MakeDirs(SENSOR_SAVING_PATH)

    sess = tf.InteractiveSession()

    count = 0
    
    for step in range(MAX_STEPS):

        testing_input, testing_gt, testing_gt_1 = dtg.generate_data('testing')
        onn_measurement_value_test = sess.run(onn_measurement, feed_dict={placeholder_input: testing_input})

        save_measurement = np.reshape(onn_measurement_value_test, (M, N))
        if(OBJECT_AMPLITUDE_INPUT):
            
            save_input = np.reshape(np.real(testing_input), (M, N))
            
        else:
            
            save_input = (np.reshape(np.angle(testing_input), (M, N))+np.pi)/2/np.pi
            
        save_input = save_input[M//2-OBJECT_ROW//2:M//2-OBJECT_ROW//2+OBJECT_ROW,\
                                N//2-OBJECT_COL//2:N//2-OBJECT_COL//2+OBJECT_COL]
        plt.imsave(SENSOR_SAVING_PATH + "/test_intensity_" + str(step) + ".bmp", save_measurement, cmap='gray')
        plt.imsave(INPUT_SAVING_PATH + "/test_intensity_" + str(step) + ".bmp", save_input, cmap='gray')
        #COMPARISON AGAINST OTHER CLASS LABELS
        save_measurement = save_measurement[M//2-SENSOR_ROW//2:M//2-SENSOR_ROW//2+SENSOR_ROW,\
                                            N//2-SENSOR_COL//2:N//2-SENSOR_COL//2+SENSOR_COL]
        class_probs = np.zeros(NUM_CLASS)
        for kk in range(NUM_CLASS):
            ulcorner = cell_locs[kk,0:2]
            ctype = cell_locs[kk,2]    
            if(ctype==1):
                class_probs[kk] = np.mean(save_measurement[ulcorner[0]:ulcorner[0]+celltype1.shape[0],\
                                          ulcorner[1]:ulcorner[1]+celltype1.shape[1]])
            else:
                class_probs[kk] = np.mean(save_measurement[ulcorner[0]:ulcorner[0]+celltype2.shape[0],\
                                          ulcorner[1]:ulcorner[1]+celltype2.shape[1]])
                
        if(np.argmax(class_probs)==np.argmax(testing_gt_1)):
            count = count+1
#            print('count is',count)
        
    accuracy = count / MAX_STEPS
    print('accuracy is', accuracy)    