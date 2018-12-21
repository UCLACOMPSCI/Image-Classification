'''
Mask Modulation Model
'''

import tensorflow as tf
import os
import numpy as np
import tf_OpticsModule as tom
import initialization as init

tc = init.init_params()
     
def leakyRelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def read_mask(path, L, mask_type):
    
    '''
    path : Path to the folder containing the trained masks to be loaded
    L : Layer number OR Mask number from 0 to MASK_NUMBER-1
    '''
    mask_list = os.listdir(path)
    if mask_type == 'amplitude':
        mask = np.loadtxt(path + "/" + mask_list[L])
    elif mask_type == 'phase':
        mask = np.loadtxt(path + "/" + mask_list[tc.MASK_NUMBER+L])
    mask = tf.cast(mask, tf.float32)
    return mask

def mask_init(masknum):

    if tc.MASK_PHASE_MODULATION is True:
        if tc.MASK_INIT_TYPE == 'const':
            mask_phase_org = tf.get_variable('mask_phase' + str(masknum), initializer=tf.constant(1.0, shape=[tc.MASK_ROW, tc.MASK_COL]))
            mask_phase = mask_phase_org * 2 * np.pi
        elif tc.MASK_INIT_TYPE == 'random':
            mask_phase_org = tf.get_variable('mask_phase' + str(masknum), initializer=tf.random_uniform(shape=[tc.MASK_ROW, tc.MASK_COL],minval=0.5,maxval=1.5))
            mask_phase = mask_phase_org * 2 * np.pi
        elif tc.MASK_INIT_TYPE == 'trained':
            mask_phase = read_mask(tc.MASK_PATH, masknum, 'phase')
    else:
        mask_phase = tf.zeros([tc.MASK_ROW, tc.MASK_COL])
        
    if tc.MASK_AMPLITUDE_MODULATION is True:
        if tc.MASK_INIT_TYPE == 'trained':
            mask_amp = read_mask(tc.MASK_PATH, masknum, 'amplitude')
        else:
            mask_amp_org = tf.get_variable('mask_amp' + str(masknum), initializer=tf.constant(1.0, shape=[tc.MASK_ROW, tc.MASK_COL]))    
            mask_amp = tf.nn.relu(mask_amp_org)
    else:
        mask_amp = tf.ones([tc.MASK_ROW, tc.MASK_COL])
        
    pad_X = int(tc.M//2-tc.MASK_ROW//2)
    pad_Y = int(tc.N//2-tc.MASK_COL//2)
    paddings = tf.constant([[pad_X,pad_X],[pad_Y,pad_Y]])
    mask_phase = tf.pad(mask_phase,paddings)
    mask_amp = tf.pad(mask_amp,paddings)
    
    return mask_phase, mask_amp

def detector_plane(measurement,cell_locs,thecell):

    cell_locs[:,0] = cell_locs[:,0]+tc.M//2-tc.SENSOR_ROW//2
    cell_locs[:,1] = cell_locs[:,1]+tc.N//2-tc.SENSOR_COL//2
    kk = 0
    ulcorner = cell_locs[kk,0:2]
    probs = tf.reduce_mean(tf.slice(measurement,[0,ulcorner[0],ulcorner[1]],[tc.BATCH_SIZE,thecell.shape[0],thecell.shape[1]]),axis=[1,2])
    class_probs = tf.expand_dims(probs,-1)
    for kk in range(1,cell_locs.shape[0]):
        ulcorner = cell_locs[kk,0:2]
        probs = tf.reduce_mean(tf.slice(measurement,[0,ulcorner[0],ulcorner[1]],[tc.BATCH_SIZE,thecell.shape[0],thecell.shape[1]]),axis=[1,2])
        probs = tf.expand_dims(probs,-1)
        class_probs = tf.concat([class_probs,probs],axis=1)

    # leakage = tf.divide(tf.reduce_mean(class_probs,axis=1),tf.reduce_mean(measurement,axis=[1,2]))
    E = tf.expand_dims(tf.reduce_mean(measurement,axis=[1,2]), -1)
    S = tf.expand_dims(tf.reduce_mean(class_probs,axis=1), -1)
    C = tf.divide(E-S, E)
    S = tf.divide(S, E)
    leakage = tf.concat([C,S],axis=1)
    return class_probs, leakage

def inference(field):

    # First Layer
    with tf.name_scope('hidden1'):

        with tf.name_scope('propagation'):
            img_p = tom.batch_propagate(field, tc.WLENGTH, tc.OBJECT_MASK_DISTANCE, tc.DX, tc.DY, 1.0, tc.THETA0)

        with tf.name_scope('mask'):
            mask_phase, mask_amp = mask_init(0)
            mask_amp = tf.divide(mask_amp,tf.reduce_max(tf.abs(mask_amp)))
            #------------------------------------------------------------------
            mask = tf.complex(mask_amp * tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
            save_mask_phase = tf.expand_dims(mask_phase,0)
            save_mask_amp = tf.expand_dims(mask_amp,0)
            
        hidden = tf.multiply(img_p, mask)
        
    # Middle Layers
    for layer_num in range(2, tc.MASK_NUMBER + 1):
        with tf.name_scope('hidden' + str(layer_num)):

            with tf.name_scope('propagation'):
                img_p = tom.batch_propagate(hidden, tc.WLENGTH, tc.MASK_MASK_DISTANCE, tc.DX, tc.DY, 1.0, tc.THETA0)

            with tf.name_scope('mask'):
                mask_phase, mask_amp = mask_init(layer_num-1)
                mask_amp = tf.divide(mask_amp,tf.reduce_max(tf.abs(mask_amp)))
                #------------------------------------------------------------------
                mask = tf.complex(mask_amp* tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
                save_mask_phase = tf.concat([save_mask_phase, tf.expand_dims(mask_phase,0)], 0)
                save_mask_amp = tf.concat([save_mask_amp, tf.expand_dims(mask_amp,0)], 0)
                
            hidden = tf.multiply(img_p, mask)

    # Last Layer
    with tf.name_scope('last'):
            
        with tf.name_scope('propagation'):
            img_p = tom.batch_propagate(hidden, tc.WLENGTH, tc.MASK_SENSOR_DISTANCE, tc.DX, tc.DY, 1.0, tc.THETA0)

        with tf.name_scope('sensor'):
            measurement = tf.square(tf.abs(img_p))

    with tf.name_scope('Hybrid'):

        measurement_cnn = tf.reshape(measurement, [tc.BATCH_SIZE, tc.M, tc.N])
        measurement_cnn = tf.layers.batch_normalization(measurement_cnn)
        roi = tf.slice(measurement_cnn,[0,tc.M//2-100//2,tc.N//2-100//2],[tc.BATCH_SIZE,100,100])
        sensor_mean, sensor_variance = tf.nn.moments(roi,axes=[1,2],keep_dims=True)
        roi = tf.divide(roi-sensor_mean,tf.sqrt(sensor_variance))
        # measurement_cnn = tf.reshape(measurement, [BATCH_SIZE, MASK_COL, MASK_ROW])
        measurement_cnn = tf.expand_dims(roi,-1)
        measurement_cnn = tf.nn.avg_pool(value=measurement_cnn, ksize=[1,10,10,1], strides=[1,10,10,1], padding='VALID')
        fc0 = tf.layers.flatten(measurement_cnn)
        fc0 = tf.layers.dropout(fc0, training='train'==tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=fc0, units=10)
            # logits, sensor_leakage = detector_plane(measurement,cell_locs,thecell)
            
    return measurement, save_mask_phase, save_mask_amp, logits

def loss_function(self, ground_truth, object_data):

#    #BUCKET LOSS
#    self = tf.divide(self,tf.reduce_sum(self,-1,keep_dims=True))
#    self_64 = tf.cast(self, dtype=tf.float64)
#    loss_ = tf.reduce_mean(tf.square(1-tf.reduce_sum(tf.multiply(self_64,ground_truth),-1)))
    
#    NORMALIZED MSE LOSS
#     corr = tf.reduce_sum(tf.multiply(self,ground_truth))
#     autoCorrIn = tf.reduce_sum(tf.multiply(self,self))
# #    autoCorrGT = tf.reduce_sum(tf.multiply(ground_truth,ground_truth))
#     scaling_factor = tf.divide(corr,autoCorrIn)
#     self = tf.multiply(self,scaling_factor)
#     squared_deltas = tf.square(self - ground_truth)
#     loss = tf.reduce_sum(squared_deltas, reduction_indices=[1,2])
#     loss_ = tf.reduce_mean(loss, name='mse')

    #UNITARY MSE LOSS
#    energyratio = tf.divide(tf.reduce_sum(tf.square(tf.abs(ground_truth))),tf.reduce_sum(self))
#    ground_truth = tf.multiply(self,energyratio)
#    squared_deltas = tf.square(self - ground_truth)
#    loss = tf.reduce_sum(squared_deltas, reduction_indices=[1])
#    loss_ = tf.reduce_mean(loss, name='mse')
    
    #LEAKAGE LOSS
#    squared_delta_leak = tf.reduce_sum(tf.square(leakage),reduction_indices=[2])
#    leak_loss = tf.reduce_mean(squared_delta_leak,reduction_indices=[1])    
    
    #loss_ = tf.cond((step*BATCH_SIZE)<(NUMBER_TRAINING_ELEMENTS*0), lambda: tf.add(loss_, tf.norm(leak_loss,name='light_leakage')), lambda: loss_)
#    tf.summary.scalar('loss', loss_)
# Cross entropy
    loss_ = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=self)

    return loss_

def tv_loss_function(measurement):

    pixel_dif1 = tf.subtract(measurement[:, 1:, :], measurement[:, :-1, :])
    pixel_dif2 = tf.subtract(measurement[:, :, 1:], measurement[:, :, :-1])
    sum_axis = [1, 2]
    tot_var = (tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) +
               tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis))
    tot_var_mean = tf.reduce_mean(tot_var, name='tv')

    return tot_var_mean


def training(loss):

    ONN_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mask')
    DNN_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense')
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if tc.OPTIMIZER == 'gradient':
        optimizer = tf.train.GradientDescentOptimizer(tc.LEARNING_RATE)
    elif tc.OPTIMIZER == 'adam':
        optimizer_ONN = tf.train.AdamOptimizer(tc.LEARNING_RATE * 10)
        optimizer_DNN = tf.train.AdamOptimizer(tc.LEARNING_RATE * 0.1)
    else:
        pass
    train_op = tf.group([optimizer_ONN.minimize(loss, var_list=ONN_variable), optimizer_DNN.minimize(loss, var_list=DNN_variable), extra_update_ops])

    return train_op


def reset_ONN():

    target_op = []

    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mask')

    if tc.MASK_AMPLITUDE_MODULATION is True:
    
        for num in range(tc.MASK_NUMBER):

            if tc.MASK_PHASE_MODULATION is True:

                if tc.MASK_INIT_TYPE == 'const':
                    op1 = train_vars[2 * num].assign(tf.constant(1.0, shape=[tc.MASK_ROW, tc.MASK_COL]))
                elif tc.MASK_INIT_TYPE == 'random':
                    op1 = train_vars[2 * num].assign(tf.random_uniform(shape=[tc.MASK_ROW, tc.MASK_COL],minval=0.5,maxval=1.5))

                target_op = tf.group(target_op, op1)
                # elif tc.MASK_INIT_TYPE == 'trained':
                #     mask_phase = read_mask(tc.MASK_PATH, masknum, 'phase')
            # else:
            #     mask_phase = tf.zeros([tc.MASK_ROW, tc.MASK_COL])
                
            if tc.MASK_AMPLITUDE_MODULATION is True:
                # if tc.MASK_INIT_TYPE == 'trained':
                #     mask_amp = read_mask(tc.MASK_PATH, masknum, 'amplitude')
                # else:
                op2 = train_vars[2 * num + 1].assign(tf.constant(1.0, shape=[tc.MASK_ROW, tc.MASK_COL]))

                target_op = tf.group(target_op, op2)
            # else:
            #     mask_amp = tf.ones([tc.MASK_ROW, tc.MASK_COL])

    else:

        for num in range(tc.MASK_NUMBER):

            if tc.MASK_INIT_TYPE == 'const':
                train_vars[num].assign(tf.constant(1.0, shape=[tc.MASK_ROW, tc.MASK_COL]))
            elif tc.MASK_INIT_TYPE == 'random':
                train_vars[num].assign(tf.random_uniform(shape=[tc.MASK_ROW, tc.MASK_COL],minval=0.5,maxval=1.5))

    return target_op