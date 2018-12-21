'''
MAIN TRAINING CODE
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mask_modulation_model as mmm
import data_generation as dtg
import data_ops as dto
import initialization as init
from datetime import datetime
import os
import scipy.io as sio
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# IMPORT and GLOBALIZE PARAMETERS
tc = init.init_params()

#GPU-COMPUTING CONFIGURATION
def get_default_config(fraction=0.9):

    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = fraction
    conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True

    return conf
tf.gfile.MakeDirs(tc.OUTPUT_PATH)
tf.gfile.MakeDirs(tc.MASK_SAVING_PATH)
tf.gfile.MakeDirs(tc.MASK_TESTING_PATH)

# cell_locs, thecell, label_fields, det_distribution = dto.sensorplane_geometry()  

# List of monitoring variables
global_step = 0
#accuracy_train
#accuracy_test
#total_loss_train
#mean_loss_train
#total_loss_test
#mean_loss_test
#hit_count
best_loss = 1e12
best_accuracy = 0

# LOG FILE
text_file = open("LOG.txt", 'w')
msg = "epoch " + "mean_train_loss " + "mean_test_loss " + "training_accuracy " + "testing_accuracy "
text_file.write(msg + '\n')

if __name__ == '__main__':
    
    conf = get_default_config()
    sess = tf.InteractiveSession(config=conf)
    
    with tf.name_scope("datasets"):
        # DEFINE DATA PIPELINE, INITIALIZE ITERATOR
        batch_train = dtg.get_data_batch('training')
        batch_test = dtg.get_data_batch('validation')
        batch_final = dtg.get_data_batch('testing')
        # DEFINE THE ITERATOR
        iterator = tf.data.Iterator.from_structure(batch_train.output_types, batch_train.output_shapes)
        # iterator: tf.data.Iterator = tf.data.Iterator.from_structure(batch_train.output_types, batch_train.output_shapes)
        batch = iterator.get_next()
        data_amp, data_phase, sensor_gt, data_label = batch
        onn_field = tf.complex(data_amp * tf.cos(data_phase), data_amp * tf.sin(data_phase))
    
    # DEFINE THE MODEL
    onn_measurement, onn_mask_phase, onn_mask_amp, onn_logits = mmm.inference(onn_field)
    onn_loss = mmm.loss_function(onn_logits, tf.cast(data_label,dtype=tf.int64), onn_field)
    onn_predictions = tf.nn.softmax(onn_logits, name = 'predictions')
    onn_hit = tf.reduce_sum(tf.cast(tf.equal(tf.cast(data_label,dtype=tf.int64), tf.argmax(onn_predictions, axis=1)),dtype=tf.int64))
    onn_tv_loss = mmm.tv_loss_function(onn_measurement)
    onn_combine_loss = onn_loss + tc.TV_LOSS_PARAM * onn_tv_loss
    onn_train = mmm.training(onn_combine_loss)

    reset_op = mmm.reset_ONN()
    
#    accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(data_label, axis=1), tf.argmax(onn_predictions, axis=1))
#    mean_loss, mean_loss_op = tf.metrics.mean(onn_loss)
        
    init_all = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver()
    sess.run(init_all)
#    sess.graph.finalize()    
    print("Entering training loop")
    
    for epoch in range(1,tc.MAX_EPOCH+1):
        
        # Initialize iterator and shuffle data
#        train_data = batch_train.shuffle(tc.NUMBER_TRAINING_ELEMENTS)
        sess.run(iterator.make_initializer(batch_train))
        train_step = input_count = hit_count_train = total_loss_train = 0
        while True:
            try:
                # run train iteration        
                _,  onn_loss_value, onn_mask_phase_value, onn_mask_amp_value, onn_predictions_value, hit_count, input_field = sess.run([onn_train, 
                                                                                                                           onn_combine_loss, 
                                                                                                                           onn_mask_phase, onn_mask_amp, 
                                                                                                                           onn_predictions, onn_hit, 
                                                                                                                           onn_field])
                # print(input_field.shape)
                # exit()

                train_step += 1
                global_step += 1
                input_count += tc.BATCH_SIZE
                hit_count_train += hit_count
                total_loss_train += onn_loss_value
                if(train_step==1):
                    first_input = input_field[0,:,:]
                    first_input_phase = np.angle(first_input)
                    first_input_amp = np.abs(first_input)
                    plt.imsave("./first_input_amp" + ".png", first_input_amp, cmap='gray')
                    plt.imsave("./first_input_phase" + ".png", first_input_phase, cmap='gray')
                    
            except (tf.errors.OutOfRangeError, StopIteration):
                break
            
        accuracy_train = hit_count_train/input_count*100
        mean_loss_train = total_loss_train/train_step
        # initialize iterator for validation dataset. No need to shuffle    
        sess.run(iterator.make_initializer(batch_test))
        test_step = test_count = hit_count_test = total_loss_test = 0
        while True:
            try:
                # run test iteration        
                onn_loss_value_test, onn_measurement_value_test, test_field, hit_count = sess.run([onn_combine_loss, onn_measurement, onn_field, onn_hit])                            
                
                test_step += 1
                test_count += tc.BATCH_SIZE
                hit_count_test += hit_count
                total_loss_test += onn_loss_value_test
            except (tf.errors.OutOfRangeError, StopIteration):
                break
        
        accuracy_test = hit_count_test/test_count*100
        mean_loss_test = total_loss_test/test_step
        #Log/Record
        msg = "epoch " + "mean_train_loss " + "mean_test_loss " + "training_accuracy " + "testing_accuracy " + str(datetime.now())
        print(msg)
        msg = str(epoch) + " " + str(mean_loss_train) + " " + str(mean_loss_test) + " " + str(accuracy_train) + " " + str(accuracy_test)
        print(msg)
        text_file.write(msg + '\n')
        
        plt.imsave(tc.OUTPUT_PATH + "/SensPlInt_" + str(epoch) + ".png", onn_measurement_value_test[0,:,:], cmap='gray')
        plt.imsave(tc.OUTPUT_PATH + "/TestInput_" + str(epoch) + ".png", np.angle(test_field[0,:,:]), cmap='gray')
        
        if tc.APPLICATION == 'classification':
            save_model = bool(accuracy_test>best_accuracy)
        elif tc.APPLICATION == 'amplitude_imaging':
            save_model = bool(mean_loss_test<best_loss)
        elif tc.APPLICATION == 'phase_imaging':
            save_model = bool(mean_loss_test<best_loss)
            
        if save_model is True:
            save_path = saver.save(sess, "./MODEL/model{}".format(epoch))
            print("Model saved in path: %s" % save_path)
            best_loss = mean_loss_test 
            best_accuracy = accuracy_test
            save_epoch = epoch
            
            for i in range(tc.MASK_NUMBER):
                np.savetxt(tc.MASK_SAVING_PATH + "/mask_phase_" + str(epoch) + "_" + str(i) + ".txt", onn_mask_phase_value[i,:,:])
                np.savetxt(tc.MASK_SAVING_PATH + "/mask_amp_" + str(epoch) + "_" + str(i) + ".txt", onn_mask_phase_value[i,:,:])

        if epoch % 15 == 0 :
            sess.run(reset_op)

    print("Training Finished! Generating Confusion Matrix Data.")

    saver.restore(sess,"./MODEL/model{}".format(save_epoch))
    sess.run(iterator.make_initializer(batch_final))
    test_step = test_count = hit_count_test = total_loss_test = 0
    prediction = np.zeros((1,10))
    label = 0
    while True:
        try:
            # run train iteration        
            onn_loss_value_test, onn_measurement_value_test, test_field, hit_count = sess.run([onn_combine_loss, onn_measurement, onn_field, onn_hit])                            
                
            test_step += 1
            test_count += tc.BATCH_SIZE
            hit_count_test += hit_count
            total_loss_test += onn_loss_value_test
        except (tf.errors.OutOfRangeError, StopIteration):
            break
    
    accuracy_test = hit_count_test/test_count*100
    mean_loss_test = total_loss_test/test_step

    msg = "epoch " + "mean_train_loss " + "mean_test_loss " + "training_accuracy " + "testing_accuracy " + str(datetime.now())
    print(msg)
    msg = str(epoch) + " " + str(mean_loss_train) + " " + str(mean_loss_test) + " " + str(accuracy_train) + " " + str(accuracy_test)
    print(msg)
    text_file.write(msg + '\n')
    print("Finished!")
