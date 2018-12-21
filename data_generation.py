'''
Data Generation
'''
import numpy as np
import data_ops as dto
import tensorflow as tf
import initialization as init

tc = init.init_params()
    
if tc.TRAINING_DATA_TYPE == 'mnist' or tc.TESTING_DATA_TYPE == 'mnist':
    data = tf.keras.datasets.mnist.load_data()

if tc.TRAINING_DATA_TYPE == 'cifar10' or tc.TESTING_DATA_TYPE == 'cifar10':
    data = tf.keras.datasets.cifar10.load_data()

if tc.TRAINING_DATA_TYPE == 'fashion-mnist' or tc.TESTING_DATA_TYPE == 'fashion-mnist':
    data = tf.keras.datasets.fashion_mnist.load_data()
    
# separating dataset components
train, test = data
data_train, label_train = train
data_test, label_test = test
global_norm = np.amax([np.amax(np.abs(data_train)), np.amax(np.abs(data_test))])
# print(global_norm)

# Create Validation Set
indices_valid = dto.create_validation(label_train, label_test, tc.validation_ratio)
data_valid = data_train[indices_valid]
label_valid = label_train[indices_valid]
N_VALID = np.amax(label_valid.shape)
data_train = np.delete(data_train, indices_valid, axis=0)
label_train = np.delete(label_train, indices_valid)

gts = tf.zeros((tc.NUM_CLASS, tc.M, tc.N),dtype=tf.float32)
padxgt = int((tc.M - tc.SENSOR_ROW) / 2)
padygt = int((tc.N - tc.SENSOR_COL) / 2)
for i in range(tc.NUM_CLASS):
    gt_i  = dto.gt_generator_classification(i)
    gt_p = tf.pad(gt_i, ((padxgt, padxgt), (padygt, padygt)), 'constant')
    gt_p = tf.expand_dims(gt_p, 0)
    gts = tf.add(gts,tf.scatter_nd([[i]], gt_p, shape=[tc.NUM_CLASS, tc.M, tc.N]))

gts_tensor = gts #tf.convert_to_tensor(gts,dtype=tf.float32)    

def _preprocess(img, label):
    
    img_r = tf.reshape(img, [tc.DATA_ROW, tc.DATA_COL, 1])
    img_r = tf.image.resize_images(img_r, [tc.OBJECT_ROW, tc.OBJECT_COL],align_corners=True)
    img_r = tf.reshape(img_r, [tc.OBJECT_ROW, tc.OBJECT_COL])
    img_r = tf.divide(img_r,global_norm)
    
    # print(img_r)
    
    padx = int((tc.M - tc.OBJECT_ROW) / 2)
    pady = int((tc.N - tc.OBJECT_COL) / 2)
    img_pad = tf.pad(img_r, [(padx, padx), (pady, pady)], 'constant')

    label = tf.cast(label,dtype=tf.int64)
    gt = tf.squeeze(tf.slice(gts_tensor, [label, 0, 0,], [1, tc.M, tc.N]))

    if tc.OBJECT_AMPLITUDE_INPUT is True:
        img_amp = img_pad*100
        img_phase = tf.zeros((tc.M,tc.N),dtype=tf.float32)
    elif tc.OBJECT_PHASE_INPUT is True:
        img_amp = 100*tf.ones((tc.OBJECT_ROW,tc.OBJECT_COL), dtype=tf.float32)
        img_amp = tf.pad(img_amp, [(padx, padx), (pady, pady)], 'constant')
        img_phase = 1.999 * np.pi * img_pad/(tf.reduce_max(img_pad))

    return img_amp, img_phase, gt, label

# The following functions return TF operation for getting the next
# training, validation, or testing data batch
def get_data_batch(request_type):
    
    if request_type == 'training':
        get_batch = tf.data.Dataset.from_tensor_slices((data_train, label_train))
        get_batch = get_batch.shuffle(tc.NUMBER_TRAINING_ELEMENTS-N_VALID)    
        
    elif request_type == 'testing':
        get_batch = tf.data.Dataset.from_tensor_slices((data_test, label_test))
        get_batch = get_batch.shuffle(tc.NUMBER_TEST_ELEMENTS)
        
    elif request_type == 'validation':
        get_batch = tf.data.Dataset.from_tensor_slices((data_valid, label_valid))
        get_batch = get_batch.shuffle(N_VALID)
   
    get_batch = get_batch.map(_preprocess, 4)
    get_batch = get_batch.prefetch(buffer_size=tc.BATCH_SIZE*100)
    get_batch = get_batch.batch(tc.BATCH_SIZE, drop_remainder = True)
    
    return get_batch#, init_val