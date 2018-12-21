# -*- coding:UTF-8 -*-
 
import collections
import tensorflow as tf
from tensorflow.contrib.layers import flatten
slim = tf.contrib.slim
 
 
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  '''
    collections.namedtuple ResNet block named tuple,
    Block：
    scope： Block
    unit_fn：ResNet V2 
    args： block[(depth, depth_bottleneck, stride)]
        ：Block('block1', bottleneck, [(256,64,1),(256,64,1),(256,64,2)])
  '''
 
def subsample(inputs, factor, scope=None): 
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
 
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None): 
  """
  if stride>1, then we do explicit zero-padding, followed by conv2d with 'VALID' padding
  """
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
  else:
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)
 
 
#---------------------Blocks-------------------
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
  """
  Args:
    net: A Tensor of size [batch, height, width, channels].inputs
    blocks: 是之前定义的Block的class的列表。
    outputs_collections: 收集各个end_points的collections
  Returns:
    net: Output tensor
  """
  # 循环Block类对象的列表blocks,即逐个Residual Unit地堆叠
  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride = unit
          net = block.unit_fn(net, depth=unit_depth, depth_bottleneck=unit_depth_bottleneck,
                              stride=unit_stride)
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
      
  return net
 
 
# ResNet arg_scope,arg_scope
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
 
  batch_norm_params = {# batch normalization
      'is_training': is_training,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
 
  with slim.arg_scope( # slim.arg_scope [slim.conv2d]
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay), # L2
      weights_initializer=slim.variance_scaling_initializer(), # 
      activation_fn=tf.nn.relu, # 
      normalizer_fn=slim.batch_norm, # BN
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc # arg_scope
 
 
 
#------------------bottleneck--------------------
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
  """
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth、depth_bottleneck:、stride三个blocks args
    rate: An integer, rate for atrous convolution.
    outputs_collections: end_points collection
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc: 
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4) #最后一个维度,即输出通道数
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact') 
 
    if depth == depth_in:
      # inputs
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
      # 
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')
      
    # 1x1,depth_bottleneck
    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
    # 3x3，depth_bottleneck
    residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
    # 1x1，1，depth, residual
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')
 
    # residual
    output = shortcut + residual
 
    return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)
 
 
#-------------------resnet_v2------------------
 
def resnet_v2(inputs, blocks, num_classes=None, global_pool=True, 
              include_root_block=True, reuse=None, scope=None):
 
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                        outputs_collections=end_points_collection):
 
      net = inputs
      if include_root_block: # resnet 64 2 7x7
        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
          net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        # 1/4
 
      net = stack_blocks_dense(net, blocks)
      net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
 
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True) # tf.reduce_mean, avg_pool
 
      if num_classes is not None:  #
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, # 
                          normalizer_fn=None, scope='logits') # num_classes, 1x1
      end_points = slim.utils.convert_collection_to_dict(end_points_collection) # collection, python, dict
 
      if num_classes is not None:
        end_points['predictions'] = slim.softmax(net, scope='predictions') #
      return net, end_points
 
 
#------------------- ResNet-50/101/152/200 model--------------------
 
def resnet_v2_50(inputs, num_classes=None, global_pool=True, reuse=None,
                 scope='resnet_v2_50'):
  blocks = [
      Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
 
      # Args:：
      # 'block1'：Block（scope）
      # bottleneck：ResNet V2
      # [(256, 64, 1)] * 2 + [(256, 64, 2)]：Block Args，Args, bottleneck
      #                                     (256, 64, 1)，(256, 64, 2)
      #                                     tuple，（depth，depth_bottleneck，stride）。
      # (256, 64, 3) bottleneck
      # depth 256，depth_bottleneck 64，3：
      # [(1*1/s1,64),(3*3/s2,64),(1*1/s1,256)]
 
      Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)
 
 
def resnet_v2_101(inputs, num_classes=None, global_pool=True, reuse=None,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
 
  blocks = [
      Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
      Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)
 
 
# unit block3
def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None,
                  scope='resnet_v2_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
 
  blocks = [
      Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
      Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)
 
 
# unit block2
def resnet_v2_200(inputs, num_classes=None, global_pool=True, reuse=None,
                  scope='resnet_v2_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
 
  blocks = [
      Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
      Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
      Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope)


def AlexNet(inputs, num_classes=10):

  network = tf.layers.conv2d(inputs, 96, 11, strides=4, activation='relu')
  network = tf.layers.max_pooling2d(network, 3, strides=2)
  network = tf.nn.lrn(network)
  network = tf.layers.conv2d(network, 256, 5, activation='relu')
  network = tf.layers.max_pooling2d(network, 3, strides=2)
  network = tf.nn.lrn(network)
  network = tf.layers.conv2d(network, 384, 3, activation='relu')
  network = tf.layers.conv2d(network, 384, 3, activation='relu')
  network = tf.layers.conv2d(network, 256, 3, activation='relu')
  network = tf.layers.max_pooling2d(network, 3, strides=2)
  network = tf.nn.lrn(network)
  network = tf.layers.flatten(network)
  network = tf.layers.dense(network, 4096, activation='tanh')
  network = tf.layers.dropout(network, 0.5)
  network = tf.layers.dense(network, 4096, activation='tanh')
  network = tf.layers.dropout(network, 0.5)
  logits = tf.layers.dense(network, num_classes)

  return logits

def LeNet_5(inputs, num_classes):
  
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(inputs,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,num_classes), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(num_classes))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits

def LeNet_5_33(inputs, num_classes):
  
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    
    # TODO: Layer 1: Convolutional. Input = 33x33x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [6,6,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(inputs,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,num_classes), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(num_classes))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits    