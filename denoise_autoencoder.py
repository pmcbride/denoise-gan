from __future__ import absolute_import, division, print_function, unicode_literals, with_statement
import os
import sys
import glob
import thread_utils
import math
import multiprocessing
import getopt
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# np.random.seed(1234)
# tf.set_random_seed(1234)

strategy = tf.distribute.get_strategy()

# Set these variables if TensorFlow should use a certain GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # "0, 1" for multiple
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'#avoid SSE warnings

# PLOTTING FUNCTIONS

def plot_loss_acc(model, title=None):
  """
  Takes a deep learning model and plots the loss ans accuracy over epochs
  Users can supply a title if needed
  target_acc: The desired/ target acc. This parameter is needed for this function to show a horizontal bar.
  """
   
  val = True
  epochs = np.array(model.history.epoch)+1 # Add one to the list of epochs which is zero-indexed
  keys = model.history.history.keys()
  n_keys = len(model.history.history.keys())
  
  # Create Figure
  nrows = 1
  ncols = n_keys // 2 if val else n_keys
  fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 4))
  fig.subplots_adjust(wspace=.75)
  if title:
    fig.suptitle(title)
      
  for i, key in enumerate(keys):
    metric = np.array(model.history.history.get(key))
    if val:
      val_i = i + n_keys // 2
      val_key = "val_" + key
      val_metric = np.array(model.history.history.get(val_key))
      if i == ncols:
        break

    ax[i] = plt.subplot(nrows, ncols, i+1)
    color = 'tab:red'
    ax[i].set_xlabel('Epochs', fontsize=15)
    ax[i].set_ylabel(key, color=color, fontsize=15)
    ax[i].plot(epochs, metric, color=color, lw=2)
    if val:
      ax[i].plot(epochs, val_metric, color=color, lw=2, linestyle='dashed')
      plt.legend(['train', 'validate'], loc='lower left')
    ax[i].tick_params(axis='y', labelcolor=color)
    ax[i].grid(True)
    #ax[i].title.set_text(key)
    ax[i].set_title(key, fontsize=15)
  plt.show()
  
#--------------------------------------------

def myplot(img_batch, autoscale=False, title=None, ncols=0):
  batch_shape = img_batch.shape
  batch_dim = img_batch.ndim
  images = img_batch.copy() #if batch_dim == 4 else tf.expand_dims(img_batch, 0)
  if images.max() <= 1:
    clip_max = 1
  else:
    clip_max = 255
    
  print(f"batch_shape: {batch_shape}, batch_dim: {batch_dim}, clip_max: {clip_max}")
  
  if batch_dim == 2:
    images = np.expand_dims(images, -1)
    images = np.expand_dims(images, 0)
  elif batch_dim == 3:
    images = np.expand_dims(images, 0)
  elif batch_dim == 4:
    images = images
  else:
    print(f"Dimension of input batch ({batch_dim}) incompatible")
    return
  
  nrows = 1
  if not ncols:
    ncols = len(images)
        
  fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(2*ncols, 2))
  if title:
    fig.suptitle(title, ha='left')
    
  if ncols == 1:
    ax = plt.subplot(nrows, ncols, 1)
    img = np.array(images[0])
    if autoscale:
      img_gray = np.array(img).mean(axis=-1)
      img_mean = img_gray.mean()
      img = (img-img_gray.min())/img_gray.ptp()
      img = np.clip(img, 0, clip_max)
    elif not autoscale:
      img = np.clip(img, 0, clip_max)
    plt.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  elif ncols > 1:
    for i in range(ncols):
      ax[i] = plt.subplot(nrows, ncols, i+1)
      img = np.array(images[i])
      if autoscale:
        img_gray = np.array(img).mean(axis=-1)
        img_mean = img_gray.mean()
        img = (img-img_gray.min())/img_gray.ptp()
        img = np.clip(img, 0, clip_max)
        #img = tf.clip_by_value(img, 0, clip_max)
      elif not autoscale:
        img = np.clip(img, 0, clip_max)
        #img = tf.clip_by_value(img, 0, clip_max)
      plt.imshow(img)

      ax[i].get_xaxis().set_visible(False)
      ax[i].get_yaxis().set_visible(False)
  plt.show()
  
# LOAD CIFAR DATA

import cv2

# Setup jpeg noise function
jpeg_params = dict(min_jpeg_quality=25, 
                   max_jpeg_quality=50)

def add_jpeg_noise(img, jpeg_quality=25):
  jpeg_quality = jpeg_quality
  img_corrupt = tf.image.adjust_jpeg_quality(img, jpeg_quality)
  img_corrupt = tf.clip_by_value(img_corrupt, 0, 1)
  return img_corrupt

def random_jpeg_noise(img, min_jpeg_quality=25, max_jpeg_quality=50):
  img_corrupt = tf.image.random_jpeg_quality(img, **jpeg_params)
  img_corrupt = tf.clip_by_value(img_corrupt, 0, 1)
  return img_corrupt

def add_random_noise(img):
  noise_factor = 0.1
  img_corrupt = img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape[:-1])
  img_corrupt = np.clip(img_corrupt, 0, 1)
  return img_corrupt

def cv2_jpeg(img, quality=25):
  jpeg_quality = quality
  img = np.array(img)
  if img.max() <= 1:
    img = np.multiply(img, 255)
  img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  return img

def corrupt_batch(batch, quality=25):
  batch = batch.copy()
  for i, img in enumerate(batch):
    img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    batch[i] = cv2.imdecode(img, cv2.IMREAD_COLOR)
  return batch

# Load CIFAR10 Images
def cifar10_dataset(debug=False):
  (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
  print("Import: Dtype: ", x_train.dtype, " Training Dataset Dimensions: ", x_train.shape, "Training Dataset Max: ", x_train.max())
  print("Import: Dtype: ", x_train.dtype, "Testing Dataset Dimensions: ", x_test.shape, "Testing Dataset Max: ", x_test.max())
  x_train_corrupt = corrupt_batch(x_train)
  x_test_corrupt = corrupt_batch(x_test)
  
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  x_train_corrupt = x_train_corrupt.astype('float32') / 255.
  x_test_corrupt = x_test_corrupt.astype('float32') / 255.
  
  return (x_train, x_train_corrupt), (x_test, x_test_corrupt)

(train_dataset, train_dataset_corrupt), (test_dataset, test_dataset_corrupt) = cifar10_dataset()

# Create Dataset from numpy arrays
train = tf.data.Dataset.from_tensor_slices((train_dataset_corrupt, train_dataset))
test = tf.data.Dataset.from_tensor_slices((test_dataset_corrupt, test_dataset))

TRAIN_BUF = 50000
TEST_BUF = 10000
BATCH_SIZE = 1000

train = train.shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test = train.shuffle(TEST_BUF).batch(BATCH_SIZE)

# CREATE AUTOENCODER

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, SpatialDropout2D, ReLU
from tensorflow.keras.models import Model

def Autoencoder(input_shape=(32, 32, 3), input_tensor=None):

  def _conv2d(x, filters, kernel_size=(3, 3), strides=(1, 1), name='conv', relu=True):
    with tf.name_scope(name) as scope:
      in_shape = x.get_shape().as_list()
      in_channel = in_shape[-1]

      # Weights according to He initializer
      filter_shape = [3, 3, in_channel, filters]

      fan_in = 3.0*3.0*in_channel
      if relu:
        kernel_initializer = tf.keras.initializers.he_normal()
        res = Conv2D(filters, kernel_size, strides,
                     padding='same',
                     activation='relu',
                     kernel_initializer=kernel_initializer)(x)
      else:
        kernel_initializer = tf.keras.initializers.lecun_normal()
        res = Conv2D(filters, kernel_size, strides,
                     padding='same',
                     activation=None,
                     kernel_initializer=kernel_initializer)(x)
    return res

  def _maxpool2d(x, k=2, name='pool'):
    # MaxPool2D wrapper
    with tf.name_scope(name) as scope:
      res = tf.keras.layers.MaxPool2D(pool_size=(k, k), strides=(k, k),
                                      padding='same')(x)
    return res

  def _unpool(value, name='unpool'):
    with tf.name_scope(name) as scope:
      res = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(value)
      #res = tf.nn.conv2d_transpose(value, W, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME')
      return tf.keras.activations.relu(res)

  def _unpool_concat(a, b, name='upconcat'):
    with tf.name_scope(name) as scope:
      up = _unpool(a)
      res = tf.keras.layers.concatenate([up, b], axis=-1)
      #res = tf.concat([up, b], 3)
    return res

  def _concat(a, b, name='concat'):
    with tf.name_scope(name) as scope:
      return tf.keras.layers.concatenate([a, b], axis=-1)
    
  # Build Autoencoder
  def build_autoencoder(x):
    prevLayer = conv1  = _conv2d(x, 32, name='conv1')
    prevLayer = conv1b = _conv2d(prevLayer, 32, name='conv1b')
    prevLayer = pool1  = _maxpool2d(prevLayer, 2, name='pool1') # 256 -> 128

    prevLayer = conv2 = _conv2d(prevLayer, 44, name='conv2')
    prevLayer = pool2 = _maxpool2d(prevLayer, 2, name='pool2') # 128 -> 64

    prevLayer = conv3 = _conv2d(prevLayer, 56, name='conv3')
    prevLayer = pool3 = _maxpool2d(prevLayer, 2, name='pool3') # 64 -> 32

    prevLayer = conv4 = _conv2d(prevLayer, 76, name='conv4')
    prevLayer = pool4 = _maxpool2d(prevLayer, 2, name='pool4') # 32 -> 16

    prevLayer = conv5 = _conv2d(prevLayer,  100, name='conv5')
    prevLayer = pool5 = _maxpool2d(prevLayer, 2, name='pool5') # 16 -> 8

    prevLayer = us6 = _unpool_concat(prevLayer, pool4, name='unpool4')
    prevLayer = conv6 = _conv2d(prevLayer,  152, name='conv6')
    prevLayer = conv6b = _conv2d(prevLayer, 152, name='conv6b')

    prevLayer = us7 = _unpool_concat(prevLayer, pool3, name='unpool3')
    prevLayer = conv7 = _conv2d(prevLayer, 112, name='conv7')
    prevLayer = conv7b = _conv2d(prevLayer, 112, name='conv7b')

    prevLayer = us8 = _unpool_concat(prevLayer, pool2, name='unpool2')
    prevLayer = conv8 = _conv2d(prevLayer, 84, name='conv8')
    prevLayer = conv8b = _conv2d(prevLayer, 84, name='conv8b')

    prevLayer = us9 = _unpool_concat(prevLayer, pool1, name='unpool1')
    prevLayer = conv9 = _conv2d(prevLayer,  64, name='conv9')
    prevLayer = conv9b = _conv2d(prevLayer, 64, name='conv9b')

    prevLayer = us10 = _unpool_concat(prevLayer, x, name='unpool0')
    prevLayer = conv10 = _conv2d(prevLayer, 64, name='conv10')
    prevLayer = conv10b = _conv2d(prevLayer, 32, name='conv10b')

    out = _conv2d(prevLayer, 3, name='conv11', relu=False)

    #print("Output var: %s" % out.name)

    return out
  
  if input_shape:
    img_shape = input_shape
  elif input_tensor:
    tensor_shape = input_tensor.get_shape().as_list()
    img_shape = tensor_shape[1:]
    
  img_noise = Input(shape=img_shape, dtype='float32', name='input_img')
  img_denoise = build_autoencoder(img_noise)
  autoencoder = Model(inputs=img_noise, outputs=img_denoise, name='autoencoder')
  
  return autoencoder

# CREATE AND COMPILE MODEL

with strategy.scope():
  """
  This essentailly takes our model and makes it 
  compatible to train on a TPU.
  """
  model = Autoencoder()
  
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer, 
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

# FIT MODEL

TRAIN_BUF = 50000
TEST_BUF = 10000
BATCH_SIZE = 2000

STEPS_PER_EPOCH=int(np.ceil(TRAIN_BUF / float(BATCH_SIZE)))
VALIDATION_STEPS=int(np.ceil(TEST_BUF / float(BATCH_SIZE)))

tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir="./logs",
  histogram_freq=1,
  write_graph=True,
  write_images=True,
  update_freq='epoch'
)

model.fit(train_dataset_corrupt, train_dataset,
          epochs=5,
          #steps_per_epoch=STEPS_PER_EPOCH,
          batch_size=BATCH_SIZE,
          shuffle=True,
          validation_data=(test_dataset_corrupt, test_dataset),
          callbacks=[tensorboard_callback])

model.save("./model.h5")