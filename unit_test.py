#!/usr/bin/env ~/anaconda3/envs/tf-gpu/bin/python
from argparse import ArgumentParser
import numpy as np
import cv2
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

parser = ArgumentParser()
parser.add_argument('--image_dir', default="./test", type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', default="./test", type=str, help='Directory where to output high res images.')
parser.add_argument('--model', default="./models/autoencoder.h5", type=str, help='Path to model to use for inference.')
parser.add_argument('--debug', default=False, type=bool, help='Show debug printing.')
parser.add_argument('--logdir', default="./test/logs", type=str, help='Tensorboard logdir.')

def denoise(img, dst=None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
  # Default kwargs: dst=None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
  return cv2.fastNlMeansDenoisingColored(img, dst, h, hColor, templateWindowSize, searchWindowSize)

def main():
    args = parser.parse_args()
    debug = args.debug
    logdir = args.logdir

    # Get all image paths
    image_dir = os.path.expanduser(os.path.expandvars(args.image_dir))
    output_dir = os.path.expanduser(os.path.expandvars(args.output_dir))
    image_paths = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir))]

    # Change model input shape to accept all size inputs
    model_path = os.path.expanduser(os.path.expandvars(args.model))
    model = tf.keras.models.load_model(model_path)
    inputs = tf.keras.Input((None, None, 3))
    output = model(inputs)
    model = tf.keras.models.Model(inputs, output)

    # Loop over all images
    for image_path in image_paths:

        # Read image
        low_res = cv2.imread(image_path, 1)

        # Convert to RGB (opencv uses BGR as default)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

        # Rescale to 0-1.
        low_res = low_res.astype(np.float32) / 255.0

        # Get super resolution image
        if debug:
          print("  Performing Inference")
          print(f"  frame type: {type(low_res)}, dtype: {low_res.dtype}, shape: {low_res.shape}\n")
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]
        if debug:
          print("  Inference Complete")
          print(f"  frame type: {type(sr)}, dtype: {sr.dtype}, shape: {sr.shape}\n")

        # Rescale values in range 0-255
        low_res = np.uint8(low_res * 255)
        sr = np.uint8(((sr + 1) / 2.) * 255)

        # Convert back to BGR for opencv
        # Default denoise kwargs: (dst=None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_RGB2BGR)
        # lr_denoise = denoise(low_res, dst=None, h=10, hColor=10, templateWindowSize=3, searchWindowSize=7)
        lr_denoise = cv2.medianBlur(low_res, 3)
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        # sr_denoise = denoise(sr, dst=None, h=10, hColor=10, templateWindowSize=3, searchWindowSize=7)
        sr_denoise = cv2.medianBlur(sr, 3)

        if debug:
          print("  Converting back to uint8")
          print(f"  sr frame type: {type(sr)}, dtype: {sr.dtype}, shape: {sr.shape}")
          print(f"  low_res frame type: {type(low_res)}, dtype: {low_res.dtype}, shape: {low_res.shape}")

        # Display input frame
        cv2.namedWindow("Input frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Input frame", low_res)

        # Display output frame
        cv2.namedWindow("LR Denoise", cv2.WINDOW_NORMAL)
        cv2.imshow("LR Denoise", lr_denoise)

        # Display output frame
        cv2.namedWindow("Output frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Output frame", sr)

        # Display output frame opencv denoise
        cv2.namedWindow("SR Denoise", cv2.WINDOW_NORMAL)
        cv2.imshow("SR Denoise", sr_denoise)

        cv2.waitKey(0) #& 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
