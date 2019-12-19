from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= "3"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

tf.config.set_soft_device_placement(True)
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
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')
parser.add_argument('--model', default="./models/autoencoder.h5", type=str, help='Path to model to use for inference.')


def main():
    args = parser.parse_args()

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
        low_res = low_res / 255.0

        # Get super resolution image
        print("  Performing Inference")
        print(f"  frame type: {type(low_res)}, dtype: {low_res.dtype}, shape: {low_res.shape}")
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]
        print("  Inference Complete")
        print(f"  frame type: {type(sr)}, dtype: {sr.dtype}, shape: {sr.shape}")

        # Rescale values in range 0-255
        sr = ((sr + 1) / 2.) * 255

        # Convert back to BGR for opencv
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR).astype(np.uint8)
        print("  Converting back to uint8")
        print(f"  frame type: {type(sr)}, dtype: {sr.dtype}, shape: {sr.shape}")

        # Display outgoing frame
        cv2.imshow("Incoming frame", low_res)

        # Display outgoing frame
        cv2.imshow("Outgoing frame", sr)

        if cv2.waitKey(100) & 0xFF == ord('q'):
          cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), sr)
          break

        # Save the results:
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), sr)


if __name__ == '__main__':
    main()
