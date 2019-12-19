from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

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
parser.add_argument('--input_video', type=str, help='Path to input video')
parser.add_argument('--output_video', type=str, help='Path to output high res video.')
parser.add_argument('--model', default="./models/autoencoder.h5", type=str, help='Path to model.')


def make_video(outvid, images=None, fps=30, size=None,
         is_color=True, format="FMP4"):
  """
  Create a video from a list of images.
 
  @param    outvid    output video
  @param    images    list of images to use in the video
  @param    fps     frame per second
  @param    size    size of each frame
  @param    is_color  color
  @param    format    see http://www.fourcc.org/codecs.php
  @return         see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
  The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
  By default, the video will have the size of the first image.
  It will resize every image to this size before adding them to the video.
  """
  fourcc = cv2.VideoWriter_fourcc(*format)
  vid = None
  for image in images:
    if not os.path.exists(image):
      raise FileNotFoundError(image)
    img = cv2.imread(image)
    if vid is None:
      if size is None:
        size = img.shape[1], img.shape[0]
      vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
    if size[0] != img.shape[1] and size[1] != img.shape[0]:
      img = cv2.resize(img, size)
    vid.write(img)
  vid.release()
  return vid

def decode_fourcc(fourcc, debug=False):
  """ Decodes fourcc value
  """
  fourcc_int = int(fourcc)
  if debug: print("int value of fourcc: '{}'".format(fourcc_int))

  fourcc_decode = ""
  for i in range(4):
    int_value = fourcc_int >> 8 * i & 0xFF
    if debug: print("int_value: '{}'".format(int_value))
    fourcc_decode += chr(int_value)
  return fourcc_decode

def get_video_info(video_path):
  VIDEO_FILE_PATH = os.path.expanduser(os.path.expandvars(video_path))
  video = cv2.VideoCapture(VIDEO_FILE_PATH)
  num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = video.get(cv2.CAP_PROP_FPS)
  fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
  fourcc_dec = decode_fourcc(fourcc)
  video.release()
  print("frames: {}, fps: {}, width: {}, height: {}, fourcc_dec/fourcc: {}/{}".format(num_frames, fps, frame_width, frame_height, fourcc_dec, fourcc))
  return num_frames, fps, frame_width, frame_height, fourcc

def process_video():
  cap = cv2.VideoCapture("input_file.mp4")
  out = cv2.VideoWriter("output_file.avi", ...)
  while (cap.isOpened()):
    ret, frame = cap.read()
    # ... DO SOME STUFF TO frame... #
    out.write(frame)

def main():
  args = parser.parse_args()
  scale = 4
  scale2 = 1

  # Get all image paths
  input_video_path = os.path.expanduser(os.path.expandvars(args.input_video))
  output_video_path = os.path.expanduser(os.path.expandvars(args.output_video))

  # Change model input shape to accept all size inputs
  model_path = os.path.expanduser(os.path.expandvars(args.model))
  model = keras.models.load_model(model_path)
  inputs = keras.Input((None, None, 3))
  output = model(inputs)
  model = keras.models.Model(inputs, output)

  # Get video info
  num_frames, fps, frame_width, frame_height, fourcc = get_video_info(input_video_path)
  size = frame_width * scale * scale2, frame_height * scale * scale2
  is_color = True

  # Set new variables
  fourcc = cv2.VideoWriter_fourcc('R','G','B','A')
  # fps = 25.0
  # frame_start = 800
  
  # Open video capture
  cap = cv2.VideoCapture(input_video_path)
  # cap.set(cv2.CAP_PROP_FPS, fps)
  cap.set(cv2.CAP_PROP_FOURCC, fourcc)
  # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

  out = cv2.VideoWriter(output_video_path, fourcc, fps, size, is_color)

  while (cap.isOpened()):
    frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("Reading frame: {}/{}".format(frame_pos, num_frames))
    ret, frame = cap.read()
    if not ret: 
      break

    ## Preprocess Image and Run Inference

    # Convert to RGB (opencv uses BGR as default)
    # print("  Color conversion:")
    # print(f"  frame type: {type(frame)}, dtype: {frame.dtype}, shape: {frame.shape}")
    low_res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # low_res = cv2.resize(low_res, (frame_width*scale2, frame_height*scale2))
    # Rescale to 0-1.
    low_res = low_res / 255.0

    # Get super resolution image
    # print("  Performing Inference")
    # print(f"  frame type: {type(low_res)}, dtype: {low_res.dtype}, shape: {low_res.shape}")
    frame_sr = model.predict(np.expand_dims(low_res, axis=0))[0]
    # print("  Inference Complete")
    # print(f"  frame type: {type(frame_sr)}, dtype: {frame_sr.dtype}, shape: {frame_sr.shape}")

    # Rescale values in range 0-255
    frame_sr = ((frame_sr + 1) / 2.) * 255

    # Convert back to BGR for opencv
    frame_sr = cv2.cvtColor(frame_sr, cv2.COLOR_RGB2BGR).astype(np.uint8)
    # print("  Converting back to uint8")
    # print(f"  frame type: {type(frame_sr)}, dtype: {frame_sr.dtype}, shape: {frame_sr.shape}")
    if size[0] != frame_sr.shape[1] and size[1] != frame_sr.shape[0]:
      # print("Frame size does not match intial size of '{}'".format(size))
      frame_sr = cv2.resize(frame_sr, size)

    # Display outgoing frame
    # cv2.imshow("Incoming frame", low_res)

    # Display outgoing frame
    # cv2.imshow("Outgoing frame", frame_sr)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   out.write(frame_sr)
    #   break

    out.write(frame_sr)

  cap.release()
  out.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
