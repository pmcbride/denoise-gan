from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
# Currently, memory growth needs to be the same across GPUs
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


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

def im2patch(img, crop=256):
  shape = img.shape
  return tf.reshape(tf.nn.space_to_depth(img, crop), [-1, crop, crop, shape[-1]])
  
def patch2im(imgs, patch_shape=[4,4]):
  shape = imgs.shape
  h = patch_shape[0] * shape[1]
  w = patch_shape[1] * shape[2]
  return tf.nn.depth_to_space(tf.reshape(imgs, [1, patch_shape[0], patch_shape[1], -1]), shape[1])

def laplacian(image):
  if tf.is_tensor(image):
    image = image.numpy()
  shape = image.shape
  if shape[0]==1:
    image = np.squeeze(image, axis=0)
  return np.reshape(cv2.Laplacian(image, cv2.CV_32F), shape)

def get_path(path):
  return os.path.realpath(os.path.expanduser(os.path.expandvars(path)))

def main(args):
  # Get all image paths
  input_video_path = get_path(args.input_video)
  output_video_path = get_path(args.output_video)
  
  scale = 4

  # Get video info
  # num_frames, fps, frame_width, frame_height, fourcc_code
  num_frames, fps, fw, fh, fourcc = get_video_info(input_video_path)
  model_scale = 1
  crop = 256
  new_h = (fh + 256 * model_scale) - fh % (256 * model_scale)
  # new_h = (fh) - fh % (256 * model_scale)
  new_w = (fw + 256 * model_scale) - fw % (256 * model_scale)
  # new_w = (fw) - fw % (256 * model_scale)
  resize_h = new_h // model_scale
  resize_w = new_w // model_scale
  size = new_w * scale, new_h * scale
  print(f"org size: {(fh, fw)}")
  print(f"new size: {(new_h, new_w)}")
  print(f"size: {size}")
  is_color = True

  # Change model input shape to accept all size inputs
  model_path = get_path(args.model)
  model = keras.models.load_model(model_path)
  inputs = keras.Input((None, None, 3))
  outputs = model(inputs, training=False)
  model = keras.Model(inputs, outputs)
  
  # Create Checkpoint
  # ckpt_dir = get_path('models/checkpoints/fsrgan')
  # # os.makedirs(ckpt_dir, exist_ok=True)
  # ckpt = tf.train.Checkpoint(generator=generator,
  #                            discriminator=discriminator)
  # ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
  # if bool(args.retrain) == True:
  #   ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

  def set_frame(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

  # Set new variables
  # fourcc = cv2.VideoWriter_fourcc('R','G','B','A')
  # fps = 25.0
  frame_start = 1600

  # Open video capture
  cap = cv2.VideoCapture(input_video_path)
  # cap.set(cv2.CAP_PROP_FPS, fps)
  # cap.set(cv2.CAP_PROP_FOURCC, fourcc)
  # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
  set_frame(cap, frame_start)
  #size = [256, 256]

  # out = cv2.VideoWriter(output_video_path, fourcc, fps, size, is_color)

  while (cap.isOpened()):
    frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("Reading frame: {}/{}".format(frame_pos, num_frames))
    ret, frame = cap.read()
    if not ret:
      break

    ## Preprocess Image and Run Inference

    # Convert to RGB (opencv uses BGR as default)
    # cv2.imwrite(f"video_out/frame_{frame_pos}.jpg", frame)

    frame_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_in = tf.image.convert_image_dtype(frame_in, 'float32')
    frame_in = tf.image.resize_with_crop_or_pad(frame_in, new_h, new_w)
    # frame_in = cv2.resize(frame_in, (new_w // model_scale, new_h // model_scale), cv2.INTER_AREA)
    # frame_in = tf.image.resize(frame_in, [resize_h, resize_w], method='bicubic', preserve_aspect_ratio=True, antialias=True)
    frame_in = frame_in * 2 - 1

    # Get super resolution image
    frame_out = model(frame_in[tf.newaxis, ...], training=False)[0].numpy()
    # frame_out = model(im2patch(frame_in[tf.newaxis, ...], crop), training=False)
    # frame_out = patch2im(frame_out, patch_shape=[resize_h//crop, resize_w//crop])[0].numpy()
    frame_out = frame_out * 0.5 + 0.5
    # Rescale values in range 0-255
    # frame_out = tf.image.convert_image_dtype(frame_out * 0.5 + 0.5, 'uint8')
    # frame_out = frame_out.astype(np.uint8)
    # frame_out = cv2.resize(frame_out, (new_w, new_h), cv2.INTER_CUBIC)
    # frame_out = tf.image.resize(frame_out, [new_h, new_w], method='bicubic', preserve_aspect_ratio=True, antialias=True)
    # frame_out = frame_out.numpy()
    frame_out = tf.clip_by_value(tf.image.resize_with_crop_or_pad(frame_out, fh*scale, fw*scale), 0, 1).numpy()
    # frame_lap = tf.image.convert_image_dtype(tf.clip_by_value(frame_out - laplacian(frame_out), 0, 1), 'uint8').numpy()
    frame_out = frame_out * 255
    frame_out = frame_out.astype(np.uint8)

    frame_in = frame_in * 0.5 + 0.5
    frame_in = tf.image.resize_with_crop_or_pad(frame_in, fh, fw).numpy()
    frame_in = cv2.resize(frame_in, (fw*4, fh*4), cv2.INTER_CUBIC)
    frame_in = frame_in * 255
    frame_in = frame_in.astype(np.uint8)
    # Convert back to BGR for opencv
    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
    # frame_lap = cv2.cvtColor(frame_lap, cv2.COLOR_RGB2BGR)
    frame_in = cv2.cvtColor(frame_in, cv2.COLOR_RGB2BGR)
    # if size[0] != frame_out.shape[1] and size[1] != frame_out.shape[0]:
      # frame_out = cv2.resize(frame_out, size)

    # Display outgoing frame
    cv2.imshow("Incoming frame", frame_in)

    # Display outgoing frame
    cv2.imshow("Outgoing frame", frame_out)
    # cv2.imshow("Outgoing laplacian", frame_lap)


    if cv2.waitKey(1) & 0xFF == ord('q'):
      # out.write(frame_out)
      break

    # out.write(frame_out)

  cap.release()
  # out.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  pix2pix = "./saved_models/pix2pix_jpeg_0105.h5"
  fsrgan = "./saved_models/fsrgan_0114_0227.h5"
  autoencoder = "./saved_models/autoencoder_jpeg_mse_0110.h5"

  parser = ArgumentParser()
  parser.add_argument('--input_video', default="./video_in/8minanal_240P.mp4", type=str, help='Path to input video')
  parser.add_argument('--output_video', default="./video_out/out.mp4", type=str, help='Path to output high res video.')
  parser.add_argument('--model', default=fsrgan, type=str, help='Path to model.')
  args = parser.parse_args()
  main(args)
