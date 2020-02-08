import tensorflow as tf
import os
import cv2
import glob

def get_path(path):
  return os.path.realpath(os.path.expanduser(os.path.expandvars(path)))

class DataLoader(object):
  """Data Loader for the SR GAN, that prepares a tf data object for training."""

  def __init__(self, args):
    """
    Initializes the dataloader.
    Args:
        image_dir: The path to the directory containing high resolution images.
        crop_size: Integer, the crop size of the images to train on (High
                    resolution images will be cropped to this width and height).
    Returns:
        The dataloader object.
    """
    self.image_dir = args.image_dir
    self.crop_size = args.crop_size
    self.scale = args.scale
    self.jpeg_quality = args.jpeg_quality
    self.batch_size = args.batch_size
    # self.image_paths = [os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir)]
    self.image_paths = glob.glob(os.path.join(self.image_dir, "*/*"))
    self.train_size = len(self.image_paths)

  def load_image(self, image_path):
    """
    Function that loads the images given the path.
    Args:
        image_path: Path to an image file.
    Returns:
        image: A tf tensor of the loaded image.
    """

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Check if image is large enough
    if tf.keras.backend.image_data_format() == 'channels_last':
     shape = tf.shape(image)[:-1]
    else:
     shape = tf.shape(image)[1:]

    cond = tf.reduce_all(shape >= tf.constant(self.crop_size))
    ## If input image is smaller than crop_size, then resize image to [crop_size, crop_size]
    ## tf.cond(pred, true_fn=None, false_fn=None, name=None)
    ## image = tf.cond(cond, lambda: tf.identity(image),
    ##                 lambda: tf.image.resize(image, [self.crop_size, self.crop_size]))
    if not cond:
     image = tf.image.resize(image, [self.crop_size, self.crop_size])

    return image

  def load_tiff(self, image_path):
    """
    Function that loads tiff image given path and converts to tf tensor
    """
    def imread(path):
      path = path.numpy().decode("utf-8")
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      return img

    image = tf.py_function(func=imread, inp=[image_path], Tout=tf.uint8)
    image = tf.expand_dims(image, axis=-1)

    return image

  def random_crop(self, image):
    cropped_image = tf.image.random_crop(image, size=[self.crop_size, self.crop_size, 3])
    return cropped_image

  def stack_crop(self, image_input, image_target):
    """
    Function that crops the image according a defined width
    and height.
    Args:
        image: A tf tensor of an image.
    Returns:
        image: A tf tensor of containing the cropped image.
    """
    #assert image_input.shape == image_target.shape
    
    stacked_image = tf.stack([image_input, image_target], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, [2, self.crop_size, self.crop_size, 3])

    return cropped_image[0], cropped_image[1]

  def generate_image_pairs(self, high_res):
    """
    Function that generates a low resolution image given the
    high resolution image. The downsampling factor is 4x.
    Args:
        high_res: A tf tensor of the high res image.
    Returns:
        low_res: A tf tensor of the low res image.
        high_res: A tf tensor of the high res image.
    """

    low_res = tf.identity(high_res)

    return low_res, high_res

  def scale_image(self, low_res, high_res):
    """
    Function that generates a low resolution image given the
    high resolution image. The downsampling factor is 4x.
    Args:
        high_res: A tf tensor of the high res image.
    Returns:
        low_res: A tf tensor of the low res image.
        high_res: A tf tensor of the high res image.
    """
    scale = self.scale
    low_res = tf.image.resize(high_res,
                              [self.crop_size // scale, self.crop_size // scale],
                              method='bicubic')

    return low_res, high_res

  def adjust_jpeg_quality(self, low_res, high_res):
    """
    Function that rescales the pixel values to the -1 to 1 range.
    For use with the generator output tanh function.
    Args:
        low_res: The tf tensor of the low res image.
        high_res: The tf tensor of the high res image.
    Returns:
        low_res: The tf tensor of the low res image, rescaled.
        high_res: the tf tensor of the high res image, rescaled.
    """
    low_res = tf.image.adjust_jpeg_quality(low_res, jpeg_quality=self.jpeg_quality)

    return low_res, high_res

  def random_jpeg_quality(self, image_input, image_target):
    """
    Function that rescales the pixel values to the -1 to 1 range.
    For use with the generator output tanh function.
    Args:
        low_res: The tf tensor of the low res image.
        high_res: The tf tensor of the high res image.
    Returns:
        low_res: The tf tensor of the low res image, rescaled.
        high_res: the tf tensor of the high res image, rescaled.
    """
    shape = image_input.shape
    image_input = tf.image.random_jpeg_quality(image_target,
                                           min_jpeg_quality=25,
                                           max_jpeg_quality=75)
    #print(f"JPEG Image shape: {tf.shape(image)}")
    #assert image.shape == shape
    return image_input, image_target

  def normalize(self, image_input, image_target):
    """
    Function that rescales the pixel values to the -1 to 1 range.
    For use with the generator output tanh function.
    Args:
        image_input: The tf tensor of the low res image.
        image_target: The tf tensor of the high res image.
    Returns:
        image_input: The tf tensor of the low res image, rescaled.
        image_target: the tf tensor of the high res image, rescaled.
    """
    # image_input = tf.image.convert_image_dtype(image_input, tf.float32) * 2 - 1
    image_input = image_input * 2 - 1
    # image_target = tf.image.convert_image_dtype(image_target, tf.float32) * 2 - 1
    image_target = image_target * 2 - 1

    return image_input, image_target

  # def preprocess_image_train(self, image_path):
  #   image_target = self.load_image(image_path)
  #   image_target = self.random_crop(image_target)
  #   image_input = self.scale_image(image_target)
  #   image_input = self.adjust_jpeg_quality(image_input)
  #   image_input = self.normalize(image_input)
  #   image_target = self.normalize(image_target)
  #   return image_input, image_target

  def dataset(self):
    """
    Returns a tf dataset object with specified mappings.
    Args:
        batch_size: Int, The number of elements in a batch returned by the dataset.
        threads: Int, CPU threads to use for multi-threaded operation.
    Returns:
        dataset: A tf dataset object.
    """

    # Generate tf dataset from high res image paths.
    # dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
    dataset = tf.data.Dataset.list_files(os.path.join(self.image_dir, "*/*"))
    #ds_target = tf.data.Dataset.list_files(os.path.join(self.image_dir, "*"))

    # Read the images
    dataset = dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Generate jpeg artifacts by jpeg compressing low res crop.
    dataset = dataset.map(self.generate_image_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = tf.data.Dataset.zip((dataset, dataset))

    dataset = dataset.map(self.stack_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(self.scale_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.map(self.random_jpeg_quality, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(self.adjust_jpeg_quality, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Normalize the values in the input
    dataset = dataset.map(self.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the input, drop remainder to get a defined batch size.
    # Prefetch the data for optimal GPU utilization.

    dataset = dataset.cache().shuffle(self.train_size).batch(self.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.cache().shuffle(self.train_size).batch(self.batch_size, drop_remainder=True).repeat(1).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.batch(self.batch_size, drop_remainder=True)
    # dataset = dataset.shuffle(buffer_size=self.train_size)
    # dataset = dataset.repeat(1)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    return dataset
