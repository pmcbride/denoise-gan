import tensorflow as tf

# CREATE AUTOENCODER
class Autoencoder(object):
  """ Denoising Autoencoder """

  def __init__(self, args):
    """
    Initialized the Autoencoder class.
    Args:
      args: CLI arguments
    Returns:
      None
    """
    self.hr_height = args.crop_size
    self.hr_width = args.crop_size
    self.lr_height = self.hr_height #// 4  # Low resolution height
    self.lr_width = self.hr_width #// 4  # Low resolution width
    self.lr_shape = (self.lr_height, self.lr_width, 3)
    self.hr_shape = (self.hr_height, self.hr_width, 3)
    self.iterations = 0
    self.epochs = 0
    self.retrain = bool(args.retrain)

    # Define a learning rate decay schedule.
    self.ae_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      args.lr,
      decay_steps=100000,
      decay_rate=0.1,
      staircase=True
    )

    self.disc_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      args.lr * 5,  # TTUR - Two Time Scale Updates
      decay_steps=100000,
      decay_rate=0.1,
      staircase=True
    )

    # Define Optimzer
    self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self.ae_schedule)
    self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.disc_schedule)

    # We use a pre-trained VGG19 model to extract image features from the high resolution
    # and the generated high resolution images and minimize the mse between them
    self.vgg = self.build_vgg()
    self.vgg.trainable = False

    # Calculate output shape of D (PatchGAN)
    patch = int(self.hr_height / 2 ** 4)
    self.disc_patch = (patch, patch, 1)

    # Number of filters in the first layer of G and D
    self.gf = 32  # Realtime Image Enhancement GAN Galteri et al.
    self.df = 32

    # Build and compile the autoencoder
    self.generator = self.build_autoencoder()

    # Build and compile the discriminator
    self.discriminator = self.build_discriminator()


  @tf.function
  def content_loss(self, hr, sr):
    sr = tf.keras.applications.vgg19.preprocess_input(((sr + 1.0) * 255) / 2.0)
    hr = tf.keras.applications.vgg19.preprocess_input(((hr + 1.0) * 255) / 2.0)
    sr_features = self.vgg(sr) / 12.75
    hr_features = self.vgg(hr) / 12.75
    return tf.keras.losses.MeanSquaredError()(hr_features, sr_features)

  def build_vgg(self):
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    # Get the vgg network. Extract features from Block 5, last convolution.
    vgg = tf.keras.applications.VGG19(weights="imagenet", input_shape=self.hr_shape, include_top=False)
    vgg.trainable = False
    for layer in vgg.layers:
      layer.trainable = False

    # Create model and compile
    model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)

    return model

  # Build Autoencoder
  def build_autoencoder(self, name='Autoencoder'):

    def conv2d(x, filters, kernel_size=(3, 3), strides=(1, 1), name='conv', relu=True):
      with tf.name_scope(name):
        if relu:
          kernel_initializer = tf.keras.initializers.he_normal()
          res = tf.keras.layers.Conv2D(filters, kernel_size, strides,
                      padding='same',
                      activation='relu',
                      kernel_initializer=kernel_initializer)(x)
        else:
          kernel_initializer = tf.keras.initializers.lecun_normal()
          res = tf.keras.layers.Conv2D(filters, kernel_size, strides,
                      padding='same',
                      activation='tanh',
                      kernel_initializer=kernel_initializer)(x)
      return res

    def maxpool2d(x, k=2, name='pool'):
      # MaxPool2D wrapper
      with tf.name_scope(name):
        res = tf.keras.layers.MaxPool2D(pool_size=(k, k), strides=(k, k), padding='same')(x)
      return res

    def unpool(x, name='unpool'):
      with tf.name_scope(name):
        in_shape = x.get_shape().as_list()
        in_channel = in_shape[-1]
        out_channel = in_channel

        filter_shape = [2, 2, in_channel, out_channel]
        out_shape = [in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]]

        res = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        # res = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(2,2), padding='same')(x)
        res = tf.keras.activations.relu(res)
        # layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        # tf.keras.layers.BatchNormalization()(x)
        # tf.keras.layers.LeakyReLU())
        #res = tf.nn.conv2d_transpose(value, W, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME')
      return res

    def unpool_concat(a, b, name='upconcat'):
      with tf.name_scope(name):
        up = unpool(a)
        # res = tf.keras.layers.concatenate([up, b], axis=-1)
        res = tf.concat([up, b], 3)
      return res
    
    # with tf.name_scope(name):

    # Build Autoencoder Model
    if self.retrain == True:
      print("Loading model from models/autoencoder.h5")
      model = tf.keras.models.load_model("models/autoencoder.h5")
      img_in = tf.keras.Input(shape=self.lr_shape)
      img_out = model(img_in)
      return tf.keras.models.Model(img_in, img_out)
    else:
      img_lr = tf.keras.Input(shape=self.lr_shape)

      prevLayer = conv1  = conv2d(img_lr, 32, name='conv1')
      prevLayer = conv1b = conv2d(prevLayer, 32, name='conv1b')
      prevLayer = pool1  = maxpool2d(prevLayer, 2, name='pool1') # 256 -> 128

      prevLayer = conv2 = conv2d(prevLayer, 44, name='conv2')
      prevLayer = pool2 = maxpool2d(prevLayer, 2, name='pool2') # 128 -> 64

      prevLayer = conv3 = conv2d(prevLayer, 56, name='conv3')
      prevLayer = pool3 = maxpool2d(prevLayer, 2, name='pool3') # 64 -> 32

      prevLayer = conv4 = conv2d(prevLayer, 76, name='conv4')
      prevLayer = pool4 = maxpool2d(prevLayer, 2, name='pool4') # 32 -> 16

      prevLayer = conv5 = conv2d(prevLayer,  100, name='conv5')
      prevLayer = pool5 = maxpool2d(prevLayer, 2, name='pool5') # 16 -> 8

      prevLayer = us6 = unpool_concat(prevLayer, pool4, name='unpool4')
      prevLayer = conv6 = conv2d(prevLayer,  152, name='conv6')
      prevLayer = conv6b = conv2d(prevLayer, 152, name='conv6b')

      prevLayer = us7 = unpool_concat(prevLayer, pool3, name='unpool3')
      prevLayer = conv7 = conv2d(prevLayer, 112, name='conv7')
      prevLayer = conv7b = conv2d(prevLayer, 112, name='conv7b')

      prevLayer = us8 = unpool_concat(prevLayer, pool2, name='unpool2')
      prevLayer = conv8 = conv2d(prevLayer, 84, name='conv8')
      prevLayer = conv8b = conv2d(prevLayer, 84, name='conv8b')

      prevLayer = us9 = unpool_concat(prevLayer, pool1, name='unpool1')
      prevLayer = conv9 = conv2d(prevLayer,  64, name='conv9')
      prevLayer = conv9b = conv2d(prevLayer, 64, name='conv9b')

      prevLayer = us10 = unpool_concat(prevLayer, img_lr, name='unpool0')
      prevLayer = conv10 = conv2d(prevLayer, 64, name='conv10')
      prevLayer = conv10b = conv2d(prevLayer, 32, name='conv10b')

      gen_hr = conv2d(prevLayer, 3, name='conv11', relu=False)

      return tf.keras.models.Model(img_lr, gen_hr, name=name)
  
  def build_discriminator(self, name='Discriminator'):
    """Builds a discriminator network based on the SRGAN design."""
    def d_block(layer_input, filters, strides=1, bn=True):
      """Discriminator layer block.
      Args:
          layer_input: Input feature map for the convolutional block.
          filters: Number of filters in the convolution.
          strides: The stride of the convolution.
          bn: Whether to use batch norm or not.
      """
      d = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
      if bn:
          d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
      d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
          
      return d

    # Build Discriminator Model
    if self.retrain == True:
      print("Loading model from models/discriminator_ae.h5")
      model = tf.keras.models.load_model("models/discriminator_ae.h5")
      img_in = tf.keras.layers.Input(shape=self.hr_shape)
      validity = model(img_in)
      return tf.keras.models.Model(img_in, validity)
    else:
      # Input img
      d0 = tf.keras.layers.Input(shape=self.hr_shape)

      d1 = d_block(d0, self.df, bn=False)
      d2 = d_block(d1, self.df, strides=2)
      d3 = d_block(d2, self.df)
      d4 = d_block(d3, self.df, strides=2)
      d5 = d_block(d4, self.df * 2)
      d6 = d_block(d5, self.df * 2, strides=2)
      d7 = d_block(d6, self.df * 2)
      d8 = d_block(d7, self.df * 2, strides=2)

      validity = tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(d8)

      return tf.keras.models.Model(d0, validity, name=name)