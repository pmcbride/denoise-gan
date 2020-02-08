import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class SRGAN(object):
  """SRGAN for fast super resolution."""

  def __init__(self, args):
    """
    Initializes the Mobile SRGAN class.
    Args:
        args: CLI arguments that dictate how to build the model.
    Returns:
        None
    """
    self.scale = args.scale
    self.hr_height = args.crop_size
    self.hr_width = args.crop_size
    self.lr_height = self.hr_height // self.scale  # Low resolution height
    self.lr_width = self.hr_width // self.scale  # Low resolution width
    self.lr_shape = [self.lr_height, self.lr_width, 3]
    self.hr_shape = [self.hr_height, self.hr_width, 3]
    self.iterations = 0
    self.epochs = 0
    self.fp16 = args.fp16

    # Define Loss objects
    self.gen_loss_object = tf.keras.losses.MeanSquaredError()
    self.disc_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define a learning rate decay schedule.
    self.gen_schedule = keras.optimizers.schedules.ExponentialDecay(
        args.lr,
        decay_steps=100000,
        decay_rate=0.1,
        staircase=True
    )

    self.disc_schedule = keras.optimizers.schedules.ExponentialDecay(
        args.lr * 5,  # TTUR - Two Time Scale Updates
        decay_steps=100000,
        decay_rate=0.1,
        staircase=True
    )

    self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self.gen_schedule)
    self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.disc_schedule)

    # We use a pre-trained VGG19 model to extract image features from the high resolution
    # and the generated high resolution images and minimize the mse between them
    self.vgg = self.build_vgg()
    self.vgg.trainable = False

    # Build and compile the generator for pretraining.
    self.generator = self.build_generator()

    # Build and compile the discriminator
    # self.discriminator = self.build_discriminator_srgan()
    self.discriminator = self.build_discriminator()

    if self.fp16:
      # Modify Optimizers
      self.gen_optimizer = mixed_precision.LossScaleOptimizer(self.gen_optimizer, loss_scale='dynamic')
      self.disc_optimizer = mixed_precision.LossScaleOptimizer(self.disc_optimizer, loss_scale='dynamic')

  @tf.function
  def content_loss(self, target, gen_output):
    gen_output = tf.keras.applications.vgg19.preprocess_input(((gen_output + 1.0) * 255) / 2.0)
    target = tf.keras.applications.vgg19.preprocess_input(((target + 1.0) * 255) / 2.0)
    gen_features = self.vgg(gen_output) / 12.75
    target_features = self.vgg(target) / 12.75
    return tf.keras.losses.MeanSquaredError()(target_features, gen_features)

  def build_vgg(self):
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    # with tf.name_scope("VGG"):
    # Get the vgg network. Extract features from Block 5, last convolution.
    # input_shape = self.hr_shape
    input_shape = (None, None, 3)
    vgg = keras.applications.VGG19(weights="imagenet", input_shape=input_shape, include_top=False)
    vgg.trainable = False
    for layer in vgg.layers:
      layer.trainable = False

    # Create model and compile
    vgg = keras.Model(inputs=vgg.inputs, outputs=keras.layers.Activation('linear', dtype='float32')(vgg.get_layer("block5_conv4").output))
    return vgg

  # SRGAN Generator Loss
  @tf.function
  def generator_loss(self, disc_generated_output, gen_output, target):
    adv_loss = 1e-3 * self.disc_loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # Total Variation error
    var_loss = 1e-5 * tf.reduce_mean(tf.image.total_variation(target - gen_output))
    
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Mean Squared Error
    l2_loss = self.gen_loss_object(target, gen_output)
    
    # VGG Content Loss
    cont_loss = self.content_loss(target, gen_output)

    # Identity Loss
    # identity_loss = tf.reduce_mean(tf.abs(self.generator(target, training=True) - target))

    total_gen_loss = adv_loss + l2_loss + cont_loss #+ var_loss + l1_loss #+ identity_loss

    return total_gen_loss, adv_loss, l1_loss, l2_loss, cont_loss, var_loss#, identity_loss

  @tf.function
  def discriminator_loss(self, disc_real_output, disc_generated_output):
    real_loss = self.disc_loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = self.disc_loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

  def build_generator(self):
    w_init = tf.random_normal_initializer(0., 0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    # bn_layer = tf.keras.layers.BatchNormalization(gamma_initializer=g_init)

    def deconv2d(layer_input, filters=256):
      """
      Upsampling layer to increase height and width of the input.
      Uses PixelShuffle for upsampling.
      Args:
          layer_input: The input tensor to upsample.
          filters: Numbers of expansion filters.
      Returns:
          u: Upsampled input by a factor of 2.
      """
      u = Conv2D(filters, (3, 3), (1, 1), padding='same', kernel_initializer=w_init)(layer_input)
      u = tf.nn.depth_to_space(u, 2)
      u = keras.layers.PReLU(shared_axes=[1, 2])(u)
      return u

    # input_shape = self.lr_shape
    input_shape = (None, None, 3)
    # nin = Input(input_shape)
    inputs = keras.Input(input_shape)
    nin = inputs
    n = Conv2D(64, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, use_bias=False)(nin)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    # n = bn_layer(n)
    n = keras.layers.PReLU(shared_axes=[1, 2])(n)
    temp = n

    # B residual blocks
    for i in range(16):
      nn = Conv2D(64, (3, 3), (1, 1), padding='SAME', activation=None, kernel_initializer=w_init, use_bias=False)(n)
      nn = BatchNormalization(gamma_initializer=g_init)(nn)
      # nn = bn_layer(nn)
      nn = keras.layers.Activation('relu')(nn)
      nn = Conv2D(64, (3, 3), (1, 1), padding='SAME', activation=None, kernel_initializer=w_init, use_bias=False)(nn)
      nn = BatchNormalization(gamma_initializer=g_init)(nn)
      # nn = bn_layer(nn)
      nn = keras.layers.Add(name=f'block_{i}_add')([n, nn])
      n = nn

    n = Conv2D(64, (3, 3), (1, 1), padding='SAME', activation=None, kernel_initializer=w_init, use_bias=False)(n)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    # n = bn_layer(n)
    n = keras.layers.Add()([n, temp])
    # B residual blocks end

    # Deconv scale // 2 times
    for i in range(self.scale // 2):
      n = deconv2d(n, 256)

    gen_outputs = Conv2D(3, (1, 1), (1, 1), padding='SAME', kernel_initializer=w_init)(n)
    gen_outputs = keras.layers.Activation('tanh', dtype='float32', name='generator_tanh')(gen_outputs)

    return tf.keras.Model(inputs=inputs, outputs=gen_outputs, name="generator")


  def build_discriminator_srgan(self):
    w_init = tf.random_normal_initializer(mean=0., stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1., stddev=0.02)
    # bn_layer = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)
    df_dim = 64
    # lrelu = keras.layers.LeakyReLU(alpha=0.2)

    def disc_block(layer_input, filters=df_dim, kernel_size=4, strides=2, bn=True, lrelu=True):
      if bn:
        d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME', kernel_initializer=w_init, use_bias=False)(layer_input)
        d = keras.layers.BatchNormalization(gamma_initializer=gamma_init)(d)
        # d = bn_layer(d)
      else:
        d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME', kernel_initializer=w_init, use_bias=True)(layer_input)
      if lrelu:
        d = keras.layers.LeakyReLU(alpha=0.2)(d)
      return d

    # input_shape = self.hr_shape
    input_shape = (None, None, 3)
    # nin = Input(input_shape)
    inputs = tf.keras.Input(input_shape)
    nin = inputs

    n = disc_block(nin, filters=df_dim, kernel_size=4, strides=2, bn=False)       # (b, h/2, w/2, 64)
    n = disc_block(n, filters=df_dim * 2, kernel_size=4, strides=2)               # (b, h/4, w/4, 128)
    n = disc_block(n, filters=df_dim * 4, kernel_size=4, strides=2)               # (b, h/8, w/8, 256)
    n = disc_block(n, filters=df_dim * 8, kernel_size=4, strides=2)               # (b, h/16, w/16, 512)
    n = disc_block(n, filters=df_dim * 16, kernel_size=4, strides=2)              # (b, h/32, w/32, 1024)
    n = disc_block(n, filters=df_dim * 32, kernel_size=4, strides=2)              # (b, h/64, w/64, 2048)
    n = disc_block(n, filters=df_dim * 16, kernel_size=1, strides=1)              # (b, h/64, w/64, 1024)
    nn = disc_block(n, filters=df_dim * 8, kernel_size=1, strides=1, lrelu=False) # (b, h/64, w/64, 512)
    n = disc_block(nn, filters=df_dim * 2, kernel_size=1, strides=1)              # (b, h/64, w/64, 128)
    n = disc_block(n, filters=df_dim * 2, kernel_size=3, strides=1)               # (b, h/64, w/64, 128)
    n = disc_block(n, filters=df_dim * 8, kernel_size=3, strides=1, lrelu=False)  # (b, h/64, w/64, 512)
    n = keras.layers.Add()([n, nn])

    # n = keras.layers.Flatten()(n)
    # outputs = keras.layers.Dense(1, kernel_initializer=w_init)(n)
    # D = Model(inputs=nin, outputs=outputs, name="discriminator")
    outputs = keras.layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), kernel_initializer=w_init, padding='same')(n)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="discriminator")

  def build_discriminator(self):
    """Builds a discriminator network based on the Fast-SRGAN design."""
    # with tf.name_scope("Discriminator"):
    # bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
    df_dim = 32

    def d_block(layer_input, filters, strides=1, bn=True):
      """Discriminator layer block.
      Args:
          layer_input: Input feature map for the convolutional block.
          filters: Number of filters in the convolution.
          strides: The stride of the convolution.
          bn: Whether to use batch norm or not.
      """
      d = keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
      if bn:
        d = keras.layers.BatchNormalization(momentum=0.8)(d)
        # d = bn_layer(d)
      d = keras.layers.LeakyReLU(alpha=0.2)(d)
          
      return d

    # Input img
    # input_shape = self.hr_shape
    input_shape = (None, None, 3)
    disc_inputs = keras.Input(input_shape) #, dtype='float32')
    d0 = disc_inputs
    d1 = d_block(d0, df_dim, bn=False)
    d2 = d_block(d1, df_dim, strides=2)
    d3 = d_block(d2, df_dim)
    d4 = d_block(d3, df_dim, strides=2)
    d5 = d_block(d4, df_dim * 2)
    d6 = d_block(d5, df_dim * 2, strides=2)
    d7 = d_block(d6, df_dim * 2)
    d8 = d_block(d7, df_dim * 2, strides=2)

    logits = keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(d8)
    # disc_outputs = keras.layers.Activation('sigmoid', dtype='float32', name='discriminator_sigmoid')(logits)
    disc_outputs = keras.layers.Activation('linear', dtype='float32', name='discriminator_logits')(logits)

    return keras.Model(inputs=disc_inputs, outputs=disc_outputs)

# def get_G2(input_shape):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     g_init = tf.random_normal_initializer(1., 0.02)
#
#     n = InputLayer(t_image, name='in')
#     n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', kernel_initializer=w_init, name='n64s1/c')
#     temp = n
#
#     # B residual blocks
#     for i in range(16):
#         nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
#         nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
#         nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
#         nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
#         nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
#         n = nn
#
#     n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n64s1/c/m')
#     n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
#     n = ElementwiseLayer([n, temp], tf.add, name='add3')
#     # B residual blacks end
#
#     # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', kernel_initializer=w_init, name='n256s1/1')
#     # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
#     #
#     # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', kernel_initializer=w_init, name='n256s1/2')
#     # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')
#
#     ## 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
#     n = UpSampling2dLayer(n, size=[size[1] * 2, size[2] * 2], is_scale=False, method=1, align_corners=False, name='up1/upsample2d')
#     n = Conv2d(n, 64, (3, 3), (1, 1), padding='SAME', kernel_initializer=w_init, b_init=b_init, name='up1/conv2d')  # <-- may need to increase n_filter
#     n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up1/batch_norm')
#
#     n = UpSampling2dLayer(n, size=[size[1] * 4, size[2] * 4], is_scale=False, method=1, align_corners=False, name='up2/upsample2d')
#     n = Conv2d(n, 32, (3, 3), (1, 1), padding='SAME', kernel_initializer=w_init, b_init=b_init, name='up2/conv2d')  # <-- may need to increase n_filter
#     n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up2/batch_norm')
#
#     n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', kernel_initializer=w_init, name='out')
#     return n


# def SRGAN_d2(t_image, is_train=False, reuse=False):
#     """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
#     feature maps (n) and stride (s) feature maps (n) and stride (s)
#     """
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     b_init = None
#     g_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tl.act.lrelu(x, 0.2)
#     with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
#         # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
#         n = InputLayer(t_image, name='in')
#         n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', kernel_initializer=w_init, name='n64s1/c')
#
#         n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n64s2/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')
#
#         n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n128s1/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')
#
#         n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n128s2/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')
#
#         n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n256s1/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')
#
#         n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n256s2/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')
#
#         n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n512s1/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')
#
#         n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', kernel_initializer=w_init, b_init=b_init, name='n512s2/c')
#         n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')
#
#         n = FlattenLayer(n, name='f')
#         n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
#         n = DenseLayer(n, n_units=1, name='out')
#
#         logits = n.outputs
#         n.outputs = tf.nn.sigmoid(n.outputs)
#
#         return n, logits


# def Vgg19_simple_api(rgb, reuse):
#     """
#     Build the VGG 19 Model
#
#     Parameters
#     -----------
#     rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
#     """
#     VGG_MEAN = [103.939, 116.779, 123.68]
#     with tf.variable_scope("VGG19", reuse=reuse) as vs:
#         start_time = time.time()
#         print("build model started")
#         rgb_scaled = rgb * 255.0
#         # Convert RGB to BGR
#         if tf.__version__ <= '0.11':
#             red, green, blue = tf.split(3, 3, rgb_scaled)
#         else:  # TF 1.0
#             # print(rgb_scaled)
#             red, green, blue = tf.split(rgb_scaled, 3, 3)
#         assert red.get_shape().as_list()[1:] == [224, 224, 1]
#         assert green.get_shape().as_list()[1:] == [224, 224, 1]
#         assert blue.get_shape().as_list()[1:] == [224, 224, 1]
#         if tf.__version__ <= '0.11':
#             bgr = tf.concat(3, [
#                 blue - VGG_MEAN[0],
#                 green - VGG_MEAN[1],
#                 red - VGG_MEAN[2],
#             ])
#         else:
#             bgr = tf.concat(
#                 [
#                     blue - VGG_MEAN[0],
#                     green - VGG_MEAN[1],
#                     red - VGG_MEAN[2],
#                 ], axis=3)
#         assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
#         """ input layer """
#         net_in = InputLayer(bgr, name='input')
#         """ conv1 """
#         network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
#         network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
#         """ conv2 """
#         network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
#         network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
#         """ conv3 """
#         network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
#         network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
#         network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
#         network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
#         """ conv4 """
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
#         conv = network
#         """ conv5 """
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
#         network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
#         network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
#         """ fc 6~8 """
#         network = FlattenLayer(network, name='flatten')
#         network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
#         network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
#         network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
#         print("build model finished: %fs" % (time.time() - start_time))
#         return network, conv


# def vgg16_cnn_emb(t_image, reuse=False):
#     """ t_image = 244x244 [0~255] """
#     with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
#         tl.layers.set_name_reuse(reuse)
#
#         mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
#         net_in = InputLayer(t_image - mean, name='vgg_input_im')
#         """ conv1 """
#         network = tl.layers.Conv2dLayer(net_in,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool1')
#         """ conv2 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool2')
#         """ conv3 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool3')
#         """ conv4 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_3')
#
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool4')
#         conv4 = network
#
#         """ conv5 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool5')
#
#         network = FlattenLayer(network, name='vgg_flatten')
#
#         # # network = DropoutLayer(network, keep=0.6, is_fix=True, is_train=is_train, name='vgg_out/drop1')
#         # new_network = tl.layers.DenseLayer(network, n_units=4096,
#         #                     act = tf.nn.relu,
#         #                     name = 'vgg_out/dense')
#         #
#         # # new_network = DropoutLayer(new_network, keep=0.8, is_fix=True, is_train=is_train, name='vgg_out/drop2')
#         # new_network = DenseLayer(new_network, z_dim, #num_lstm_units,
#         #             b_init=None, name='vgg_out/out')
#         return conv4, network
