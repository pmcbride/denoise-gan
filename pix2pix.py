import tensorflow as tf

# CREATE GENERATOR
class Pix2Pix(object):
  """ Denoising Pix2Pix """

  def __init__(self, args):
    """
    Initialized the Pix2Pix class.
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
    
    # Define Loss objects
    self.gen_loss_object = tf.keras.losses.MeanSquaredError()
    self.disc_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define Optimzer
    self.gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # We use a pre-trained VGG19 model to extract image features from the high resolution
    # and the generated high resolution images and minimize the mse between them
    self.vgg = self.build_vgg()
    self.vgg.trainable = False

    # Number of filters in the first layer of G and D
    self.gf = 32  # Realtime Image Enhancement GAN Galteri et al.
    self.df = 32

    # Build and compile the GAN (generator and discriminator)
    self.generator, self.discriminator = self.build_gan()

  @tf.function
  def content_loss(self, target, gen_output):
    gen_output = tf.keras.applications.vgg19.preprocess_input(((gen_output + 1.0) * 255) / 2.0)
    target = tf.keras.applications.vgg19.preprocess_input(((target + 1.0) * 255) / 2.0)
    gen_features = self.vgg(gen_output) / 12.75
    target_features = self.vgg(target) / 12.75
    return self.gen_loss_object(target_features, gen_features)

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
  
  # @tf.function
  # def loss_object(self, logits, disc_output):
  #   return tf.keras.losses.BinaryCrossentropy(from_logits=True)(logits, disc_output)

  # Pix2Pix Generator Loss
  def generator_loss(self, disc_generated_output, gen_output, target):
    gan_loss = 1e-3 * self.disc_loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # Total Variation error
    var_loss = 1e-5 * tf.reduce_mean(tf.image.total_variation(target - gen_output))
    
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Mean Squared Error
    l2_loss = tf.keras.losses.MeanSquaredError()(target, gen_output)
    
    # VGG Content Loss
    cont_loss = self.content_loss(gen_output, target)

    # Identity Loss
    identity_loss = tf.reduce_mean(tf.abs(self.generator(target, training=True) - target))

    total_gen_loss = gan_loss + l2_loss + cont_loss + var_loss + l1_loss + identity_loss

    return total_gen_loss, gan_loss, l1_loss, l2_loss, cont_loss, var_loss, identity_loss

  def discriminator_loss(self, disc_real_output, disc_generated_output):
    real_loss = self.disc_loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = self.disc_loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

  # Build Pix2Pix Generator
  def build_gan(self, name='Pix2Pix'):
    OUTPUT_CHANNELS = 3

    # Downsampling layer
    def downsample(filters, size, apply_batchnorm=True):
      initializer = tf.random_normal_initializer(0., 0.02)

      result = tf.keras.Sequential()
      result.add(
          tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

      if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

      result.add(tf.keras.layers.LeakyReLU())

      return result

    def upsample(filters, size, apply_dropout=False):
      initializer = tf.random_normal_initializer(0., 0.02)

      result = tf.keras.Sequential()
      result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

      result.add(tf.keras.layers.BatchNormalization())

      if apply_dropout:
          result.add(tf.keras.layers.Dropout(0.5))

      result.add(tf.keras.layers.ReLU())

      return result

    def Generator():
      inputs = tf.keras.layers.Input(shape=self.lr_shape)

      down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
      ]

      up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
      ]

      initializer = tf.random_normal_initializer(0., 0.02)
      last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation='tanh') # (bs, 256, 256, 3)

      x = inputs

      # Downsampling through the model
      skips = []
      for down in down_stack:
        x = down(x)
        skips.append(x)

      skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections
      for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

      x = last(x)

      return tf.keras.Model(inputs=inputs, outputs=x)
    
    def Discriminator():
      initializer = tf.random_normal_initializer(0., 0.02)

      inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
      tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

      x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

      down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
      down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
      down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

      zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
      conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

      batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

      leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

      zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

      last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

      return tf.keras.Model(inputs=[inp, tar], outputs=last)

    # Build generator and discriminator  
    generator = Generator()
    discriminator = Discriminator()

    return generator, discriminator