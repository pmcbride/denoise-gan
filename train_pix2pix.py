from argparse import ArgumentParser
from dataloader import DataLoader
from pix2pix import Pix2Pix
import os
import glob
import shutil
from time import time
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

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


@tf.function
def train_step(model, x, y):
  """
  Single step of generator pre-training.
  Args:
    model: A model object with a tf keras compiled generator.
    x: The low resolution image tensor.
    y: The high resolution image tensor.
  """
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    # Generate denoised image
    gen_output = model.generator(x, training=True)

    # Train discriminators (original images = real ; generated = Fake)
    disc_real_output = model.discriminator([x, y], training=True)
    disc_generated_output = model.discriminator([x, gen_output], training=True)

    # Generator Loss
    gen_total_loss, gen_gan_loss, gen_l1_loss, gen_l2_loss, content_loss, var_loss, identity_loss = model.generator_loss(disc_generated_output, gen_output, y)
    # content_loss = model.content_loss(y, gen_output)
    # adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, disc_generated_output)
    # mse_loss = tf.keras.losses.MeanSquaredError()(y, gen_output)
    # perceptual_loss = content_loss + adv_loss + mse_loss

    # Discriminator Loss
    disc_loss = model.discriminator_loss(disc_real_output, disc_generated_output)
    # valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, disc_real_output)
    # fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, disc_generated_output)
    # disc_loss = tf.add(valid_loss, fake_loss)

  # Apply Gradients
  gen_grads = gen_tape.gradient(gen_total_loss, model.generator.trainable_variables)
  disc_grads = disc_tape.gradient(disc_loss, model.discriminator.trainable_variables)

  # Backprop
  model.gen_optimizer.apply_gradients(zip(gen_grads, model.generator.trainable_variables))
  model.disc_optimizer.apply_gradients(zip(disc_grads, model.discriminator.trainable_variables))

  return gen_total_loss, gen_gan_loss, gen_l1_loss, gen_l2_loss, content_loss, disc_loss, var_loss, identity_loss

def train(model, dataset, args, writer):
  """
  Function that defines a single training step for the SR-GAN.
  Args:
    model: An object that contains tf keras compiled generator and
         discriminator models.
    dataset: A tf data object that contains low and high res images.
    log_iter: Number of iterations after which to add logs in
          tensorboard.
    writer: Summary writer
  """
  log_iter = args.save_iter

  with writer.as_default():
    # Iterate over dataset
    for x, y in dataset:
      gen_total_loss, gen_gan_loss, gen_l1_loss, gen_l2_loss, content_loss, disc_loss, var_loss, identity_loss = train_step(model, x, y)
      # train_step(model, x, y, writer)
      model.iterations += 1
        # Log tensorboard summaries if log iteration is reached.
      if model.iterations % log_iter == 0:
  #     with writer.as_default():
        tf.summary.scalar('Generator Losses/gen_total_loss', gen_total_loss, step=model.epochs+1)
        tf.summary.scalar('Generator Losses/gen_gan_loss', gen_gan_loss, step=model.epochs+1)
        tf.summary.scalar('Generator Losses/gen_l1_loss', gen_l1_loss, step=model.epochs+1)
        tf.summary.scalar('Generator Losses/gen_l2_loss', gen_l2_loss, step=model.epochs+1)
        tf.summary.scalar('Generator Losses/content_loss', content_loss, step=model.epochs+1)
        tf.summary.scalar('Generator Losses/total_variation', var_loss, step=model.epochs+1)
        tf.summary.scalar('Generator Losses/identity_loss', identity_loss, step=model.epochs+1)
        tf.summary.scalar('Discriminator Losses/disc_loss', disc_loss, step=model.epochs+1)
        tf.summary.image('Images/Low IQ', tf.cast(255 * (x + 1) / 2, tf.uint8), step=model.epochs+1)
        tf.summary.image('Images/High IQ', tf.cast(255 * (y + 1) / 2, tf.uint8), step=model.epochs)
        tf.summary.image('Images/Generated', tf.cast(255 * (model.generator(x, training=False) + 1) / 2, tf.uint8), step=model.epochs+1)
        # tf.summary.image('Disc Generated', tf.cast(255 * (model.discriminator([x, y], training=False) + 1.0) / 2.0, tf.uint8), step=model.iterations)
        writer.flush()

def get_path(path):
  return os.path.realpath(os.path.expanduser(os.path.expandvars(path)))

def main(args):
  # Parse the CLI arguments.
  # args = parser.parse_args()

  # create directory for saving trained models.
  os.makedirs('models/checkpoints', exist_ok=True)
  os.makedirs('models/backups', exist_ok=True)
  os.makedirs('logs', exist_ok=True)

  # Calculate steps per epoch
  image_dir = get_path(args.image_dir)
  num_images = len(os.listdir(image_dir))
  batch_size = args.batch_size
  steps_per_epoch = num_images // batch_size
  print(f"Steps per epoch: {steps_per_epoch}")
  if args.save_iter > steps_per_epoch:
    args.save_iter = steps_per_epoch

  # Create the tensorflow dataset.
  ds = DataLoader(args).dataset()

  # Define the directory for saving the SRGAN training tensorbaord summary.
  logdir = get_path(args.logdir)
  traindirs = glob.glob(os.path.join(logdir,"train_*"))
  if traindirs:
    train_num = max([int(x.split('_')[-1]) for x in traindirs])
    train_num += 1
  else:
    train_num = 1
  traindir = os.path.join(logdir, f"train_{train_num}")
  train_summary_writer = tf.summary.create_file_writer(traindir)

  # if args.retrain == False:
  #   print("Retrain model is False")
  #   if os.path.exists("./models/pix2pix.h5") & os.path.exists("./models/discriminator.h5"):
  #     args.retrain = bool(int(input("Retrain model [0/1]? ")))
  # elif args.retrain == True:
  #   print("Retrain model is True")
  # else:
  #   print("Retrain model is not boolean")

  model = Pix2Pix(args)

  # Create Checkpoint
  checkpoint_dir = get_path('models/checkpoints')
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(gen_optimizer=model.gen_optimizer,
                                   disc_optimizer=model.disc_optimizer,
                                   generator=model.generator,
                                   discriminator=model.discriminator)
  ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
  if bool(args.retrain) == True:
    checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

  # Collect Timestamp for training
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # Run training.
  for epoch in range(args.epochs):
    # print("====== Beginning epoch: {} ======".format(epoch))
    train_begin = time()
    train(model, ds, args, train_summary_writer)
    train_end = time()
    train_time = train_end - train_begin
    if args.ckpt:
      if epoch % 5 == 0:
        ckpt_manager.save()
        # checkpoint.save(file_prefix=checkpoint_prefix)
        # model.generator.save(f"models/backups/pix2pix_ckpt_{timestamp}_{epoch}.h5")
        # model.discriminator.save(f"models/backups/pix_discriminator_ckpt_{timestamp}_{epoch}.h5")
    end = time()
    save_time = end - train_end
    total_time = end - train_begin
    model.epochs += 1
    print(f"====== Finished epoch: {epoch+1}, iterations: {model.iterations}, train time: {train_time:0.2f}, total time: {total_time:0.2f} ======")

  # Save final models
  if args.save_model:
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_manager.save()
    model.generator.save("models/pix2pix.h5")
    model.discriminator.save("models/discriminator_p2p.h5")
    model.generator.save(f"models/backups/pix2pix_{timestamp}.h5")
    model.discriminator.save(f"models/backups/discriminator_p2p_{timestamp}.h5")

if __name__ == '__main__':
  params = dict(
    image_dir = get_path("~/Data/DIV2K/DIV2K_train_HR"),
    batch_size = 1,
    epochs = 1,
    crop_size = 256,
    lr = 1e-3,
    save_iter = 200,
    model_dir = get_path("./models"),
    logdir = get_path("./logs"),
    retrain = 0,
    save_model = 1,
    ckpt = 1,
    fp16 = 0
  )

  parser = ArgumentParser()

  for key, value in params.items():
    flag = str("--" + key)
    # print(f"parser.add_argument({flag}, default={value}, type={type(value)})")
    parser.add_argument(flag, default=value, type=type(value))

  args = parser.parse_args()
  args.retrain = bool(args.retrain)
  args.save_model = bool(args.save_model)
  args.ckpt = bool(args.ckpt)
  args.fp16 = bool(args.fp16)

  if args.fp16:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

  print("COMPUTATION PARAMETERS")
  print('Compute dtype: %s' % policy.compute_dtype)
  print('Variable dtype: %s' % policy.variable_dtype)

  for k, v in vars(args).items():
    v = type(v)(v)
    print(f"  {k}: {v}, type: {str(type(v))}")

  main(args)
