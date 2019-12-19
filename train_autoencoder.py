from argparse import ArgumentParser
from dataloader import DataLoader
from autoencoder import Autoencoder
import os
import glob
import numpy as np
from datetime import datetime
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL']= "2"

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
parser.add_argument('--image_dir', type=str, help='Path to high resolution image directory.')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
parser.add_argument('--crop_size', default=256, type=int, help='Low resolution input size.')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int, help='The number of iterations to save the tensorboard summaries and models.')
parser.add_argument('--model_dir', default="./models", type=str, help='Model directory if different from ./models/autoencoder.h5.')
parser.add_argument('--logdir', default="./logs", type=str, help='Tensorboard logdir.')
parser.add_argument('--retrain_model', default=False, type=bool, help='True for retraining current model in models/autoencoder.h5.')
parser.add_argument('--save_model', default=True, type=bool, help='Save model during iterations.')




@tf.function
def train_step(model, x, y):
  """
  Single step of generator pre-training.
  Args:
    model: A model object with a tf keras compiled generator.
    x: The low resolution image tensor.
    y: The high resolution image tensor.
  """
  with tf.GradientTape() as tape:
    fake_hr = model.autoencoder(x)
    content_loss = model.content_loss(y, fake_hr)
    mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
    total_loss = content_loss + mse_loss

  grads = tape.gradient(total_loss, model.autoencoder.trainable_variables)
  model.optimizer.apply_gradients(zip(grads, model.autoencoder.trainable_variables))

  return content_loss, mse_loss

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
      content_loss, mse_loss = train_step(model, x, y)
      # Log tensorboard summaries if log iteration is reached.
      if model.iterations % log_iter == 0:
        tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
        tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
        tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.iterations)
        tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
        tf.summary.image('Generated', tf.cast(255 * (model.autoencoder.predict(x) + 1.0) / 2.0, tf.uint8), step=model.iterations)
        writer.flush()
      model.iterations += 1

def get_path(path):
  return os.path.expanduser(os.path.expandvars(path))

def main():
  # Parse the CLI arguments.
  args = parser.parse_args()

  # create directory for saving trained models.
  if not os.path.exists('models'):
    os.makedirs('models/checkpoints')

  # Create the tensorflow dataset.
  # image_dir = get_path(args.image_dir)
  ds = DataLoader(args.image_dir, args.crop_size).dataset(args.batch_size)

  # Define the directory for saving the SRGAN training tensorbaord summary.
  logdir = get_path(args.logdir)
  traindirs = glob.glob(os.path.join(logdir,"train_*"))
  if traindirs:
    train_num = int(max([x.split('_')[-1] for x in traindirs]))
    train_num += 1
  else:
    train_num = 1
  traindir = os.path.join(logdir, f"train_{train_num}")
  train_summary_writer = tf.summary.create_file_writer(traindir)

  tf.summary.trace_on(graph=True, profiler=False)
  model = Autoencoder(args)
  # with train_summary_writer.as_default():
  #   tf.summary.trace_export("Autoencoder", step=0)
  #   train_summary_writer.flush()
  
  # Run training.
  for epoch in range(args.epochs):
    print("====== Beginning epoch {} ======".format(epoch))
    train(model, ds, args, train_summary_writer)
    with train_summary_writer.as_default():
      tf.summary.trace_export("Autoencoder", step=0)
      train_summary_writer.flush()
    if args.save_model:
      timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      model.autoencoder.save(f"models/checkpoints/autoencoder_ckpt_{epoch}_{timestamp}.h5")
    print("====== Finished epoch {} ======".format(epoch))

  # Save final models
  if args.save_model:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.autoencoder.save("models/autoencoder.h5")
    model.autoencoder.save(f"models/checkpoints/autoencoder_{timestamp}.h5")

if __name__ == '__main__':
  main()
