import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import capsules

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './tmp/capsules_train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of steps to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often to log results to the console.""")

def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        with tf.device("/cpu:0"):
            # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            image, label = mnist.train.next_batch(100)
            images = tf.convert_to_tensor(image, tf.float32)
            labels = tf.convert_to_tensor(label, tf.float32)
            images1 = tf.unstack(images, axis=0)
            labels1 = tf.unstack(labels, axis=0)

            batch_im, batch_la = tf.train.batch([images1, labels1], 10, enqueue_many=True)
            print(np.shape(image))
            print(batch_im)
            print(batch_la)
            # print(batch)

            # images = mnist.train.images
            # labels = np.asarray(mnist.train.labels, dtype=np.int32)

        capsule_logits, reconstruction_logits = capsules.inference(batch_im)

        loss = capsules.total_loss(capsule_logits, reconstruction_logits, batch_im, batch_la)

        predictions_, labels_, accuracy_ = capsules.accuracy(capsule_logits, batch_la)

        train_op = capsules.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs([loss, predictions_, labels_, accuracy_])  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    # current_time = time.time()
                    # duration = current_time - self._start_time
                    # self._start_time = current_time

                    loss_value = run_values.results[0]
                    predictions_ = run_values.results[1]
                    labels_ = run_values.results[2]
                    accuracy_ = run_values.results[3]
                    # examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    # sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('Step %d \nPredictions: %s \nLabels: %s \nAccuracy: %.3f \nLoss = %.2f')
                    print (format_str % (self._step, predictions_, labels_, accuracy_, loss_value))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    train()


if __name__ == "__main__":
    tf.app.run()
