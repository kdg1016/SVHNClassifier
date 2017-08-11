#-*- coding: utf-8 -*-

import os
from datetime import datetime
import time
import tensorflow as tf
from meta import Meta
from donkey import Donkey
from model import Model
from evaluator import Evaluator

import sys
import argparse





def _train(path_to_train_tfrecords_file, num_train_examples, path_to_val_tfrecords_file, num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file, training_options):
    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = Donkey.build_batch(path_to_train_tfrecords_file,
                                                                     num_examples=num_train_examples,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)
        length_logtis, digits_logits = Model.inference(image_batch, drop_rate=0.2)
        loss = Model.loss(length_logtis, digits_logits, length_batch, digits_batch)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.image('image', image_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print 'Model restored from file: %s' % path_to_restore_checkpoint_file

            print 'Start training'
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, loss, summary, global_step, learning_rate])
                duration += time.time() - start_time

                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print '=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_val, loss_val, examples_per_sec)

                if global_step_val % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_val, global_step=global_step_val)

                print '=> Evaluating on validation dataset...'
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, path_to_val_tfrecords_file,
                                              num_val_examples,
                                              global_step_val)
                print '==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy)

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print '=> Model saved to file: %s' % path_to_checkpoint_file
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                print '=> patience = %d' % patience
                if patience == 0:
                    break

            coord.request_stop()
            coord.join(threads)
            print 'Finished'


def main_train(_):

    parser = argparse.ArgumentParser(description="Training Routine for SVHNClassifier")
    parser.add_argument("--data_dir", required=True, help="Path to SVHN (format 1) folders")
    parser.add_argument("--path_to_train_log_dir", required=True, help="Directory to write training logs")
    parser.add_argument("--path_to_restore_checkpoint_file", required=False, help="Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100")
    parser.add_argument("--path_to_train_tfrecords_file", required=True, help="Tfrecords file in train directory")
    parser.add_argument("--path_to_val_tfrecords_file", required=True, help="Tfrecords file in val directory")
    parser.add_argument("--path_to_tfrecords_meta_file", required=True, help="Meta file in directory")

    parser.add_argument("--batch_size", type=int, required=True, help="Default 32")
    parser.add_argument("--learning_rate", type=float, required=True, help="Default 1e-2")
    parser.add_argument("--patience", type=int, required=True, help="Default 100, set -1 to train infinitely")
    parser.add_argument("--decay_steps", type=int, required=True, help="Default 10000")
    parser.add_argument("--decay_rate", type=float, required=True, help="Default 0.9")
    args = parser.parse_args()



    training_options = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'decay_steps': args.decay_steps,
        'decay_rate': args.decay_rate
    }

    meta = Meta()
    meta.load(args.path_to_tfrecords_meta_file)

    _train(args.path_to_train_tfrecords_file, meta.num_train_examples,
           args.path_to_val_tfrecords_file, meta.num_val_examples,
           args.path_to_train_log_dir, args.path_to_restore_checkpoint_file,
           training_options)


if __name__ == '__main__':

    if len(sys.argv) == 1:

        # 재 학습시 --path_to_train_log_dir => ./logs/train2
        #         --path_to_restore_checkpoint_file => ./logs/train/latest.ckpt 로 변경하여 학습 진행

        sys.argv.extend(["--data_dir",  "./data",                                           "--path_to_train_log_dir", "./logs/train",
                         "--path_to_restore_checkpoint_file", None,                         "--path_to_train_tfrecords_file", "./data/train.tfrecords",
                         "--path_to_val_tfrecords_file", "./data/val.tfrecords",            "--path_to_tfrecords_meta_file", "./data/meta.json",
                         "--batch_size", "32",                                              "--learning_rate", "1e-2",
                         "--patience", "100",                                               "--decay_steps", "10000",
                         "--decay_rate", "0.9"])

    tf.app.run(main=main_train)
