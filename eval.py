#-*- coding: utf-8 -*-
import os
import tensorflow as tf
from meta import Meta
from evaluator import Evaluator
import sys
import argparse




def _eval(path_to_checkpoint_dir, path_to_eval_tfrecords_file, num_eval_examples, path_to_eval_log_dir):
    evaluator = Evaluator(path_to_eval_log_dir)

    checkpoint_paths = tf.train.get_checkpoint_state(path_to_checkpoint_dir).all_model_checkpoint_paths
    for global_step, path_to_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
        try:
            global_step_val = int(global_step)
        except ValueError:
            continue

        accuracy = evaluator.evaluate(path_to_checkpoint, path_to_eval_tfrecords_file, num_eval_examples,
                                      global_step_val)
        print 'Evaluate %s on %s, accuracy = %f' % (path_to_checkpoint, path_to_eval_tfrecords_file, accuracy)


def main_eval(_):

    parser = argparse.ArgumentParser(description="Evaluation Routine for SVHNClassifier")
    parser.add_argument("--data_dir", required=True, help="Directory to read TFRecords files")
    parser.add_argument("--path_to_checkpoint_dir", required=True, help="Directory to read checkpoint files")
    parser.add_argument("--eval_logdir", required=True, help="Directory to write evaluation logs")
    parser.add_argument("--path_to_train_tfrecords_file", required=True, help="Tfrecords file in train directory")
    parser.add_argument("--path_to_val_tfrecords_file", required=True, help="Tfrecords file in val directory")
    parser.add_argument("--path_to_test_tfrecords_file", required=True, help="Tfrecords file in test directory")
    parser.add_argument("--path_to_tfrecords_meta_file", required=True, help="Meta file in directory")
    parser.add_argument("--path_to_train_eval_log_dir", required=True, help="Training and evaluating log directory")
    parser.add_argument("--path_to_val_eval_log_dir", required=True, help="Validating and evaluating log directory")
    parser.add_argument("--path_to_test_eval_log_dir", required=True, help="Testing and evaluating log directory")
    args = parser.parse_args()

    meta = Meta()
    meta.load(args.path_to_tfrecords_meta_file)

    _eval(args.path_to_checkpoint_dir, args.path_to_train_tfrecords_file, meta.num_train_examples, args.path_to_train_eval_log_dir)
    _eval(args.path_to_checkpoint_dir, args.path_to_val_tfrecords_file, meta.num_val_examples, args.path_to_val_eval_log_dir)
    _eval(args.path_to_checkpoint_dir, args.path_to_test_tfrecords_file, meta.num_test_examples, args.path_to_test_eval_log_dir)


if __name__ == '__main__':

    if len(sys.argv) == 1:

        sys.argv.extend(["--data_dir",  "./data",                                           "--path_to_checkpoint_dir", "./logs/train",
                         "--eval_logdir", "./logs/eval",                                    "--path_to_train_tfrecords_file", "./data/train.tfrecords",
                         "--path_to_val_tfrecords_file", "./data/val.tfrecords",            "--path_to_test_tfrecords_file", "./data/test.tfrecords",
                         "--path_to_tfrecords_meta_file", "./data/meta.json",               "--path_to_train_eval_log_dir", "./logs/eval/train",
                         "--path_to_val_eval_log_dir", "./logs/eval/val",                   "--path_to_test_eval_log_dir", "./logs/eval/test"])

    tf.app.run(main=main_eval)
