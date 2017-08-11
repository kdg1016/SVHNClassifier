#-*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import random
from PIL import Image
import tensorflow as tf
from meta import Meta

import csv
import sys
import argparse






class ExampleReader(object):
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._example_pointer = 0

    
    @staticmethod
    def _get_csv_writer(filename, rows, delimiter):
        with open(filename, 'w') as csvfile:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                try:
                    writer.writerow(row)
                except Exception as detail:
                    print type(detail)
                    print detail

    @staticmethod
    def _get_csv_reader(filename, delimiter):
        reader = []
        if not os.path.isfile(filename):
            csvfile = open(filename, "w")
        else:
            csvfile = open(filename, "rb")
            reader = csv.DictReader(csvfile, delimiter=delimiter)
        return list(reader)





    @staticmethod
    def _get_attrs(digit_struct_mat_file,index):
        """
        Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
        """
        attrs = {}
        f = digit_struct_mat_file


        item = f['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = f[item][key]
            values = [f[attr.value[i].item()].value[0][0] for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
            '''
            values = []
            if len(attr) > 1:
                for i in range(len(attr)):
                    values.append(f[attr.value[i].item()].value[0][0])
            else:
                values = [attr.value[0][0]]
            '''

            attrs[key] = values
        return attrs

    csvList = {}

    @staticmethod
    def _get_attrs_for_csv(digit_struct_csv_file, index, csvList):
        """
        Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
        """
        attrs = {}
        values = []


        attr_length = int(csvList[index]['length'])


        # 사전 key 값 로드
        for key in ['height', 'label', 'left', 'top', 'width']:

            attr = key # 해당 되는 속성 지정

            if (attr=='height'):
                if attr_length  > 1:
                    for i in range(attr_length):
                        attr_key = "%s%s%d%s" % (attr, '(', i+1, ')')
                        values.append(csvList[index][attr_key])
                else:
                    values = values

            elif (attr=='label'):
                if attr_length  > 1:
                    for i in range(attr_length):
                        attr_key = "%s%s%d%s" % (attr, '(', i+1, ')')
                        values.append(csvList[index][attr_key])
                else:
                    values = values

            elif (attr=='left'):
                if attr_length  > 1:
                    for i in range(attr_length):
                        attr_key = "%s%s%d%s" % (attr, '(', i+1, ')')
                        values.append(csvList[index][attr_key])
                else:
                    values = values

            elif (attr=='top'):
                if attr_length  > 1:
                    for i in range(attr_length):
                        attr_key = "%s%s%d%s" % (attr, '(', i+1, ')')
                        values.append(csvList[index][attr_key])
                else:
                    values = values

            else :
                if attr_length  > 1:
                    for i in range(attr_length):
                        attr_key = "%s%s%d%s" % (attr, '(', i+1, ')')
                        values.append(csvList[index][attr_key])
                else:
                    values = values

            attrs[key] = values
            values = []

        return attrs




    @staticmethod
    def _preprocess(image, bbox_left, bbox_top, bbox_width, bbox_height):
        cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                    int(round(bbox_top - 0.15 * bbox_height)),
                                                                    int(round(bbox_width * 1.3)),
                                                                    int(round(bbox_height * 1.3)))
        image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
        image = image.resize([64, 64])
        return image

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self, digit_struct_mat_file):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]
        index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
        self._example_pointer += 1

        attrs = ExampleReader._get_attrs(digit_struct_mat_file, index)
        label_of_digits = attrs['label']
        length = len(label_of_digits)
        if length > 5:
            # skip this example
            return self.read_and_convert(digit_struct_mat_file)

        digits = [10, 10, 10, 10, 10]   # digit 10 represents no digit
        for idx, label_of_digit in enumerate(label_of_digits):
            digits[idx] = int(label_of_digit if label_of_digit != 10 else 0)    # label 10 is essentially digit zero


        attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
        min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                    min(attrs_top),
                                                    max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                        center_y - max_side / 2.0,
                                                        max_side,
                                                        max_side)
        image = np.array(ExampleReader._preprocess(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()
        

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'length': ExampleReader._int64_feature(length),
            'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
        }))
        return example

    ##
    def read_and_convert_for_csv(self, digit_struct_csv_file, csvList):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]
        index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
        self._example_pointer += 1

        attrs = ExampleReader._get_attrs_for_csv(digit_struct_csv_file, index, csvList)
        label_of_digits = attrs['label']
        length = len(label_of_digits)
        if length > 5:
            # skip this example
            return self.read_and_convert_for_csv(digit_struct_csv_file, csvList)

        digits = [10, 10, 10, 10, 10]   # digit 10 represents no digit
        for idx, label_of_digit in enumerate(label_of_digits):
            digits[idx] = int(label_of_digit if label_of_digit != 10 else 0)    # label 10 is essentially digit zero


        attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
        min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                    min(attrs_top),
                                                    max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                        center_y - max_side / 2.0,
                                                        max_side,
                                                        max_side)
        image = np.array(ExampleReader._preprocess(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()



        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'length': ExampleReader._int64_feature(length),
            'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
        }))
        return example


def convert_to_tfrecords(path_to_dataset_dir_and_digit_struct_mat_file_tuples,
                         path_to_tfrecords_files, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))


    for path_to_dataset_dir, path_to_digit_struct_mat_file in path_to_dataset_dir_and_digit_struct_mat_file_tuples:
        path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.png'))
        total_files = len(path_to_image_files)
        print '%d files found in %s' % (total_files, path_to_dataset_dir)


        with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
            example_reader = ExampleReader(path_to_image_files)
            for index, path_to_image_file in enumerate(path_to_image_files):
                print '(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file)

                # mat 파일의 각종 속성정보(이미지의  각각에 해당하는 좌표정보)를 불러와 example 형태로 다음 내용 저장 : "digits", "image", length"
                example = example_reader.read_and_convert(digit_struct_mat_file)
                if example is None:
                    break

                idx = choose_writer_callback(path_to_tfrecords_files)
                writers[idx].write(example.SerializeToString())
                num_examples[idx] += 1

    ############################################################
    # 폴더 위치가 ./extra 이면, 해당 루틴 실행(user_train)
    if (path_to_dataset_dir == './data/extra'):
        # Step 3
        # 이미지 폴더 경로(./data/user_train)에서 이미지 파일, 갯수 및 csv 파일 로딩
        path_to_user_train_dir = './data/user_train'
        path_to_image_files = tf.gfile.Glob(os.path.join(path_to_user_train_dir, '*.png'))
        total_files = len(path_to_image_files)
        print '%d files found in %s' % (total_files, path_to_user_train_dir)

        with open('./data/user_train/user_train.csv', 'r') as csvfile:

            # csv 파일에서 각각의 line을 csvList 형태로 누적 로드(이미지 관련 info)
            ############################################################
            csvList = {}
            line_tmp = []

            csv_reader = csv.DictReader(csvfile)

            for line in csv_reader:
                line_tmp.append(line)
                csvList = line_tmp
            ############################################################

            example_reader = ExampleReader(path_to_image_files)
            for index, path_to_image_file in enumerate(path_to_image_files):
                print '(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file)

                # mat 파일의 각종 속성정보(이미지의  각각에 해당하는 좌표정보)를 불러와 example 형태로 다음 내용 저장 : "digits", "image", length"
                example = example_reader.read_and_convert_for_csv(csvfile, csvList)
                if example is None:
                    break

                idx = choose_writer_callback(path_to_tfrecords_files)
                writers[idx].write(example.SerializeToString())
                num_examples[idx] += 1


    else :
        pass
            ############################################################

    for writer in writers:
        writer.close() # write 후 닫기

    return num_examples


def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file):
    print 'Saving meta file to %s...' % path_to_tfrecords_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_tfrecords_meta_file)


def main_convert_to_tfrecords(_):

    # 기존 tfrecords 파일 제거(디버깅시 사용)
    # if os.path.isfile('./data/train.tfrecords'):
    #     os.remove('./data/train.tfrecords')
    # if os.path.isfile('./data/val.tfrecords'):
    #     os.remove('./data/val.tfrecords')
    # if os.path.isfile('./data/test.tfrecords'):
    #     os.remove('./data/test.tfrecords')


    parser = argparse.ArgumentParser(description="Convert to tfrecords foramt for SVHNClassifier")
    parser.add_argument("--data_dir", required=True, help="Directory to SVHN (format 1) folders and write the converted files")
    parser.add_argument("--path_to_train_dir", required=True, help="Directory for train data")
    parser.add_argument("--path_to_test_dir", required=True, help="Directory for test data")
    parser.add_argument("--path_to_extra_dir", required=True, help="Directory for extra data")
    parser.add_argument("--path_to_user_train_dir", required=True, help="Directory for user train data")
    parser.add_argument("--path_to_train_digit_struct_mat_file", required=True, help="Digit struct mat file in train directory")
    parser.add_argument("--path_to_test_digit_struct_mat_file", required=True, help="Digit struct mat file in test directory")
    parser.add_argument("--path_to_extra_digit_struct_mat_file", required=True, help="Digit struct mat file in extra directory")
    parser.add_argument("--path_to_train_tfrecords_file", required=True, help="Tfrecords file in train directory")
    parser.add_argument("--path_to_val_tfrecords_file", required=True, help="Tfrecords file in val directory")
    parser.add_argument("--path_to_test_tfrecords_file", required=True, help="Tfrecords file in test directory")
    parser.add_argument("--path_to_tfrecords_meta_file", required=True, help="Tfrecords meta file")
    parser.add_argument("--in_csv_file", required=True, help="User train csv file in user_train directory")
    args = parser.parse_args()



    for path_to_file in [args.path_to_train_tfrecords_file, args.path_to_val_tfrecords_file, args.path_to_test_tfrecords_file]:
        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

    print 'Processing training and validation data...'
    [num_train_examples, num_val_examples] = convert_to_tfrecords([(args.path_to_train_dir, args.path_to_train_digit_struct_mat_file),
                                                                   (args.path_to_extra_dir, args.path_to_extra_digit_struct_mat_file)],
                                                                  [args.path_to_train_tfrecords_file, args.path_to_val_tfrecords_file],
                                                                  lambda paths: 0 if random.random() > 0.1 else 1)

    print 'Processing test data...'
    [num_test_examples] = convert_to_tfrecords([(args.path_to_test_dir, args.path_to_test_digit_struct_mat_file)],
                                               [args.path_to_test_tfrecords_file],
                                               lambda paths: 0)

    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               args.path_to_tfrecords_meta_file)

    print 'Done'


if __name__ == '__main__':

    if len(sys.argv) == 1:

        sys.argv.extend(["--data_dir",  "./data",                                               "--path_to_train_dir", "./data/train",
                         "--path_to_test_dir", "./data/test",                                   "--path_to_extra_dir", "./data/extra",
                         "--path_to_user_train_dir", "./data/user_train",                       "--path_to_train_digit_struct_mat_file", "./data/train/digitStruct.mat",
                         "--path_to_test_digit_struct_mat_file", "./data/test/digitStruct.mat", "--path_to_extra_digit_struct_mat_file", "./data/extra/digitStruct.mat",
                         "--path_to_train_tfrecords_file", "./data/train.tfrecords",            "--path_to_val_tfrecords_file", "./data/val.tfrecords",
                         "--path_to_test_tfrecords_file", "./data/test.tfrecords",              "--path_to_tfrecords_meta_file", "./data/meta.json",
                         "--in_csv_file", "./data/user_train/user_train.csv"])

    tf.app.run(main=main_convert_to_tfrecords)
