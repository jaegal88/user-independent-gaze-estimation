import numpy as np
import cv2
import tensorflow as tf
# import parameters
import os
# import matplotlib.pyplot as plt
# import glob
# import sys

# BATCH_SIZE = 1
# data_path = 'D:/Users/Documents/Python/ExampleCode/sequence_data_set_I.tfrecords'
Size_X = 320
Size_Y = 240

def _parse_function(example_proto):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, feature)
    image = tf.decode_raw(parsed_features['train/image'], tf.uint8)
    image = tf.reshape(image, [Size_Y, Size_X])
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(parsed_features['train/label'], tf.float32)
    label = tf.reshape(label, [2])
    return image, label

def _parse_function_Reg(example_proto):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label_reg': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, feature)
    image = tf.decode_raw(parsed_features['train/image'], tf.uint8)
    image = tf.reshape(image, [Size_Y, Size_X])
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(parsed_features['train/label_reg'], tf.float32)
    label = tf.reshape(label, [2])
    return image, label

def _parse_function_3channel(example_proto):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, feature)
    image = tf.decode_raw(parsed_features['train/image'], tf.uint8)
    image = tf.reshape(image, [Size_Y, Size_X, 3])
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(parsed_features['train/label'], tf.float32)
    label = tf.reshape(label, [2])
    return image, label

def _parse_function_ellipseNet(example_proto):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, feature)
    image = tf.decode_raw(parsed_features['train/image'], tf.uint8)
    image = tf.reshape(image, [Size_Y, Size_X])
    image = tf.cast(image, tf.float32)
    #image = image.astype(np.float32)
    label = tf.decode_raw(parsed_features['train/label'], tf.uint8)
    label = tf.reshape(label, [Size_Y, Size_X])
    label = tf.cast(label, tf.float32)
    #label = label.astype(np.float32)
    return image, label


def _parse_function_pupilnet(example_proto):
    feature = {'train/slidewindow': tf.FixedLenFeature([], tf.string),
               'train/windowlabel': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, feature)
    image = tf.decode_raw(parsed_features['train/slidewindow'], tf.uint8)
    # image = tf.reshape(image, [89, 89])
    image = tf.reshape(image, [24, 24])
    label = tf.decode_raw(parsed_features['train/windowlabel'], tf.float32)
    label = tf.reshape(label, [2])
    return image, label

def _parse_function_Seg_PC_IEC(example_proto):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/seg_label': tf.FixedLenFeature([], tf.string),
               'train/pc_label': tf.FixedLenFeature([], tf.string),
               'train/iec_label': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, feature)

    image = tf.decode_raw(parsed_features['train/image'], tf.uint8)
    image = tf.reshape(image, [Size_Y, Size_X])
    image = tf.cast(image, tf.float32)

    seg_label = tf.decode_raw(parsed_features['train/seg_label'], tf.uint8)
    seg_label = tf.reshape(seg_label, [Size_Y, Size_X])
    seg_label = tf.cast(seg_label, tf.float32)

    pc_label = tf.decode_raw(parsed_features['train/pc_label'], tf.float32)
    pc_label = tf.reshape(pc_label, [2])

    iec_label = tf.decode_raw(parsed_features['train/iec_label'], tf.float32)
    iec_label = tf.reshape(iec_label, [2])
    return image, seg_label, pc_label, iec_label


def _parse_function_ellipse_coeff(example_proto):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, feature)
    image = tf.decode_raw(parsed_features['train/image'], tf.uint8)
    image = tf.reshape(image, [Size_Y, Size_X])
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(parsed_features['train/label'], tf.float32)
    label = tf.reshape(label, [5])
    return image, label


def make_batch_pc_iec_seg(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function_Seg_PC_IEC)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, seg_batch, pc_label_batch, iec_label_batch = iterator.get_next()

    return image_batch, seg_batch, pc_label_batch, iec_label_batch




def make_batch(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def make_batch_reg(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function_Reg)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch


def make_batch_pupilnet(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function_pupilnet)
    dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def make_batch_ellipseUNet(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function_ellipseNet)
    dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def make_batch_ellipse_only_once(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function_ellipseNet)
    # dataset = dataset.shuffle(buffer_size=199)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch


def make_batch_only_once(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.shuffle(buffer_size=199)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def make_batch_ellipse_coeff(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function_ellipse_coeff)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def make_batch_3channel(data_path, BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function_3channel)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def main():
    BATCH_SIZE = 20
    # IMAGE_BATCH, LABEL_BATCH = make_batch_ellipseUNet('D:/Users/Documents/Python/ExampleCode/tfRecords/ExCuSe/all/BinarySegment/384-288_ver2/train.tfrecords',
    #                                       BATCH_SIZE)
    IMAGE_BATCH, LABEL_BATCH, PC, IEC = make_batch_pc_iec_seg('D:/Users/Documents/Python/ExampleCode/tfRecords/HSY/10seq/Seg_PC_IEC/320-240/train.tfrecords',
                                          BATCH_SIZE)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        i = 0
        while True:

            try:
                img, seg, pc, iec = sess.run([IMAGE_BATCH, LABEL_BATCH, PC, IEC])
                # img_valid, label_valid = sess.run([IMAGE_BATCH_for_validation, LABEL_BATCH_for_validation])
                img2show = img[0].astype(np.float32)/255
                # img2show2 = img_valid[0].astype(np.float32)/255
                label2show = seg[0].astype(np.float32)/255
                img_plus_label = (img2show + label2show) / 2

                iec_0 = (int(iec[0][0]*320), int(iec[0][1]*240))
                cv2.circle(img_plus_label, iec_0, 3, 255, 3)
                # # cv2.imshow("result", outputImg)
                cv2.imshow("sdf", img_plus_label)
                # cv2.imshow("sdf2", img2show2)
                # print(label[0], label_valid[0])
                cv2.waitKey(0)
                # cv2.imshow("sdf", seg2show)
                # cv2.waitKey(0)

                # print(i)
                # i=i+1
            except tf.errors.OutOfRangeError:
                print('Error_OutOfRange')
                break


if __name__ == '__main__':
    main()
