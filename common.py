from __future__ import print_function

import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt
import argparse
import torch
import time
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import parameters_PupilLabs
import Model_UNet_Segmentation_PupilLabs
import cv2
import warnings
# import gpustat

def print_expected_time_train(total_batch, total_epoch, start_time, cost):
    s = "Expected time : %d hour %d min %d sec" % (((total_batch * total_epoch / 10 * cost) / 60) / 60,
                                                   (total_batch * total_epoch / 10 * cost) / 60 % 60,
                                                   (total_batch * total_epoch / 10 * cost) % 60)
    print(s)
    expected_end_time = time.localtime(start_time + (total_batch * total_epoch / 10 * cost))
    s = "\t\t%d.%d.%d. %02d:%02d:%02d" % (expected_end_time.tm_year, expected_end_time.tm_mon,
                                      expected_end_time.tm_mday, expected_end_time.tm_hour,
                                      expected_end_time.tm_min, expected_end_time.tm_sec)
    print(s)


def print_expected_time_test(numfile, start_time, cost):
    s = "Expected time : %d hour %d min %d sec" % (((numfile / 10 * cost) / 60) / 60,
                                                   (numfile / 10 * cost) / 60 % 60,
                                                   (numfile / 10 * cost) % 60)
    print(s)
    expected_end_time = time.localtime(start_time + (numfile / 10 * cost))
    s = "\t\t\t\t%d.%d.%d. %02d:%02d:%02d" % (expected_end_time.tm_year, expected_end_time.tm_mon,
                                        expected_end_time.tm_mday, expected_end_time.tm_hour,
                                        expected_end_time.tm_min, expected_end_time.tm_sec)
    print(s)

def print_report_each_epoch(cross_val_num, current_epoch, current_iter, avg_cost_seg, avg_cost_G, avg_cost_D, learning_rate):
    current_time = time.localtime(time.time())
    string_current_seq_time = str(cross_val_num).zfill(3) + ' Time : {:02}:{:02}:{:02}\t'.format(
        current_time.tm_hour, current_time.tm_min, current_time.tm_sec)
    string_current_epoch_iter = 'Epoch: {:03}\tIters: {:06}\t'.format(current_epoch, current_iter)
    string_S_loss = 'Seg Loss: {:.6f}\t'.format(avg_cost_seg)
    string_G_loss = 'Generator Loss: {:.6f}\t'.format(avg_cost_G)
    string_D_loss = 'Discriminator Loss: {:.6f}\t'.format(avg_cost_D)
    string_learning_rate = 'lr: {:.6f}'.format(learning_rate)

    print(
        string_current_seq_time + string_current_epoch_iter + string_S_loss + string_G_loss + string_D_loss + string_learning_rate)