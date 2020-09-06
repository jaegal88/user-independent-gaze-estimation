import tensorflow as tf
import OS_Server_Set
Size_X = 320
Size_Y = 240
os_server = OS_Server_Set.os_server_num

# batchsize = 1
batchsize = 10

cam_world_num = 1
cam_right_num = 0
cam_left_num = 2


shuffle_buffer = 16000
total_epoch = 100
initial_learning_rate = 1e-4
final_learning_rate = 1e-6
rate_decay = 0.95

def get_digits(text):
    return filter(str.isdigit, text)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

feature = {'train/image': tf.FixedLenFeature([], tf.string),
           'train/label': tf.FixedLenFeature([], tf.string)}

feature_GAN = {'train/image': tf.FixedLenFeature([], tf.string),
           'train/label': tf.FixedLenFeature([], tf.string),
            'train/seq_num': tf.FixedLenFeature([], tf.string)}

feature_Seg_Reg = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label_seg': tf.FixedLenFeature([], tf.string),
                   'train/label_reg': tf.FixedLenFeature([], tf.string)}

feature_Seg_PC_IEC = {
                'train/image': tf.FixedLenFeature([], tf.string),
               'train/seg_label': tf.FixedLenFeature([], tf.string),
               'train/pc_label': tf.FixedLenFeature([], tf.string),
               'train/iec_label': tf.FixedLenFeature([], tf.string)}


def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, feature)

def _parse_image_function_GAN(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, feature_GAN)

def _parse_image_function_Seg_Reg(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, feature_Seg_Reg)

def _parse_image_function_Seg_PC_IEC(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, feature_Seg_PC_IEC)

def calc_lr(initial_lr, steps, global_step):
    """Sets the learning rate to the initial LR"""
    lr = initial_lr * (rate_decay ** (steps//global_step))
    if lr < final_learning_rate:
        lr = final_learning_rate
    return lr

def adjust_learning_rate(optimizer, num_iter, global_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = calc_lr(initial_learning_rate, num_iter, global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_GAN(optimizer, num_iter, global_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = calc_lr(initial_learning_rate, num_iter, global_step)*1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def show_memusage(device=0):
#     gpu_stats = gpustat.GPUStatCollection.new_query()
#     item = gpu_stats.jsonify()["gpus"][device]
#     print("{}/{}".format(item["memory.used"], item["memory.total"]))

C_END = "\033[0m"
C_BOLD = "\033[1m"
C_INVERSE = "\033[7m"

C_BLACK = "\033[30m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_PURPLE = "\033[35m"
C_CYAN = "\033[36m"
C_WHITE = "\033[37m"

C_BGBLACK = "\033[40m"
C_BGRED = "\033[41m"
C_BGGREEN = "\033[42m"
C_BGYELLOW = "\033[43m"
C_BGBLUE = "\033[44m"
C_BGPURPLE = "\033[45m"
C_BGCYAN = "\033[46m"
C_BGWHITE = "\033[47m"
