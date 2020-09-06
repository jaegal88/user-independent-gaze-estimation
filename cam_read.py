from __future__ import print_function
import sys
# sys.path.append("C:/Users/HANSY_LAB/Anaconda3/envs/camera2/Lib\site-packages")
import uvc
import logging
import cv2
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import torch
import Model_UNet_Segmentation_PupilLabs
import numpy as np
import time
opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
conf = tf.ConfigProto(gpu_options=opts)
tfe.enable_eager_execution(config=conf)
device = torch.device("cuda")
logging.basicConfig(level=logging.INFO)

dev_list = uvc.device_list()
print(dev_list)
cap_world = uvc.Capture(dev_list[1]["uid"])
cap_right = uvc.Capture(dev_list[0]["uid"])
cap_left = uvc.Capture(dev_list[2]["uid"])

# Uncomment the following lines to configure the Pupil 200Hz IR cameras:
# controls_dict = dict([(c.display_name, c) for c in cap.controls])
# controls_dict['Auto Exposure Mode'].value = 1
# controls_dict['Gamma'].value = 200
print(cap_world.avaible_modes)
cap_world.frame_mode = (1280, 720, 60)
cap_left.frame_mode = (320, 240, 120)
cap_right.frame_mode = (320, 240, 120)

model = Model_UNet_Segmentation_PupilLabs.UNet4f16ch_seg_reg()
print('Loading Model....')
model.load_state_dict(torch.load('./trained_model/Seg_PC_IEC_temp/my_test_model_00020000iters.pt'))
model.eval()
model.to(device)
print('Model Loaded!')
while True:
    frame_world = cap_world.get_frame_robust()
    # frame_left = cap_left.get_frame_robust()
    frame_right = cap_right.get_frame_robust()
    # frame_left = cv2.flip(frame_left, 1)
    cv2.imshow("img_w", frame_world.img)
    # frame_left.gray
    # right_gray = cv2.flip(frame_right.gray, 0)
    # left_gray = frame_left.gray
    left_gray = cv2.flip(frame_right.gray, 0)
    left_gray = left_gray[np.newaxis, np.newaxis, :]
    left_gray = left_gray.astype(np.float32) / 255
    left_gray = torch.from_numpy(left_gray)
    left_gray = left_gray.to(device)

    out_seg, out_iec, _ = model(left_gray)


    out_seg_cp = out_seg[0, 0].detach().cpu().numpy()
    out_seg_thre = out_seg_cp.copy()
    out_seg_thre[out_seg_thre < 0.5] = 0
    out_seg_thre[out_seg_thre >= 0.5] = 1



    if np.count_nonzero(out_seg_thre) != 0:
        _, labels, stats, center = cv2.connectedComponentsWithStats(out_seg_thre.astype(np.uint8))
        confs = np.zeros((len(center)), np.float32)
        for ind in range(len(center)):
            confs[ind] = sum(out_seg_cp[labels == ind])
        pupil_candidate = np.argmax(confs)
        out_seg_thre[labels != pupil_candidate] = 0

    out_seg_cp[out_seg_thre == 0] = 0
    confidence = np.sum(out_seg_cp)/(np.sum(out_seg_thre) + 1e-5)
    print('{:.5f}'.format(confidence))
    if confidence > .97:
        print('Eye Detected')
    else:
        print('Blink')
        out_seg_thre = 0
    iec_0 = (int(out_iec[0][0] * 320), int(out_iec[0][1] * 240))
    img_plus_label = (cv2.flip(frame_right.gray, 0) / 255 + out_seg_thre) / 2
    cv2.circle(img_plus_label, iec_0, 3, 255, 2)

    # cv2.imshow("img_l", frame_left.gray)
    cv2.imshow("img_seg", img_plus_label)
    # cv2.imshow("img_r", frame_right.gray)
    key = cv2.waitKey(1)
    # print(end_time-start_time)
    if key == ord('c') or key == ord('C') or key == 27:
        break

cap = None
