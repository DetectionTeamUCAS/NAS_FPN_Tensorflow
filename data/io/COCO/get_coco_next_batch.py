# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import xml.etree.cElementTree as ET
import cv2
import numpy as np
import json
import os
from libs.label_name_dict import coco_dict
from libs.label_name_dict.label_dict import *


coco_trainvalmini = '/unsullied/sharefs/_research_detection/GeneralDetection/COCO/data/MSCOCO/odformat/coco_trainvalmini.odgt'


def next_img(step):
    with open(coco_trainvalmini) as f:
        files = f.readlines()

    total_imgs = len(files)
    if step % total_imgs == 0:
        np.random.shuffle(files)

    raw_line = files[step % total_imgs]
    file = json.loads(raw_line)
    img_name = file['ID']
    # img_height, img_width = file['height'], file['width']

    img = cv2.imread(file['fpath'])

    gtboxes = file['gtboxes']

    gtbox_label = []
    for gt in gtboxes:
        box = gt['box']
        label = gt['tag']
        gtbox_label.append([box[0], box[1], box[0]+box[2], box[1]+box[3], NAME_LABEL_MAP[label]])

    gtbox_and_label_list = np.array(gtbox_label, dtype=np.int32)
    if gtbox_and_label_list.shape[0] == 0:
        return next_img(step+1)
    else:
        return img_name, img[:, :, ::-1], gtbox_and_label_list


if __name__ == '__main__':

    imgid, img,  gtbox = next_img(3234)

    print("::")
    from libs.box_utils.draw_box_in_img import draw_boxes_with_label_and_scores

    img = draw_boxes_with_label_and_scores(img_array=img, boxes=gtbox[:, :-1], labels=gtbox[:, -1],
                                           scores=np.ones(shape=(len(gtbox), )))
    print("_----")

    cv2.imwrite("test.jpg", img)



