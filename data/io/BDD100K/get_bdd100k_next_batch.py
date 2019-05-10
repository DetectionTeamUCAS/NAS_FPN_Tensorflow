# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import xml.etree.cElementTree as ET
import cv2
import numpy as np
import os
from libs.label_name_dict import coco_dict
from libs.label_name_dict.label_dict import *


root_path = '/unsullied/sharefs/yangxue/isilon/yangxue/data/BDD100K/BDD100K_VOC/bdd100k_train/'
xmls = os.listdir(os.path.join(root_path, 'Annotations'))
total_imgs = len(xmls)

# print (NAME_LABEL_DICT)


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(int(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def next_img(step):

    if step % total_imgs == 0:
        np.random.shuffle(xmls)
    xml_name = xmls[step % total_imgs]
    img_name = xml_name.replace('.xml', '.jpg')

    img = cv2.imread(os.path.join(root_path, 'train', img_name))

    img_height, img_width, gtbox_label = read_xml_gtbox_and_label(os.path.join(root_path, 'Annotations', xml_name))

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
    print ("_----")


    cv2.imshow("test", img)
    cv2.waitKey(0)


