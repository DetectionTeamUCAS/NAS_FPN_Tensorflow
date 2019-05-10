# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import cv2
import argparse
import numpy as np
sys.path.append('../../')

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.box_utils import draw_box_in_img
from help_utils import tools


def load_graph(frozen_graph_file):

    # we parse the graph_def file
    with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            name="",
                            op_dict=None,
                            producer_op_list=None)
    return graph


def test(frozen_graph_path, test_dir):

    graph = load_graph(frozen_graph_path)
    print("we are testing ====>>>>", frozen_graph_path)

    img = graph.get_tensor_by_name("input_img:0")
    dets = graph.get_tensor_by_name("DetResults:0")

    with tf.Session(graph=graph) as sess:
        for img_path in os.listdir(test_dir):
            a_img = cv2.imread(os.path.join(test_dir, img_path))[:, :, ::-1]
            st = time.time()
            dets_val = sess.run(dets, feed_dict={img: a_img})

            show_indices = dets_val[:, 1] >= 0.5
            dets_val = dets_val[show_indices]
            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(a_img,
                                                                                boxes=dets_val[:, 2:],
                                                                                labels=dets_val[:, 0],
                                                                                scores=dets_val[:, 1])
            cv2.imwrite(img_path,
                        final_detections[:, :, ::-1])
            print("%s cost time: %f" % (img_path, time.time() - st))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    test('/home/yangxue/isilon/yangxue/code/yxdet/FPN_TF_DEV/output/Pbs/FPN_Res50_COCO_Frozen.pb',
         '/unsullied/sharefs/yangxue/isilon/yangxue/data/COCO/train2017')











