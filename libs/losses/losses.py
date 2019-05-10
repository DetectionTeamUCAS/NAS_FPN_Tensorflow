# -*- coding: utf-8 -*-
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from libs.box_utils import encode_and_decode
from libs.configs import cfgs


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box

def smooth_l1_loss_rpn(bbox_pred, bbox_targets, label, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4]
    :param bbox_targets: [-1, 4]
    :param label: [-1]
    :param sigma:
    :return:
    '''
    value = _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = tf.reduce_sum(value, axis=1)  # to sum in axis 1
    rpn_positive = tf.where(tf.greater(label, 0))

    # rpn_select = tf.stop_gradient(rpn_select) # to avoid
    selected_value = tf.gather(value, rpn_positive)
    non_ignored_mask = tf.stop_gradient(
        1.0 - tf.to_float(tf.equal(label, -1)))  # positve is 1.0 others is 0.0

    bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))

    return bbox_loss

def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
    '''

    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])
    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask, 1)*outside_mask) / normalizer

    return bbox_loss

def smooth_l1_loss_rcnn_iou(bbox_pred, bbox_targets, label, ious, num_classes, sigma=1.0):
    '''

    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    ious = tf.reshape(ious, [-1, ])
    ious = tf.stop_gradient(ious)

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])

    tmp = tf.reduce_sum(value * inside_mask, 1)
    tmp = tf.stop_gradient(tmp)
    iou_factor = (1 - ious) / tmp
    iou_factor = tf.stop_gradient(iou_factor)

    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask, 1)*outside_mask*iou_factor) / normalizer

    return bbox_loss


def iou_loss_(label, ious):
    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    ious = tf.reshape(ious, [-1, ])

    normalizer = tf.to_float(tf.shape(ious)[0])

    bbox_loss = tf.reduce_sum(-tf.log(ious+1e-5) * outside_mask) / normalizer

    return bbox_loss


def iou_loss(bbox_pred, bbox_targets, gtbox, label, num_classes):
    """
    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets: [-1, (cfgs.CLS_NUM +1) * 4]
    :param gtbox: [-1, 4]
    :param label: [-1]
    :param num_classes:
    :return:
    """

    gtbox = tf.tile(gtbox, [1, num_classes])
    bbox_pred = tf.reshape(bbox_pred, [-1, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, 4])
    gtbox = tf.reshape(gtbox, [-1, 4])
    pred_box = encode_and_decode.decode_boxes(bbox_pred, gtbox, scale_factors=cfgs.ROI_SCALE_FACTORS)
    gt_box = encode_and_decode.decode_boxes(bbox_targets, gtbox, scale_factors=cfgs.ROI_SCALE_FACTORS)

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.reshape(inside_mask, [-1, ])
    iou = iou_calculate(pred_box, gt_box)
    iou_loss = tf.reduce_mean(-tf.log(iou*inside_mask+1e-5))

    pred = tf.cast(tf.greater(iou, 0.5), tf.float32)
    pred = tf.reshape(pred, [-1, num_classes])
    pred_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label))

    loss = iou_loss * 0.1 + pred_loss * 0.0
    return loss

def iou_calculate(boxes_1, boxes_2):

    with tf.name_scope('iou_caculate'):

        xmin_1, ymin_1, xmax_1, ymax_1 = tf.unstack(boxes_1, axis=1)  # ymin_1 shape is [N, 1]..

        xmin_2, ymin_2, xmax_2, ymax_2 = tf.unstack(boxes_2, axis=1)  # ymin_2 shape is [M, ]..

        max_xmin = tf.maximum(xmin_1, xmin_2)
        min_xmax = tf.minimum(xmax_1, xmax_2)

        max_ymin = tf.maximum(ymin_1, ymin_2)
        min_ymax = tf.minimum(ymax_1, ymax_2)

        overlap_h = tf.maximum(0., min_ymax - max_ymin)  # avoid h < 0
        overlap_w = tf.maximum(0., min_xmax - max_xmin)

        overlaps = overlap_h * overlap_w

        area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
        area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

        iou = overlaps / (area_1 + area_2 - overlaps)

        return iou



def sum_ohem_loss(cls_score, label, bbox_pred, bbox_targets,
                  nr_ohem_sampling, nr_classes, sigma=1.0):

    raise NotImplementedError('not implement Now. YJR will implemetn in the future')