import tensorflow as tf
import numpy as np

from config import *

class YOLOLoss(object):
    def __init__(self):
        self.base_boxes = self._initialize_base_boxes()
        if VERSION == 2:
            self.anchors = tf.reshape(tf.constant(ANCHORS, dtype=tf.float32), [1, 1, BOXES_PER_CELL, 2])

    def _initialize_base_boxes(self):
        base_boxes = np.zeros([CELL_SIZE, CELL_SIZE, 4])

        for y in range(CELL_SIZE):
            for x in range(CELL_SIZE):
                base_boxes[y, x, :] = [x, y, 0, 0]

        return np.tile(np.resize(base_boxes, [CELL_SIZE, CELL_SIZE, 1, 4]), [1, 1, BOXES_PER_CELL, 1])

    def loss(self, predicts, labels, objects_num):
        batch_size = predicts.shape[0]
        class_losses = 0.0
        object_losses = 0.0
        noobject_losses = 0.0
        coord_losses = 0.0

        for i in tf.range(batch_size):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            object_num = objects_num[i]

            for j in tf.range(object_num):
                if VERSION == 1:
                    class_l, object_l, noobject_l, coord_l = self._calculate_loss(predict, label[j:j+1, :])
                else:
                    class_l, object_l, noobject_l, coord_l = self._calculate_loss_anchors(predict, label[j:j+1, :])
                class_losses += class_l
                object_losses += object_l
                noobject_losses += noobject_l
                coord_losses += coord_l

        return class_losses / batch_size, object_losses / batch_size, noobject_losses / batch_size, coord_losses / batch_size

    def iou(self, boxes1, boxes2):
        boxes1 = tf.stack([
            boxes1[:, :, :, 0]-boxes1[:, :, :, 2]/2,
            boxes1[:, :, :, 1]-boxes1[:, :, :, 3]/2,
            boxes1[:, :, :, 0]+boxes1[:, :, :, 2]/2,
            boxes1[:, :, :, 1]+boxes1[:, :, :, 3]/2
        ])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])

        boxes2 =  tf.stack([
            boxes2[0]-boxes2[2]/2,
            boxes2[1]-boxes2[3]/2,
            boxes2[0]+boxes2[2]/2,
            boxes2[1]+boxes2[3]/2
        ])

        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        intersection = rd - lu 
        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
        mask = tf.cast(intersection[:, :, :, 0]>0, tf.float32) * tf.cast(intersection[:, :, :, 1]>0, tf.float32)
        inter_square = mask * inter_square

        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        return inter_square / (square1 + square2 - inter_square + 1e-6)

    def _calculate_loss(self, predict, label):
        label = tf.reshape(label, [-1])

        min_x = (label[0] - label[2] / 2) / (IMAGE_SIZE / CELL_SIZE)
        max_x = (label[0] + label[2] / 2) / (IMAGE_SIZE / CELL_SIZE)
        min_y = (label[1] - label[3] / 2) / (IMAGE_SIZE / CELL_SIZE)
        max_y = (label[1] + label[3] / 2) / (IMAGE_SIZE / CELL_SIZE)
        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)
        max_x = tf.minimum(tf.math.ceil(max_x), CELL_SIZE)
        max_y = tf.minimum(tf.math.ceil(max_y), CELL_SIZE)

        temp = tf.cast(tf.stack([max_y-min_y, max_x-min_x]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)
        temp = tf.cast(tf.stack([min_y, CELL_SIZE-max_y, min_x, CELL_SIZE-max_x]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, 'CONSTANT')

        center_x = label[0] / (IMAGE_SIZE / CELL_SIZE)
        center_x = tf.floor(center_x)
        center_y = label[1] / (IMAGE_SIZE / CELL_SIZE)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)
        temp = tf.cast(tf.stack([center_y, CELL_SIZE-center_y-1, center_x, CELL_SIZE-center_x-1]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, 'CONSTANT')

        predict_boxes = predict[:, :, NUM_CLASSES+BOXES_PER_CELL:]
        predict_boxes = tf.reshape(predict_boxes, [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4])
        predict_boxes = tf.stack([
            predict_boxes[..., 0]+self.base_boxes[..., 0],
            predict_boxes[..., 1]+self.base_boxes[..., 1],
            tf.square(predict_boxes[..., 2]),
            tf.square(predict_boxes[..., 3])
        ], axis=-1)
        predict_boxes = predict_boxes * [IMAGE_SIZE/CELL_SIZE, IMAGE_SIZE/CELL_SIZE, IMAGE_SIZE, IMAGE_SIZE]

        iou_predict_truth = self.iou(predict_boxes, label[0:4])
        C = iou_predict_truth * tf.reshape(response, (CELL_SIZE, CELL_SIZE, 1))
        I = iou_predict_truth * tf.reshape(response, (CELL_SIZE, CELL_SIZE, 1))
        max_I = tf.reduce_max(I, 2, keepdims=True)
        I = tf.cast((I>=max_I), tf.float32) * tf.reshape(response, (CELL_SIZE, CELL_SIZE, 1))
        no_I = tf.ones_like(I, dtype=tf.float32) - I

        p_C = predict[:, :, NUM_CLASSES:NUM_CLASSES+BOXES_PER_CELL]

        x = label[0]
        y = label[1]
        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))

        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]
        p_sqrt_w = tf.sqrt(tf.minimum(IMAGE_SIZE*1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(IMAGE_SIZE*1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

        P = tf.one_hot(tf.cast(label[4], tf.int32), NUM_CLASSES, dtype=tf.float32)
        
        p_P = predict[:, :, 0:NUM_CLASSES]

        class_loss = tf.nn.l2_loss(tf.reshape(objects, (CELL_SIZE, CELL_SIZE, 1))*(p_P-P)) * CLASS_SCALE
        object_loss = tf.nn.l2_loss(I*(p_C-C)) * OBJECT_SCALE
        noobject_loss = tf.nn.l2_loss(no_I*(p_C)) * NOOBJECT_SCALE
        coord_loss = (tf.nn.l2_loss(I*(p_x-x)/(IMAGE_SIZE/CELL_SIZE)) +
            tf.nn.l2_loss(I*(p_y-y)/(IMAGE_SIZE/CELL_SIZE)) +
            tf.nn.l2_loss(I*(p_sqrt_w-sqrt_w))/IMAGE_SIZE +
            tf.nn.l2_loss(I*(p_sqrt_h-sqrt_h))/IMAGE_SIZE) * COORD_SCALE

        return class_loss, object_loss, noobject_loss, coord_loss

    def _calculate_loss_anchors(self, predict, label):
        # (num_classes, confidnece, x_center, y_center, w, h)
        predict = tf.reshape(predict, [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, NUM_CLASSES+5])
        label = tf.reshape(label, [-1])

        min_x = (label[0] - label[2] / 2) / (IMAGE_SIZE / CELL_SIZE)
        max_x = (label[0] + label[2] / 2) / (IMAGE_SIZE / CELL_SIZE)
        min_y = (label[1] - label[3] / 2) / (IMAGE_SIZE / CELL_SIZE)
        max_y = (label[1] + label[3] / 2) / (IMAGE_SIZE / CELL_SIZE)
        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)
        max_x = tf.minimum(tf.math.ceil(max_x), CELL_SIZE)
        max_y = tf.minimum(tf.math.ceil(max_y), CELL_SIZE)

        temp = tf.cast(tf.stack([max_y-min_y, max_x-min_x]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)
        temp = tf.cast(tf.stack([min_y, CELL_SIZE-max_y, min_x, CELL_SIZE-max_x]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, 'CONSTANT')

        center_x = label[0] / (IMAGE_SIZE / CELL_SIZE)
        center_x = tf.floor(center_x)
        center_y = label[1] / (IMAGE_SIZE / CELL_SIZE)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)
        temp = tf.cast(tf.stack([center_y, CELL_SIZE-center_y-1, center_x, CELL_SIZE-center_x-1]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, 'CONSTANT')

        predict_boxes = predict[:, :, :, NUM_CLASSES+1:]
        predict_boxes = tf.stack([
            tf.nn.sigmoid(predict_boxes[..., 0])+self.base_boxes[..., 0],
            tf.nn.sigmoid(predict_boxes[..., 1])+self.base_boxes[..., 1],
            tf.exp(predict_boxes[..., 2]*self.anchors[..., 0],),
            tf.exp(predict_boxes[..., 3]*self.anchors[..., 1])
        ], axis=-1)
        predict_boxes = predict_boxes * [IMAGE_SIZE/CELL_SIZE, IMAGE_SIZE/CELL_SIZE, IMAGE_SIZE, IMAGE_SIZE]

        iou_predict_truth = self.iou(predict_boxes, label[0:4])

        C = iou_predict_truth * tf.reshape(response, (CELL_SIZE, CELL_SIZE, 1))
        C = tf.reshape(C, (CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 1))

        I = iou_predict_truth * tf.reshape(response, (CELL_SIZE, CELL_SIZE, 1))
        max_I = tf.reduce_max(I, 2, keepdims=True)
        I = tf.cast((I>=max_I), tf.float32) * tf.reshape(response, (CELL_SIZE, CELL_SIZE, 1))
        I = tf.reshape(I, (CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 1))
        no_I = tf.ones_like(I, dtype=tf.float32) - I

        p_C = predict[:, :, :, NUM_CLASSES:NUM_CLASSES+1]

        x = label[0]
        y = label[1]
        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))

        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]
        p_sqrt_w = tf.sqrt(tf.minimum(IMAGE_SIZE*1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(IMAGE_SIZE*1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))

        P = tf.one_hot(tf.cast(label[4], tf.int32), NUM_CLASSES, dtype=tf.float32)

        p_P = tf.nn.softmax(predict[:, :, :, :NUM_CLASSES])

        class_loss = tf.nn.l2_loss(tf.reshape(objects, (CELL_SIZE, CELL_SIZE, 1, 1))*(p_P-P)) * CLASS_SCALE
        object_loss = tf.nn.l2_loss(I*(p_C-C)) * OBJECT_SCALE
        noobject_loss = tf.nn.l2_loss(no_I*(p_C)) * NOOBJECT_SCALE
        coord_loss = (tf.nn.l2_loss(I*(p_x-x)/(IMAGE_SIZE/CELL_SIZE)) +
            tf.nn.l2_loss(I*(p_y-y)/(IMAGE_SIZE/CELL_SIZE)) +
            tf.nn.l2_loss(I*(p_sqrt_w-sqrt_w))/IMAGE_SIZE +
            tf.nn.l2_loss(I*(p_sqrt_h-sqrt_h))/IMAGE_SIZE) * COORD_SCALE

        return class_loss + object_loss + noobject_loss + coord_loss
