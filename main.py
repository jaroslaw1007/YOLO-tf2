import os
import sys

import tensorflow as tf
import numpy as np

from yolo_model import YOLO
from dataloader import DatasetGenerator
from yolo_loss import YOLOLoss
from lr_schedule import CosineAnnealingLR
from config import *

@tf.function
def train_step(model, optimizer, criterion, inputs, train_loss_metric_total,
                       train_loss_metric_class, train_loss_metric_object, train_loss_metric_noobject, train_loss_metric_coord):
    
    images, labels, objects_num = inputs
    with tf.GradientTape() as tape:
        outputs = model(images)

        n1 = CELL_SIZE * CELL_SIZE * NUM_CLASSES
        n2 = n1 + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL
        if VERSION == 1:
            class_probs = tf.reshape(outputs[:, 0:n1], (-1, 7, 7, 20))
            scales = tf.reshape(outputs[:, n1:n2], (-1, 7, 7, 2))
            boxes = tf.reshape(outputs[:, n2:], (-1, 7, 7, 2*4))
            predicts = tf.concat([class_probs, scales, boxes], 3)
        else:
            pred = tf.reshape(outputs, (-1, 13, 13, 25))
            class_probs = pred[:, :, :, :20]
            scales = pred[:, :, :, 20:21]
            boxes = pred[:, :, :, 21:]
            predicts = tf.concat([class_probs, scales, boxes])
        
        class_l, object_l, noobject_l, coord_l = criterion.loss(predicts, labels, objects_num)
        total = class_l + object_l + noobject_l + coord_l
        train_loss_metric_total(total)
        train_loss_metric_class(class_l)
        train_loss_metric_object(object_l)
        train_loss_metric_noobject(noobject_l)
        train_loss_metric_coord(coord_l)

    grads = tape.gradient(total, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

def train(model):
    train_dataset = DatasetGenerator().generate()
    optimizer = tf.keras.optimizers.SGD(
          learning_rate=LEARNING_RATE,
          momentum=MOMENTUM,
    )
    #optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    criterion = YOLOLoss()
    #scheduler = CosineAnnealingLR(optimizer, len(list(train_dataset)), WARMUP)
    train_loss_metric_total = tf.keras.metrics.Mean(name='total_loss')
    train_loss_metric_class = tf.keras.metrics.Mean(name='class_loss')
    train_loss_metric_object = tf.keras.metrics.Mean(name='object_loss')
    train_loss_metric_noobject = tf.keras.metrics.Mean(name='noobject_loss')
    train_loss_metric_coord = tf.keras.metrics.Mean(name='coord_loss')

    steps = 0
    for epoch in range(EPOCHS):
        train_loss_metric_total.reset_states()
        train_loss_metric_class.reset_states()
        train_loss_metric_object.reset_states()
        train_loss_metric_noobject.reset_states()
        train_loss_metric_coord.reset_states()
        #scheduler.step(steps)
        print('Epoch {:3d} lr={:.8f}'.format(epoch, optimizer.learning_rate.read_value()))
            
        for batch_idx, inputs in enumerate(train_dataset):
            train_step(model, optimizer, criterion, inputs, train_loss_metric_total,
                       train_loss_metric_class, train_loss_metric_object, train_loss_metric_noobject, train_loss_metric_coord)
            print('  Batch {:2d} total_loss={:.6f} class_loss={:.6f} object_loss={:.6f} noobject_loss={:.6f} \
coord_loss={:.6f}'.format(batch_idx, train_loss_metric_total.result(), train_loss_metric_class.result(), \
                                              train_loss_metric_object.result(), train_loss_metric_noobject.result(), \
                                              train_loss_metric_coord.result()), end='\r')
            steps += 1

        print()
        if (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(CKPT_DIR, 'ckpt_yolo_InceptRes_aug_freeze_'+str(epoch + 95)))

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    model = YOLO()
    model.load_weights(sys.argv[1])
 
    train(model)
