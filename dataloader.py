import tensorflow as tf
import numpy as np

from config import *

class DatasetGenerator:
    def __init__(self):
        self.image_names = []
        self.record_list = []
        self.object_num_list = []
        input_file = open(DATA_PATH, 'r')

        for line in input_file:
            line = line.strip()
            ss = line.split(' ')
            self.image_names.append(ss[0])

            self.record_list.append([float(num) for num in ss[1:]])

            self.object_num_list.append(min(len(self.record_list[-1])//5, MAX_OBJECTS_PER_IMAGE))
            if len(self.record_list[-1]) < MAX_OBJECTS_PER_IMAGE * 5:
                self.record_list[-1] = self.record_list[-1] + \
                    [0., 0., 0., 0., 0.] * (MAX_OBJECTS_PER_IMAGE-len(self.record_list[-1])//5)
            elif len(self.record_list[-1]) > MAX_OBJECTS_PER_IMAGE * 5:
                self.record_list[-1] = self.record_list[-1][:MAX_OBJECTS_PER_IMAGE*5]

    """
    def preprocess(self, image_name, raw_labels, object_num):
        image_file = tf.io.read_file(IMAGE_DIR+image_name)
        image = tf.io.decode_jpeg(image_file, channels=3)

        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        
        raw_labels = tf.cast(tf.reshape(raw_labels, [-1, 5]), tf.float32)
        
        class_num = raw_labels[:, 4]
        
        aug_image, aug_bbox, aug_h, aug_w = AugmentData(image / 255, raw_labels, h, w, object_num)
        
        width_rate = IMAGE_SIZE * 1.0 / tf.cast(aug_w, tf.float32) 
        height_rate = IMAGE_SIZE * 1.0 / tf.cast(aug_h, tf.float32) 
        
        xmin = aug_bbox[:, 0]
        ymin = aug_bbox[:, 1]
        xmax = aug_bbox[:, 2]
        ymax = aug_bbox[:, 3]

        aug_image = tf.image.resize(aug_image, size=[IMAGE_SIZE, IMAGE_SIZE])
        #aug_image = aug_image / 255
        #aug_image = aug_image[..., ::-1]
        #aug_image -= VGG_MEANS

        xcenter = (xmin + xmax) * 1.0 / 2.0 * width_rate
        ycenter = (ymin + ymax) * 1.0 / 2.0 * height_rate

        box_w = (xmax - xmin) * width_rate
        box_h = (ymax - ymin) * height_rate

        labels = tf.stack([xcenter, ycenter, box_w, box_h, class_num], axis = 1)

        return aug_image, labels, tf.cast(object_num, tf.int32)
    
    """
    def preprocess(self, image_name, raw_labels, object_num):
        image_file = tf.io.read_file(IMAGE_DIR+image_name)
        image = tf.io.decode_jpeg(image_file, channels=3)

        h = tf.shape(image)[0]
        w = tf.shape(image)[1]

        width_rate = IMAGE_SIZE * 1.0 / tf.cast(w, tf.float32) 
        height_rate = IMAGE_SIZE * 1.0 / tf.cast(h, tf.float32) 

        image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255
        #image = tf.keras.applications.vgg16.preprocess_input(image[tf.newaxis, :])[0]

        raw_labels = tf.cast(tf.reshape(raw_labels, [-1, 5]), tf.float32)

        xmin = raw_labels[:, 0]
        ymin = raw_labels[:, 1]
        xmax = raw_labels[:, 2]
        ymax = raw_labels[:, 3]
        class_num = raw_labels[:, 4]

        xcenter = (xmin + xmax) * 1.0 / 2.0 * width_rate
        ycenter = (ymin + ymax) * 1.0 / 2.0 * height_rate

        box_w = (xmax - xmin) * width_rate
        box_h = (ymax - ymin) * height_rate

        labels = tf.stack([xcenter, ycenter, box_w, box_h, class_num], axis = 1)

        return image, labels, tf.cast(object_num, tf.int32)

    def generate(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            self.image_names, 
            np.array(self.record_list), 
            np.array(self.object_num_list)
        ))
        dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(BATCH_SIZE)

        return dataset
