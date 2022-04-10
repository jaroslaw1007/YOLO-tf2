import tensorflow as tf
from tensorflow.keras import layers

from config import *

def Conv2D(channels, kernel_size, stride, bias=True):
    return layers.Conv2D(
        channels,
        kernel_size,
        strides=stride,
        padding='same',
        use_bias=bias,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)
    )

class ConvBnLeakyReLU(layers.Layer):
    def __init__(self, channels, kernel_size, stride):
        super(ConvBnLeakyReLU, self).__init__()

        self.conv = tf.keras.Sequential([
            Conv2D(channels, kernel_size, stride),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1)
        ])

    def call(self, x, training):
        return self.conv(x, training=training)
 
class Linear(layers.Layer):
    def __init__(self, features, activation=None):
        super(Linear, self).__init__()

        self.linear = layers.Dense(
            features,
            activation=activation,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)
        )

    def call(self, x, training):
        return self.linear(x, training=training)

class YOLO(tf.keras.Model):
    def __init__(self, name='YOLO', **kwargs):
        super(YOLO, self).__init__(name=name, **kwargs)

        self.backbone_model = tf.keras.applications.InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
        )
        # self.backbone_model.trainable = False

        self.additional_conv = tf.keras.Sequential([
            ConvBnLeakyReLU(1024, 3, 1),
            ConvBnLeakyReLU(1024, 3, 2),
            ConvBnLeakyReLU(1024, 3, 1),
            ConvBnLeakyReLU(1024, 3, 1)
        ])

        self.fc_layers = tf.keras.Sequential([
            layers.Flatten(),
            Linear(4096),
            layers.LeakyReLU(0.1),
            layers.Dropout(0.5),
            Linear(1470)
        ])

    def call(self, x, training=True):
        x = self.backbone_model(x, training=training)
        x = self.additional_conv(x, training=training)
        y = self.fc_layers(x, training=training)

        return y

class YOLOv2(tf.keras.Model):
    def __init__(self, name='YOLOv2', **kwargs):
        super(YOLOv2, self).__init__(name=name, **kwargs)

        self.backbone_model = self._vgg16_layers(['block5_conv3', 'block5_pool'])

        self.additional_conv1 = tf.keras.Sequential([
            ConvBnLeakyReLU(1024, 3, 1),
            ConvBnLeakyReLU(1024, 3, 1)
        ])
        self.skip_connect_conv = ConvBnLeakyReLU(64, 1, 1)
        self.additional_conv2 = ConvBnLeakyReLU(1024, 3, 1)

        self.detection_conv = Conv2D(125, 1, 1, True)

    def _vgg16_layers(self, layer_names):
        vgg16 = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(416, 416, 3)
        )

        outputs = [vgg16.get_layer(layer_name).output for layer_name in layer_names]
        model = tf.keras.Model([vgg16.input], outputs)

        return model

    def call(self, x, training=True):
        skip_connect, x = self.backbone_model(x, training=training)
        x = self.additional_conv1(x, training=training)

        skip_connect = self.skip_connect_conv(skip_connect, training=training)
        skip_connect = tf.nn.space_to_depth(skip_connect, 2)

        x = layers.concatenate([skip_connect, x])
        x = self.additional_conv2(x, training=training)
        y = self.detection_conv(x, training=training)

        return y
