# 作者:王勇
# 开发时间:2024/3/19 10:26
import tensorflow as tf
from keras import layers
from tensorflow import keras


class criss_cross_attention_Affinity(tf.keras.layers.Layer):

    def __init__(self, axis=1, **kwargs):
        super(criss_cross_attention_Affinity, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        batch_size, H, W, Channel = x.shape
        outputs = []
        for i in range(H):
            for j in range(W):
                ver = x[:, i, j, :]
                temp_x = tf.concat([x[:, i, 0:j, :], x[:, i, j + 1:W, :], x[:, :, j, :]], axis=1)
                trans_temp = tf.matmul(temp_x, tf.expand_dims(ver, -1))
                trans_temp = tf.squeeze(trans_temp, -1)
                trans_temp = tf.expand_dims(trans_temp, axis=1)
                outputs.append(trans_temp)
        outputs = layers.Concatenate(axis=self.axis)(outputs)
        C = outputs.shape[2]
        outputs = tf.reshape(outputs, [-1, H, W, C])
        return outputs

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(criss_cross_attention_Affinity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class criss_cross_attention_Aggregation(tf.keras.layers.Layer):

    def __init__(self, axis=1, **kwargs):
        super(criss_cross_attention_Aggregation, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x, Affinity):
        batch_size, H, W, Channel = x.shape
        Affinity = layers.Activation('softmax')(Affinity)
        outputs = []
        for i in range(H):
            for j in range(W):
                ver = Affinity[:, i, j, :]
                temp_x = tf.concat([x[:, i, 0:j, :], x[:, i, j + 1:W, :], x[:, :, j, :]], axis=1)
                trans_temp = tf.matmul(tf.transpose(tf.expand_dims(ver, -1), [0, 2, 1]), temp_x)
                outputs.append(trans_temp)
        outputs = layers.Concatenate(axis=self.axis)(outputs)
        C = outputs.shape[2]
        outputs = tf.reshape(outputs, [-1, H, W, C])
        return outputs

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(criss_cross_attention_Aggregation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def criss_cross_attention(x):
    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=2)(x)
    x_origin = x
    affinity = criss_cross_attention_Affinity(1)(x)
    out = criss_cross_attention_Aggregation(1)(x, affinity)
    out = layers.Add()([out, x_origin])
    out = layers.UpSampling2D(size=2, interpolation='bilinear')(out)
    return out


