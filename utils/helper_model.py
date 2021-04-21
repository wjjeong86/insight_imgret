
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import group_norm


# =-------=-------=-------=------- Layer Functions



# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    z_shape = K.shape(z_mean)
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=z_shape)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[-1],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

def bias(x):
    l = BiasLayer()
    return l(x)

def conv2d(x, c, k, s=(1, 1), d_rate=(1, 1), padding="SAME", use_bias=False):
    l = keras.layers.Conv2D(
        c, k, s, dilation_rate=d_rate, padding=padding, use_bias=use_bias
    )
    return l(x)


def dwconv2d(x, k, s=(1, 1), padding='SAME', use_bias=False):
    l = keras.layers.DepthwiseConv2D(
        k, s, padding=padding, depth_multiplier=1, use_bias=use_bias
    )
    return l(x)


def bn(x):
    return keras.layers.BatchNormalization()(x)

def gn(x, group=32):
    N,H,W,C = x.shape.as_list()
    if C<group: group = C
    return group_norm.GroupNormalization(group)(x)

def relu(x):
    return keras.layers.Activation('relu')(x)

def swish(x):
    return x * keras.backend.sigmoid(x)


def sigmoid(x):
    return keras.backend.sigmoid(x)


def max_pool(x, k=(3,3), s=(2,2)):
    return keras.layers.MaxPool2D(k, s, "same")(x)


def avg_pool(x,k=3,s=2):
    return keras.layers.AvgPool2D(k,s, "same")(x)


def upsample(x, mul):
    _, H, W, _ = x.shape.as_list()
    return tf.image.resize(x, (int(H * mul), int(W * mul)))


# def dropout(x, rate=ph_dropout_rate):
#     return tf.nn.dropout(x, rate)

from tensorflow.keras.layers import Dense, Wrapper
import tensorflow.keras.backend as K


class DropConnectDense(Dense):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True
        super(DropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x, mask=None):
        if 0. < self.prob < 1.:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob), self.kernel)
            self.b = K.in_train_phase(K.dropout(self.b, self.prob), self.b)

        # Same as original
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)


class DropConnect(Wrapper):
    def __init__(self, layer, prob=1., **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob) * (1-self.prob), self.layer.kernel)
            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob) * (1-self.prob), self.layer.bias) 
        return self.layer.call(x)





# =-------=-------=-------=------- 블럭 정의


def S_block(x, out_filters, nor=bn):
    P = nor(conv2d(x, out_filters, 1, (2, 2)))
    #     x = bn(conv2d(swish(x), out_filters, 3))
    x = nor(conv2d(swish(x), out_filters, 5, (1, 1)))
    x = avg_pool(x)
    #     x = dropout(x)
    return x + P


def S_block2(x, out_filters, nor=bn):
    P = nor(conv2d(x, out_filters, 1, (2, 2)))
    x = nor(conv2d((x), out_filters, 3))
    x = nor(conv2d(swish(x), out_filters, 3))
    #         x = dropout( x )
    x = max_pool(x)
    return x + P


def R_block(x, Cout=0, nor=bn):
    _, _, _, Cin = x.shape.as_list()
    if Cout <= 0     : Cout = Cin
    if Cout == Cin   : P = x
    else             : P = conv2d(swish(x), Cout, 1)

    x = nor(conv2d(swish(x), Cout, 3))
    x = nor(conv2d(swish(x), Cout, 3))

    return x + P


# def R_block(x, Cout=0, nor=bn):  ### depth wise 버전
#     _, _, _, Cin = x.shape.as_list()
#     if Cout <= 0     : Cout = Cin
#     if Cout == Cin   : P = x
#     else             : P = conv2d(swish(x), Cout, 1)
    
#     x = nor(conv2d(swish(x), Cout, 1))
#     x = nor(dwconv2d(swish(x), 3))
#     x = nor(conv2d(swish(x), Cout, 1))
#     x = nor(dwconv2d(swish(x), 3))
#     x = nor(conv2d(swish(x), Cout, 1))

#     return x + P

def bot_block(x, out_filters, nor=bn, t=6):
    P = nor(conv2d(x, out_filters, 1))
    x = nor(conv2d(swish(x), out_filters * t, 1))
    x = nor(dwconv2d(swish(x), 3))
    x = nor(conv2d(swish(x), out_filters, 1))
    return x + P

