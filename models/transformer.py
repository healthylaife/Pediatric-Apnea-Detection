import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras import Model
from keras.activations import sigmoid, relu
from keras.layers import Dense, Dropout, Reshape, LayerNormalization, MultiHeadAttention, Add, Flatten, Input, Layer, \
    GlobalAveragePooling1D, AveragePooling1D, Concatenate, SeparableConvolution1D, Conv1D
from keras.regularizers import L2



class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, input):
        input = input[:, tf.newaxis, :, :]
        batch_size = tf.shape(input)[0]
        patches = tf.image.extract_patches(
            images=input,
            sizes=[1, 1, self.patch_size, 1],
            strides=[1, 1, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches,
                             [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, l2_weight):
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.l2_weight = l2_weight
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim, kernel_regularizer=L2(l2_weight),
                                bias_regularizer=L2(l2_weight))
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) # + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate, l2_weight):
    for _, units in enumerate(hidden_units):
        x = Dense(units, activation=None, kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight))(x)
        x = tf.nn.gelu(x)
        x = Dropout(dropout_rate)(x)
    return x


def create_transformer_model(input_shape, num_patches,
                             projection_dim, transformer_layers,
                             num_heads, transformer_units, mlp_head_units,
                             num_classes, drop_out, reg, l2_weight, demographic=False):
    if reg:
        activation = None
    else:
        activation = 'sigmoid'
    inputs = Input(shape=input_shape)
    patch_size = input_shape[0] / num_patches
    if demographic:
        normalized_inputs = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                                             beta_initializer="glorot_uniform",
                                                             gamma_initializer="glorot_uniform")(inputs[:,:,:-1])
        demo = inputs[:, :12, -1]

    else:
        normalized_inputs = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                                             beta_initializer="glorot_uniform",
                                                             gamma_initializer="glorot_uniform")(inputs)

    # patches = Reshape((num_patches, -1))(normalized_inputs)
    patches = Patches(patch_size=patch_size)(normalized_inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, l2_weight=l2_weight)(patches)
    for i in range(transformer_layers):
        x1 = encoded_patches # LayerNormalization(epsilon=1e-6)(encoded_patches) # TODO
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=drop_out, kernel_regularizer=L2(l2_weight),  # i *
            bias_regularizer=L2(l2_weight))(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, transformer_units, drop_out, l2_weight)  # i *
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = GlobalAveragePooling1D()(x)
    #x = Concatenate()([x, demo])
    features = mlp(x, mlp_head_units, 0.0, l2_weight)

    logits = Dense(num_classes, kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight),
                   activation=activation)(features)

    return tf.keras.Model(inputs=inputs, outputs=logits)



def create_hybrid_transformer_model(input_shape):
    transformer_units =  [32,32]
    transformer_layers = 2
    num_heads = 4
    l2_weight = 0.001
    drop_out= 0.25
    mlp_head_units = [256, 128]
    num_patches=30
    projection_dim=  32

    # Conv1D(32...
    input1 = Input(shape=input_shape)
    conv11 = Conv1D(16, 256)(input1) #13
    conv12 = Conv1D(16, 256)(input1) #13
    conv13 = Conv1D(16, 256)(input1) #13

    pwconv1 = SeparableConvolution1D(32, 1)(input1)
    pwconv2 = SeparableConvolution1D(32, 1)(pwconv1)

    conv21 = Conv1D(16, 256)(conv11) # 7
    conv22 = Conv1D(16, 256)(conv12) # 7
    conv23 = Conv1D(16, 256)(conv13) # 7

    concat = keras.layers.concatenate([conv21, conv22, conv23], axis=-1)
    concat = Dense(64, activation=relu)(concat) #192
    concat = Dense(64, activation=sigmoid)(concat) #192
    concat = SeparableConvolution1D(32,1)(concat)
    concat = keras.layers.concatenate([concat, pwconv2], axis=1)

    ####################################################################################################################
    patch_size = input_shape[0] / num_patches

    normalized_inputs = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                                             beta_initializer="glorot_uniform",
                                                             gamma_initializer="glorot_uniform")(concat)

    # patches = Reshape((num_patches, -1))(normalized_inputs)
    patches = Patches(patch_size=patch_size)(normalized_inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, l2_weight=l2_weight)(patches)
    for i in range(transformer_layers):
        x1 = encoded_patches # LayerNormalization(epsilon=1e-6)(encoded_patches) # TODO
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=drop_out, kernel_regularizer=L2(l2_weight),  # i *
            bias_regularizer=L2(l2_weight))(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, transformer_units, drop_out, l2_weight)  # i *
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = GlobalAveragePooling1D()(x)
    #x = Concatenate()([x, demo])
    features = mlp(x, mlp_head_units, 0.0, l2_weight)

    logits = Dense(1, kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight),
                   activation='sigmoid')(features)

    ####################################################################################################################

    model = Model(inputs=input1, outputs=logits)
    return model
