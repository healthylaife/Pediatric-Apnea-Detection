import keras
from keras import Input, Model
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, LSTM, Bidirectional, Permute, \
    Reshape, GRU, Conv1D, MaxPooling1D, Activation, Dropout, GlobalAveragePooling1D, multiply, MultiHeadAttention, Add, \
    LayerNormalization, SeparableConvolution1D
from keras.models import Sequential
from keras.activations import relu, sigmoid
from keras.regularizers import l2
import tensorflow_addons as tfa
from .transformer import create_transformer_model, mlp, create_hybrid_transformer_model




def create_cnn_model(input_shape):
    model = Sequential()
    for i in range(5): # 10
        model.add(Conv1D(45, 32, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation(relu))
        model.add(MaxPooling1D())
        model.add(Dropout(0.5))

    model.add(Flatten())
    for i in range(2): #4
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation(relu))
        model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model


def create_cnnlstm_model(input_a_shape, weight=1e-3):
    cnn_filters = 32 # 128
    cnn_kernel_size = 4 # 4
    input1 = Input(shape=input_a_shape)
    input1 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                              beta_initializer="glorot_uniform",
                                              gamma_initializer="glorot_uniform")(input1)
    x1 = Conv1D(cnn_filters, cnn_kernel_size, activation='relu')(input1)
    x1 = Conv1D(cnn_filters, cnn_kernel_size, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D()(x1)

    x1 = Conv1D(cnn_filters, cnn_kernel_size, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D()(x1)

    x1 = Conv1D(cnn_filters, cnn_kernel_size, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D()(x1)

    x1 = LSTM(32, return_sequences=True)(x1) #256
    x1 = LSTM(32, return_sequences=True)(x1) #256
    x1 = LSTM(32)(x1) #256
    x1 = Flatten()(x1)

    x1 = Dense(32, activation='relu')(x1) #64
    x1 = Dense(32, activation='relu')(x1) #64
    outputs = Dense(1, activation='sigmoid')(x1)

    model = Model(inputs=input1, outputs=outputs)
    return model


def create_semscnn_model(input_a_shape):
    input1 = Input(shape=input_a_shape)
    # input1 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
    #                                           beta_initializer="glorot_uniform",
    #                                           gamma_initializer="glorot_uniform")(input1)
    x1 = Conv1D(45, 32, strides=1)(input1) #kernel_size=11
    x1 = Conv1D(45, 32, strides=2)(x1) #64 kernel_size=11
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D()(x1)

    x1 = Conv1D(45, 32, strides=2)(x1) #64 kernel_size=11
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D()(x1)

    x1 = Conv1D(45, 32, strides=2)(x1) #64 kernel_size=11
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)
    x1 = MaxPooling1D()(x1)

    squeeze = Flatten()(x1)
    excitation = Dense(128, activation='relu')(squeeze)
    excitation = Dense(64, activation='relu')(excitation)
    logits = Dense(1, activation='sigmoid')(excitation)
    model = Model(inputs=input1, outputs=logits)
    return model


model_dict = {

    "cnn": create_cnn_model((60 * 32, 3)),
    "sem-mscnn": create_semscnn_model((60 * 32, 3)),
    "cnn-lstm": create_cnnlstm_model((60 * 32, 3)),
    "hybrid": create_hybrid_transformer_model((60 * 32, 3)),
}


def get_model(config):
    if config["model_name"].split('_')[0] == "Transformer":
        return create_transformer_model(input_shape=(60 * 32, len(config["channels"])),
                                        num_patches=config["num_patches"], projection_dim=config["transformer_units"],
                                        transformer_layers=config["transformer_layers"], num_heads=config["num_heads"],
                                        transformer_units=[config["transformer_units"] * 2,
                                                           config["transformer_units"]],
                                        mlp_head_units=[256, 128], num_classes=1, drop_out=config["drop_out_rate"],
                                        reg=config["regression"], l2_weight=config["regularization_weight"])
    else:
        return model_dict.get(config["model_name"].split('_')[0])


if __name__ == "__main__":
    config = {
        "model_name": "hybrid",
        "regression": False,

        "transformer_layers": 4,  # best 5
        "drop_out_rate": 0.25,
        "num_patches": 20,  # best
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 100,  # best
        "channels": [14, 18, 19, 20],
    }
    model = get_model(config)
    model.build(input_shape=(1, 60 * 32, 10))
    print(model.summary())
