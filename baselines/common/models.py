import numpy as np
import tensorflow as tf
from baselines.a2c.utils import ortho_init, conv

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def nature_cnn(input_shape, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    print('input shape is {}'.format(input_shape))
    x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
    h = x_input
    h = tf.cast(h, tf.float32) / 255.
    h = conv('c1', nf=32, rf=8, stride=4, activation='relu', init_scale=np.sqrt(2))(h)
    h2 = conv('c2', nf=64, rf=4, stride=2, activation='relu', init_scale=np.sqrt(2))(h)
    h3 = conv('c3', nf=64, rf=3, stride=1, activation='relu', init_scale=np.sqrt(2))(h2)
    h3 = tf.keras.layers.Flatten()(h3)
    h3 = tf.keras.layers.Dense(units=512, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='fc1', activation='relu')(h3)
    network = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return network

@register("mlp2_64")
def mlp2_64(num_layers=2, num_hidden=64, activation=tf.tanh):
    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        #h = tf.keras.layers.Flatten()(x_input)
        h = x_input
        for i in range(num_layers):
            h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc{}'.format(i), activation=activation)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("mlp2_256")
def mlp2_256(num_layers=2, num_hidden=256, activation=tf.tanh):
    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        # h = tf.keras.layers.Flatten(x_input)
        h = x_input
        for i in range(num_layers):
            h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc{}'.format(i), activation=activation)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("mlp2_1024")
def mlp2_1024(num_layers=2, num_hidden=1024, activation=tf.tanh):
    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        # h = tf.keras.layers.Flatten(x_input)
        h = x_input
        for i in range(num_layers):
            h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc{}'.format(i), activation=activation)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("mlp3_64")
def mlp3_64(num_layers=3, num_hidden=64, activation=tf.tanh):
    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        # h = tf.keras.layers.Flatten(x_input)
        h = x_input
        for i in range(num_layers):
            h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc{}'.format(i), activation=activation)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("mlp3_256")
def mlp3_256(num_layers=3, num_hidden=256, activation=tf.tanh):
    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        # h = tf.keras.layers.Flatten(x_input)
        h = x_input
        for i in range(num_layers):
            h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc{}'.format(i), activation=activation)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("mlp3_1024")
def mlp3_1024(num_layers=3, num_hidden=1024, activation=tf.tanh):
    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        # h = tf.keras.layers.Flatten(x_input)
        h = x_input
        for i in range(num_layers):
            h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc{}'.format(i), activation=activation)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("cnn_mlp_64")
def cnn4small(nh=64, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=4, activation='relu')(x_input)
        h = tf.keras.layers.Flatten()(h)
        h = tf.keras.layers.Dense(units=nh, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc0', activation=tf.tanh)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("cnn_mlp_256")
def cnn4medium(nh=256, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=4, activation='relu')(x_input)
        h = tf.keras.layers.Flatten()(h)
        h = tf.keras.layers.Dense(units=nh, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc0', activation=tf.tanh)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("cnn_mlp_1024")
def cnn4medium(nh=1024, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=4, activation='relu')(x_input)
        h = tf.keras.layers.Flatten()(h)
        h = tf.keras.layers.Dense(units=nh, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc0', activation=tf.tanh)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("cnn2_64")
def cnn2_64(nh=64, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=2, activation='relu')(x_input)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=2, activation='relu')(h)
        h = tf.keras.layers.Flatten()(h)
        #h = tf.keras.layers.Dense(units=nh, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc0', activation=tf.tanh)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("cnn2_256")
def cnn2_256(nh=256, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=2, activation='relu')(x_input)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=2, activation='relu')(h)
        h = tf.keras.layers.Flatten()(h)
        #h = tf.keras.layers.Dense(units=nh, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc0', activation=tf.tanh)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("cnn2_1024")
def cnn2_1024(nh=1024, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=2, activation='relu')(x_input)
        h = tf.keras.layers.Conv1D(filters=nh, kernel_size=2, activation='relu')(h)
        h = tf.keras.layers.Flatten()(h)
        #h = tf.keras.layers.Dense(units=nh, kernel_initializer=ortho_init(np.sqrt(2)), name='mlp_fc0', activation=tf.tanh)(h)
        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn

@register("lstm1_64")
def lstm1_64(nh=64, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True)(x_input)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("lstm1_256")
def lstm1_256(nh=256, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True)(x_input)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("lstm1_1024")
def lstm1_1024(nh=1024, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True)(x_input)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("lstm2_64")
def lstm2_64(nh=64, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True, return_sequences=True)(x_input)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True)(out)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("lstm2_256")
def lstm2_256(nh=256, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True, return_sequences=True)(x_input)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True)(out)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("lstm2_1024")
def lstm2_1024(nh=1024, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True, return_sequences=True)(x_input)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True)(out)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("alstm_64")
def alstm_64(nh=64, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True, return_sequences=True)(x_input)
        ht = tf.expand_dims(h, 1)
        score = tf.nn.tanh(tf.keras.layers.Dense(nh)(out) + tf.keras.layers.Dense(nh)(ht))
        attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(score), axis=1)
        out = attention_weights * out
        out = tf.reduce_sum(out, axis=1)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("alstm_256")
def alstm_256(nh=256, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True, return_sequences=True)(x_input)
        ht = tf.expand_dims(h, 1)
        score = tf.nn.tanh(tf.keras.layers.Dense(nh)(out) + tf.keras.layers.Dense(nh)(ht))
        attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(score), axis=1)
        out = attention_weights * out
        out = tf.reduce_sum(out, axis=1)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("blstm_64")
def blstm_64(nh=64, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, fh, fc, bh, bc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nh, return_state=True))(x_input)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("blstm_256")
def blstm_256(nh=256, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, fh, fc, bh, bc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nh, return_state=True))(x_input)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("ablstm_64")
def ablstm_64(nh=64, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, fh, fc, bh, bc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nh, return_state=True, return_sequences=True))(x_input)
        h = tf.keras.layers.Concatenate()([fh, bh])
        ht = tf.expand_dims(h, 1)
        score = tf.nn.tanh(tf.keras.layers.Dense(nh)(out) + tf.keras.layers.Dense(nh)(ht))
        attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(score), axis=1)
        out = attention_weights * out
        out = tf.reduce_sum(out, axis=1)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("ablstm_256")
def ablstm_256(nh=256, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        out, fh, fc, bh, bc = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(nh, return_state=True, return_sequences=True))(x_input)
        h = tf.keras.layers.Concatenate()([fh, bh])
        ht = tf.expand_dims(h, 1)
        score = tf.nn.tanh(tf.keras.layers.Dense(nh)(out) + tf.keras.layers.Dense(nh)(ht))
        attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(score), axis=1)
        out = attention_weights * out
        out = tf.reduce_sum(out, axis=1)
        network = tf.keras.Model(inputs=[x_input], outputs=[out])
        return network
    return network_fn

@register("slstm_64")
def slstm_64(nh=64, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        x_reshaped = tf.keras.layers.Reshape((1, *input_shape))(x_input)
        state_h = tf.keras.Input(shape=(nh,))
        state_c = tf.keras.Input(shape=(nh,))
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True)(x_reshaped, initial_state=[state_h, state_c])
        network = tf.keras.Model(inputs=[x_input, state_h, state_c], outputs=[out, h, c])
        return network
    return network_fn

@register("slstm_256")
def slstm_256(nh=256, **conv_kwargs):
    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape)
        x_reshaped = tf.keras.layers.Reshape((1, *input_shape))(x_input)
        state_h = tf.keras.Input(shape=(nh,))
        state_c = tf.keras.Input(shape=(nh,))
        out, h, c = tf.keras.layers.LSTM(nh, return_state=True)(x_reshaped, initial_state=[state_h, state_c])
        network = tf.keras.Model(inputs=[x_input, state_h, state_c], outputs=[out, h, c])
        return network
    return network_fn

@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(input_shape):
        return nature_cnn(input_shape, **conv_kwargs)
    return network_fn


@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net
    Parameters:
    ----------
    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.
    Returns:
    function that takes tensorflow tensor as input and returns the output of the last convolutional layer
    '''

    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        h = x_input
        h = tf.cast(h, tf.float32) / 255.
        with tf.name_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                h = tf.keras.layers.Conv2D(
                    filters=num_outputs, kernel_size=kernel_size, strides=stride,
                    activation='relu', **conv_kwargs)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
