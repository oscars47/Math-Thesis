# file to run sphynx architecture
# input is 2^N len vector of angles of eigenvalues sorted smallest to largest
# output is a circuit of length N2*max_depth

'''sketch of architecture:

input vec of 2^N angles -> fully connected neural network, end with N nodes -> each of these nodes are mapped to a separate series of layers ending with g nodes, where g is number of gates in gate set -> choose gate for that mode

'''

import tensorflow as tf
import numpy as np
import os

def create_branch_network(input_layer, num_gates, name):
    ''' branch network for one position'''
    x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(num_gates, activation='softmax', name=name)(x)
    return output

def create_model(input_size, num_branches, num_gates):
    '''Initiates the model'''
    inputs = tf.keras.Input(shape=(input_size,))

    # base network
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # branch networks for each entry in the output
    outputs = [create_branch_network(x, num_gates, f'gate_output_{i}') for i in range(num_branches)]

    # create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    ## --------- create and compile the model --------- ##
    N2 = 10
    num_gates = 8
    model_name = 'v0'
    model = create_model(2**N2, num_branches = N2, num_gates=num_gates)
    model.compile(
        optimizer='adam', 
        loss=['categorical_crossentropy' for _ in range(N2)], 
        metrics=['accuracy']
    )

    ## --------- load data --------- ##
    # load dataset
    x = np.load('data/x_10_100_100000.npy')
    y = np.load('data/y_10_100_100000.npy')

    # split into train and validation and test
    # train: 80%, validation: 10%, test: 10%
    len_data = len(x)
    train_split = int(0.8 * len_data)
    val_split = int(0.9 * len_data)

    x_train = x[:train_split]
    y_train = y[:train_split]
    x_val = x[train_split:val_split]
    y_val = y[train_split:val_split]
    x_test = x[val_split:]
    y_test = y[val_split:]

    # split y into branches
    y_train_branches = [y_train[:, i, :] for i in range(N2)]
    y_val_branches = [y_val[:, i, :] for i in range(N2)]
    y_test_branches = [y_test[:, i, :] for i in range(N2)]

    ## --------- train the model --------- ##
    history = model.fit(
        x_train, 
        y_train_branches, 
        epochs=10,
        batch_size=32,
        validation_data=(x_val, y_val_branches)
    )
    # save model #
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(os.path.join('models', f'model_{model_name}.h5'), save_format='h5')