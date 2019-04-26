import numpy as np
import random
import tensorflow as tf

# forward, cam left + forward, cam right + forward
action_options = [18, 24, 30]

action_map = {
    '18': 0,
    '24': 1,
    '30': 2
}


def get_model():
    tf.enable_eager_execution()
    input_layer = 17936
    return tf.keras.Sequential([
        tf.keras.layers.Dense(input_layer, input_shape=(1, input_layer), dtype='double'),
        tf.keras.layers.Dense(2000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])



if __name__ == '__main__':
    #Load data
    x = []
    y = []
    for i in range(0, 101):
        x_data = open('./data/x_data_' + str(i), 'r').read()
        for x_line in x_data.strip().split('\n'):
            x.append(x_line.strip().split(' '))
        y_data = open('./data/y_data_' + str(i), 'r').read()
        for yi in y_data.strip().split(' '):
            y.append(int(yi))

    y_probs = np.zeros((len(y), 3))
    for yi, y_prob in zip(y, y_probs):
        y_prob[yi] = 1
    y = y_probs

    data = tf.data.Dataset.from_tensor_slices((tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
    data = data.shuffle(1000).batch(32)

    #Train
    model = get_model()

    optimizer = tf.train.AdamOptimizer()

    loss_history = []
    for (batch, (images, labels)) in enumerate(data.take(500)):
        if batch % 10 == 0:
            print('.', end='')
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        loss_history.append(loss_value.numpy())
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())

