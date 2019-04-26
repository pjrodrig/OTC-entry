import numpy as np
import random
import tensorflow as tf
import shutil

tf.enable_eager_execution()

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
        tf.keras.layers.Dense(input_layer, input_shape=(1, input_layer), dtype='float'),
        tf.keras.layers.Dense(2000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])



if __name__ == '__main__':
    #Load data
    print("Loading data")
    x = []
    y = []
    for i in range(0, 101):
        x_data = open('./data/x_data_' + str(i), 'r').read()
        for x_line in x_data.strip().split('\n'):
            x.append([float(xi) for xi in x_line.strip().split(' ')])
        y_data = open('./data/y_data_' + str(i), 'r').read()
        for yi in y_data.strip().split(' '):
            y.append(int(yi))

    #Train

    model = get_model()
    optimizer = tf.optimizers.GradientDescentOptimizer(learning_rate=0.8)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint('./tf_saves/'))

    while True:
        print("Creating batches")
        data = tf.data.Dataset.from_tensor_slices((x, y))
        data = data.shuffle(1000).batch(32)

        

        print("Training model")
        loss_history = []
        for (batch, (images, labels)) in enumerate(data.take(500)):
            if batch % 10 == 0:
                print('.', end='')
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
                print(loss_value)
            loss_history.append(loss_value.numpy())
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
        
        shutil.rmtree('./tf_saves/')
        checkpoint.save('./tf_saves/')