from obstacle_tower_env import ObstacleTowerEnv
import argparse
import numpy as np
import cv2.cv2 as cv

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
        tf.keras.layers.Dense(input_layer, input_shape=(1, input_layer), dtype='float'),
        tf.keras.layers.Dense(2000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

def process_image(image):
        top = cv.cvtColor(image[:10], cv.COLOR_RGB2GRAY)
        image = image[10:]
        grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        canny = cv.Canny(grayscale, 180, 200)
        median = np.median(grayscale)
        mean = np.mean(grayscale)
        tp_image = cv.resize(image, (37, 42)).transpose()
        return np.concatenate((
            [mean],
            [median],
            top.flatten(),
            tp_image[0].flatten(),
            tp_image[1].flatten(),
            tp_image[2].flatten(),
            grayscale.flatten(),
            canny.flatten()
        ), axis=None)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()
    
    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, realtime_mode=True)

    model = get_model()
    optimizer = tf.train.AdamOptimizer()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint('./tf_saves/'))

    total_count = 0
    for i in range(0, 101):
        #setup environment
        env.seed(i)
        obs = env.reset()
        reward = 0
        actions = []
        rerun_actions = False
        obs = env.reset()
        while True:
            observation = process_image(obs)
            prediction = model(tf.cast([observation], dtype=tf.float32))[0]
            print('prediction', prediction)
            selection = np.argmax(prediction)
            print('selection', selection)
            action = action_options[selection]
            print('action', action)
            obs, step_reward, done, info = env.step(action)


    env.close()
