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
        tf.keras.layers.Dense(input_layer, input_shape=(1, input_layer), dtype='double'),
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

    # model = get_model()

    paths = []
    seed_paths_arr = []
    save_file = open('./save/save.txt', 'r')
    save_data = save_file.read()
    reading_seeds = False
    for line in save_data.split('\n'):
        if line == '':
            reading_seeds = True
        elif reading_seeds:
            seed_paths_arr.append(line.split(' '))
        else:
            paths.append(line.split(' '))

    total_count = 0
    for i in range(0, 101):
        #get paths for this seed
        if len(seed_paths_arr) > i:
            seed_paths = seed_paths_arr[i]
        else:
            seed_paths = []
            seed_paths_arr.append(seed_paths)
        i_path = 0 #paths iterator
        path_i = 0 #path iterator

        x = []
        y = []
        xcount = 0
        ycount = 0
        #setup environment
        env.seed(i)
        obs = env.reset()
        reward = 0
        actions = []
        rerun_actions = False
        while reward < 1:
            if(i_path < len(seed_paths)):
                while i_path < len(seed_paths):
                    current_path = paths[int(seed_paths[int(i_path)])]
                    while path_i < len(current_path):
                        current_action = int(current_path[path_i])
                        observation = process_image(obs)
                        total_count += 1
                        x.append([str(ob) for ob in observation])
                        y.append(action_map[str(current_action)])
                        obs, step_reward, done, info = env.step(current_action)
                        reward += step_reward
                        path_i += 1
                    i_path += 1
                    path_i = 0

        print("x", np.array(x).shape)
        print("y", np.array(y).shape)
        x_data = open('./data/x_data_' + str(i), 'w')
        x_data = open('./data/x_data_' + str(i), 'a')
        for xi in x:
            x_data.write(' '.join(xi) + '\n')
        y_data = open('./data/y_data_' + str(i), 'w')
        y_data = open('./data/y_data_' + str(i), 'a')
        for yi in y:
            y_data.write(str(yi) + ' ')


    env.close()
