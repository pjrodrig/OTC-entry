from obstacle_tower_env import ObstacleTowerEnv
import numpy as np
import cv2 as cv

import tensorflow as tf

# forward, cam left + forward, cam right + forward
action_options = [18, 24, 30]

memory = 25
stats = 2
top = 840
images = 37632

def get_model():
    tf.enable_eager_execution()
    input_layer = memory + stats + top + images
    return tf.keras.Sequential([
        tf.keras.layers.Dense(input_layer, input_shape=(1, input_layer), dtype='double'),
        tf.keras.layers.Dense(2000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(2000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

def process_image(image):
        top = image[:10]
        image = image[10:]
        grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        canny1 = cv.Canny(grayscale, 0, 100)
        canny2 = cv.Canny(grayscale, 130, 150)
        canny3 = cv.Canny(grayscale, 180, 200)
        median = np.median(grayscale)
        mean = np.mean(grayscale)
        # cv.imwrite('./img/top.jpg', top)
        # cv.imwrite('./img/image.jpg', image)
        # cv.imwrite('./img/gray.jpg', grayscale)
        # cv.imwrite('./img/canny1.jpg', canny1)
        # cv.imwrite('./img/canny2.jpg', canny2)
        # cv.imwrite('./img/canny3.jpg', canny3)
        # cv.imwrite('./img/cannyMean.jpg', cmean)
        # cv.imwrite('./img/cannyMedian.jpg', cmedian)
        tp_image = image.transpose()
        return top.flatten(), mean, median, [
            tp_image[0],
            tp_image[1],
            tp_image[2],
            grayscale,
            canny1,
            canny2,
            canny3
        ]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()
    
    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, realtime_mode=True)

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

    print('paths', paths)
    print(seed_paths_arr)
    for i in range(0, 101):
        #get paths for this seed
        if len(seed_paths_arr) > i:
            seed_paths = seed_paths_arr[i]
        else:
            seed_paths = []
            seed_paths_arr.append(seed_paths)
        i_path = 0 #paths iterator
        path_i = 0 #path iterator

        #setup environment
        env.seed(i)
        env.reset()
        reward = 0
        actions = []
        rerun_actions = False
        print("starting here")
        while reward < 1:
            if(i_path < len(seed_paths)):
                while i_path < len(seed_paths):
                    current_path = paths[int(seed_paths[int(i_path)])]
                    while path_i < len(current_path):
                        current_action = int(current_path[path_i])
                        obs, step_reward, done, info = env.step(current_action)
                        reward += step_reward
                        path_i += 1
                    i_path += 1
                    path_i = 0


    env.close()
