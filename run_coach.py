from obstacle_tower_env import ObstacleTowerEnv
import sys
import argparse
import random
import numpy as np

from utils import *

network = None

# none, right, left, forward, forward-right, forward-left, backward, backward-right, backward-left
direction = [0, 1, 2, 18, 19, 20, 36, 37, 38]
# none, jump
j = [0, 3]
# none, left, right
c = [0, 6, 12]

# forward, cam left + forward, cam right + forward
action_options = [18, 24, 30]

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
        reward_total = 0
        while reward_total < 2:
            reward = 0
            if(i_path < len(seed_paths)):
                reward_before = reward_total
                while i_path < len(seed_paths) and reward_total < reward_before + 1:
                    current_path = paths[int(seed_paths[int(i_path)])]
                    while path_i < len(current_path) and reward_total < reward_before + 1:
                        current_action = int(current_path[path_i])
                        obs, reward, done, info = env.step(current_action)
                        reward_total += reward
                        print('loop reward', reward)
                        path_i += 1
                    i_path += 1
                    path_i = 0
            print('before if reward', reward)
            if(reward == 0):
                if rerun_actions:
                    for action in actions:
                        env.step(action)
                    rerun_actions = False
                else:
                    print("left: 1, right: 2")
                    action = input("action: ")
                    if action == "restart":
                        i_path = 0 #paths iterator
                        path_i = 0 #path iterator
                        env.seed(i)
                        env.reset()
                        env.step(18)
                        actions = []
                        reward_total = 0
                    elif action == "undo":
                        i_path = 0 #paths iterator
                        path_i = 0 #path iterator
                        env.seed(i)
                        env.reset()
                        env.step(18)
                        actions.pop()
                        reward_total = 0
                        rerun_actions = True
                    elif action == "path":
                        for pi, path in enumerate(paths, start=0):
                            print(pi, path)
                        path_selection = input("path: ")
                        if path_selection != "cancel":
                            current_path = paths[int(path_selection)]
                            seed_paths.append(int(path_selection))
                            i_path = 0 #paths iterator
                            path_i = 0 #path iterator
                            env.seed(i)
                            env.reset()
                            actions = []
                            reward_total = 0
                    elif action == '' or action == '1' or action == '2':
                        if action == '':
                            action = '0'
                        action = action_options[int(action)]
                        actions.append(action)
                        obs, reward, done, info = env.step(action)
                        reward_total += reward
                        print("reward", reward_total)
        if(len(actions) > 0):
            seed_paths.append(len(paths))
            paths.append(actions)
            save_file = open('./save/save.txt', 'w')
            to_write = ''
            for path in paths:
                to_write += ' '.join([str(action) for action in path]) + '\n'

            for seed_paths_to_save in seed_paths_arr:
                to_write += '\n' + ' '.join([str(seed_path_to_save) for seed_path_to_save in seed_paths_to_save])
            save_file.write(to_write)
        env.reset()


    env.close()
