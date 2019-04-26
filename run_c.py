from obstacle_tower_env import ObstacleTowerEnv
import sys
import argparse
import random
import numpy as np

from test import Test

from utils import *

network = None

# none, right, left, forward, forward-right, forward-left, backward, backward-right, backward-left
direction = [0, 1, 2, 18, 19, 20, 36, 37, 38]
# none, jump
j = [0, 3]
# none, left, right
c = [0, 6, 12]

# forward, cam left + forward, cam right + forward
actions = [18, 24, 30]
def run_episode(env):
    done = False
    episode_reward = 0.0
    env.seed(3)
    obs = env.reset()
    steps = []
    episode_bias_deltas = np.array([])
    episode_weight_deltas = np.array([])
    step_count = 0
    
    while not done:
        choices = test.eval(obs)
        obs, reward, done, info = env.step(actions[random.choice(list(choices))])
        
    return 1

def run_evaluation(env):
    while not env.done_grading():
        run_episode(env)
        env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()
    
    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, realtime_mode=True)



    test = Test(84 * 84, 3)


    if env.is_grading():
        episode_reward = run_evaluation(env)
    else:
        while True:
            episode_reward = run_episode(env)
            print("Episode reward: " + str(episode_reward))

    env.close()
