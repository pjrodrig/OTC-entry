from obstacle_tower_env import ObstacleTowerEnv
import sys
import argparse
import random
import numpy as np
import tensorflow as tf #gpu
from tensorflow import keras

network = None

# none, right, left, forward, forward-right, forward-left, backward, backward-right, backward-left
direction = [0, 1, 2, 18, 19, 20, 36, 37, 38]
# none, jump
j = [0, 3]
# none, left, right
c = [0, 6, 12]

def get_model():
    tf.enable_eager_execution()
    return tf.keras.Sequential([
        tf.keras.layers.Dense(7056, input_shape=(1, 7056), dtype='double'),
        tf.keras.layers.Dense(500, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(500, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])


# forward, cam left + forward, cam right + forward
actions = [18, 24, 30]
def run_episode(env):
    done = False
    episode_reward = 0.0
    obs = env.reset()
    steps = []
    step_count = 0
    action_size = 3

    memory = []

    while not done:
        observations = []
        for y in range(0, 84):
            obsy = obs[y]
            for x in range(0, 84):
                obsx = obsy[x]
                grayscale = 0
                for val in range(0, 3):
                        grayscale += obsx[val]
                observations.append(grayscale/3)
                
        probabilities = net(np.array([observations]), training=True)[0]
        print(probabilities)
        randomPick = random.random() * sum(probabilities)
        for i in range(0, action_size):
            if(randomPick < probabilities[i]):
                action = actions[i]
                break
            else:
                randomPick = randomPick - probabilities[i]
        obs, reward, done, info = env.step(action)

        step = {
            'probabilities': probabilities, 
            'choice': i
        }
        steps.append(step)
        episode_reward += reward
        if(reward > 0):
            for step in reversed(steps):
                costs = []
                for i in range(0, len(step['probabilities'])):
                    if(i == step['choice']):
                        costs.append(((step['probabilities'][i] - 1)**2)/2)
                    else:
                        costs.append(((step['probabilities'][i])**2)/2)
            
            steps = []
    return episode_reward

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
    net = get_model()
    optimizer = tf.train.AdamOptimizer()
    if env.is_grading():
        episode_reward = run_evaluation(env)
    else:
        while True:
            episode_reward = run_episode(env)
            print("Episode reward: " + str(episode_reward))

    env.close()