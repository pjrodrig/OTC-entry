from obstacle_tower_env import ObstacleTowerEnv
import sys
import argparse
import random
import numpy as np

from network import Network
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
    obs = env.reset()
    steps = []
    episode_bias_deltas = np.array([])
    episode_weight_deltas = np.array([])
    step_count = 0

    while not done:
        observations = []
        for y in range(0, 84):
            obsy = obs[y]
            for x in range(0, 84):
                obsx = obsy[x]
                grayscale = 0
                for val in range(0, 3):
                        grayscale += obsx[val]
                observations.append([grayscale/3])
        probabilities, activations, connections = network.evaluate(np.array(observations))

        randomPick = random.random() * sum(probabilities)
        for i in range(0, action_size):
            if(randomPick < probabilities[i]):
                action = actions[i]
                break
            else:
                randomPick = randomPick - probabilities[i]

        obs, reward, done, info = env.step(action)
        episode_bias_deltas = np.array([np.zeros(bias.shape) for bias in network.biases])
        episode_weight_deltas = np.array([np.zeros(weight.shape) for weight in network.weights])
        step = {
            'activations': activations, 
            'connections': connections, 
            'probabilities': probabilities, 
            'choice': i, 
            'bias_deltas': episode_bias_deltas, 
            'weight_deltas': episode_weight_deltas
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
                
                delta = costs * sigmoid_prime(step['connections'][-1])
                step['bias_deltas'][-1] = delta
                step['weight_deltas'][-1] = np.dot(delta, step['activations'][-2].transpose())
                for i in range(2, len(network.layers)):
                    connections = step['connections'][-i]
                    sp = sigmoid_prime(connections)
                    delta = np.dot(network.weights[-i+1].transpose(), delta) * sp
                    step['bias_deltas'][-i] = delta
                    step['weight_deltas'][-i] = np.dot(delta, activations[-i-1].transpose())
                episode_bias_deltas = episode_bias_deltas + step['bias_deltas']
                episode_weight_deltas = episode_weight_deltas + step['weight_deltas']
            steps = []
        step_count += 1
    episode_bias_deltas = episode_bias_deltas / step_count
    episode_weight_deltas = episode_weight_deltas / step_count
    return episode_reward, episode_bias_deltas, episode_weight_deltas

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

    layers = [7056, 500, 500, 3]
    eta = 0.5
    action_size = 3
    gamma = 0.5

    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, realtime_mode=True)
    network = Network(layers)
    if env.is_grading():
        episode_reward = run_evaluation(env)
    else:
        bias_deltas = np.array([np.zeros(bias.shape) for bias in network.biases])
        weight_deltas = np.array([np.zeros(weight.shape) for weight in network.weights])
        while True:
            episode_reward, episode_bias_deltas, episode_weight_deltas = run_episode(env)
            if(episode_reward > 0):
                bias_deltas = bias_deltas + episode_bias_deltas
                weight_deltas = weight_deltas + episode_weight_deltas
            print("Episode reward: " + str(episode_reward))
            network.biases = network.biases - (bias_deltas * eta)
            network.weights = network.weights - (weight_deltas * eta)

    env.close()