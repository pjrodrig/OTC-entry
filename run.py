from obstacle_tower_env import ObstacleTowerEnv
import sys
import argparse
import random

# none, forward, backward
# fb = [0, 18, 36]
# none, right, left
# lr = [0, 1, 2]
# none, jump
j = [0, 3]
# none, left, right
c = [0, 6, 12]

# right, left, forward, forward-right, forward-left, backward, backward-right, backward-left
direction = [1, 2, 18, 19, 20, 36, 37, 38]

def run_episode(env):
    done = False
    episode_reward = 0.0
    
    while not done:
        # pick random options
        #fbChoice = random.choice(fb)
        #lrChoice = random.choice(lr)
        directionChoice = 18 #always forward # random.choice(direction) # combination of the two above
        jChoice = 0 # random.choice(j)
        cChoice = random.choice(c)

        action = directionChoice + jChoice + cChoice

        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
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
    if env.is_grading():
        episode_reward = run_evaluation(env)
    else:
        while True:
            episode_reward = run_episode(env)
            print("Episode reward: " + str(episode_reward))
            env.reset()

    env.close()

