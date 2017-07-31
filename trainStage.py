import pylab
import random, pickle, os, time, sys
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gamelogic import *
from DDQNAgent import *

EPISODES = 50000
STAGE_LIMIT = [7, 10, 14]

if __name__ == "__main__":
    stage = 0
    gridSize = 4
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    env = GameLogic(size = gridSize)
    env._normalize = True
    env.reset()

    # get size of state and action from environment
    state_size = gridSize * gridSize
    action_size = 4

    agent = DoubleDQNAgent(state_size, action_size)
    agent.load_model("result/ms-ddqn-2048-s{}.h5".format(stage))

    # games = []
    # try:
    #     with open("result/model-data-s{}.dat".format(stage), "rb") as pickle_file:
    #         games = pickle.load(pickle_file)
    # except:
    #     pass

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, (1, state_size)) 

        while not done:
            try:
                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, (1, state_size)) 

                # if an action make the episode end, then gives penalty of -100
                reward = reward if not done or score == 499 else -100

                # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state, action, reward, next_state, done)
                # every time step do the training
                agent.train_model()
                score += reward
                state = next_state

                if stage < 3 and env._getMaxNumber() > STAGE_LIMIT[stage]:
                    done = True
                    # games.append((env._score, env._gridMatrix))
                    # pickle.dump(games, open("result/model-data-s{}.dat".format(stage), "wb"), protocol=2)

                if done:
                    # every episode update the target model to be same with model
                    agent.update_target_model()
                    # env._printGrid()
                    print("E:", e, "  score:", env._score, "  MaxTile:", 2**env._getMaxNumber(), "  epsilon:", agent.epsilon)
            except:
                print("Saving Model")
                agent.save_model("result/ms-ddqn-2048-s{}.h5".format(stage))
                sys.exit()

    print("Saving Model")
    agent.save_model("result/ms-ddqn-2048-s{}.h5".format(stage))
