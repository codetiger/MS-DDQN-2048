import pylab, random, pickle, os, time, sys

import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gamelogic import *
from DDQNAgent import *
import tensorflow as tf
from supervisor import *

learn = True
EPISODES = 10000

STAGE_LIMIT = [7, 10, 14]
stage = 0
gridSize = 4
random.seed(int(time.time()))
np.random.seed(int(time.time()))

action_size = 4

agent = DoubleDQNAgent(gridSize, action_size)
agent.load_model("result/ms-ddqn-2048-s{}.h5".format(stage))
supervisor = Supervisor(gridSize)

totalScore = 0
totalMaxTile = 0

def generateGame():
    env = GameLogic(size = gridSize)
    env._normalize = True
    env._verbose = 0

    done = False
    score = 0
    randomNextMove = False
    state = env.reset()
    state = np.reshape(state, (1, gridSize, gridSize)) 
    totalSteps = 0
    randomStep = 0

    while not done:
        if randomNextMove:
            action = random.randrange(4)
            randomStep += 1
        else:
            if learn:
                action = supervisor.find_best_move_mt(env._gridMatrix)
            else:
                action = agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, (1, gridSize, gridSize)) 

        randomNextMove = reward < 0

        if learn:
            agent.append_sample(state, action, reward, next_state, done)
            agent.train_model()

        score += reward
        state = next_state
        totalSteps += 1

        global totalScore
        global totalMaxTile

        if stage < 3 and env._getMaxNumber() > STAGE_LIMIT[stage]:
            done = True
            totalMaxTile += 1

        if done:
            print("E:", e, "  score:", env._score, "  MaxTile:", 2**env._getMaxNumber(), " Randomness:", randomStep / totalSteps)
            if learn:
                agent.update_target_model()
            else:
                totalScore += score


if __name__ == "__main__":
    if len(sys.argv) > 1:
        learn = sys.argv[1] == 'True'

    if not learn:
        EPISODES = 100

    try:
        for e in range(EPISODES):
            generateGame()
    except KeyboardInterrupt:
        if learn:
            print("Saving Model")
            agent.save_model("result/ms-ddqn-2048-s{}.h5".format(stage))

    if learn:
        print("Saving Model")
        agent.save_model("result/ms-ddqn-2048-s{}.h5".format(stage))
    else:
        print("Average Score: ", totalScore / EPISODES, " Max Tile Ratio: ", totalMaxTile)
