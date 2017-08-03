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
EPISODES = 50000

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

storeEpisodeCount = 0
storeTotalSteps = 0 
storeTotalCPUTime = 0

def generateGame():
    env = GameLogic(size = gridSize)
    env._normalize = True
    env._verbose = 0

    done = False
    score = 0
    randomNextMove = False
    env.reset()
    state = env._gridMatrix
    state = np.reshape(state, (1, gridSize, gridSize)) 
    totalSteps = 0
    randomStep = 0

    global storeEpisodeCount
    global storeTotalSteps
    storeEpisodeCount += 1

    global totalScore
    global totalMaxTile

    while not done:
        if randomNextMove:
            action = random.randrange(4)
            randomStep += 1
        else:
            if learn:
                action = supervisor.find_best_move_mt(env._gridMatrix)
            else:
                action = agent.get_action(state)

        storeTotalSteps += 1

        next_state, reward, done, info = env.step(action)
        next_state = env._gridMatrix
        next_state = np.reshape(next_state, (1, gridSize, gridSize)) 

        randomNextMove = reward < 0

        if learn:
            agent.append_sample(state, action, reward, next_state, done)
            agent.train_model()

        score += reward
        state = next_state
        totalSteps += 1

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
    start_time = time.time()

    if os.path.isfile("result/ms-ddqn-2048-s{}.txt".format(stage)):
        with open("result/ms-ddqn-2048-s{}.txt".format(stage), 'r') as f:
            val = [x for x in next(f).split()]
            storeEpisodeCount, storeTotalSteps, storeTotalCPUTime = int(val[0]), int(val[1]), float(val[2])
    else:
        storeEpisodeCount, storeTotalSteps, storeTotalCPUTime = 0, 0, 0

    if len(sys.argv) > 1:
        learn = sys.argv[1] == 'True'

    if not learn:
        EPISODES = 100

    try:
        for e in range(EPISODES):
            generateGame()
    except KeyboardInterrupt:
        pass

    storeTotalCPUTime += time.time() - start_time
    
    if learn:
        print("Saving Model")
        agent.save_model("result/ms-ddqn-2048-s{}.h5".format(stage))
        with open("result/ms-ddqn-2048-s{}.txt".format(stage), 'w') as f:
            f.write('{} {} {}\n'.format(storeEpisodeCount, storeTotalSteps, storeTotalCPUTime))
    else:
        print("Average Score: ", totalScore / EPISODES, " Max Tile Ratio: ", totalMaxTile)
