import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gamelogic import *
from DDQNAgent import *

EPISODES = 500

if __name__ == "__main__":
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

    scores, episodes, games = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            # get action for the current state and go one step in environment
            action = agent.get_action(np.array([state]))
            next_state, reward, done, info = env.step(action)
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_ddqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./save_model/cartpole_ddqn.h5")
