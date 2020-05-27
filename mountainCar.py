import matplotlib.pyplot as plt
import gym
import numpy as np
import random
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from collections import deque, namedtuple


# Parameters
NUM_STEPS = 200
NUM_EPISODES = 1000
LEN_EPISODE = 200
GAMMA  = 0.995
LEARNING_RATE = 0.0005
MEMORY_SIZE = 1000000
BATCH_SIZE = 64
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
DECAY_RATE = 0.99995

reward_history = []
position_history = []
step_history = []

np.random.seed(10)
random.seed(10)

class MountainCar:

    def __init__(self, total_input, total_output):
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON_MAX
        self.decay_rate = DECAY_RATE
        self.model = self.createModel(total_input, total_output)

        self.batch = namedtuple("Batch", "state action reward next_state done")
        self.batches = deque(maxlen = MEMORY_SIZE)

        self.total_input = total_input
        self.total_output = total_output

    def createModel(self, total_input, total_output):
        model  = Sequential()
        model.add(Dense(24, input_shape=(total_input,), activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(total_output, activation="linear"))
        model.compile(loss="mse", optimizer = Adam(lr = LEARNING_RATE))
        return model

    def store_values(self, state, action, reward, next_state, done):
        self.batches.append(self.batch(state, action, reward, next_state, (1-done)))

    def getAction(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.total_output)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def updateQValues(self):
        if len(self.batches) < BATCH_SIZE:
            return

        batch = random.sample(self.batches, BATCH_SIZE)
        batch_index = np.arange(BATCH_SIZE, dtype=np.int32)

        states = np.vstack([i.state for i in batch if i is not None])
        actions = np.vstack([i.action for i in batch if i is not None])
        rewards = np.vstack([i.reward for i in batch if i is not None])
        next_states = np.vstack([i.next_state for i in batch if i is not None])
        dones = np.vstack([i.done for i in batch if i is not None])
        oldQ = self.model.predict(states)
        newQ = self.model.predict(next_states)
        q_values = oldQ.copy()
        tempQ = rewards + self.gamma * \
                np.amax(newQ, axis=1).reshape(BATCH_SIZE,1)*dones
        q_values[batch_index, actions[:,0]] = tempQ[:,0]
        self.model.fit(states, q_values, verbose = 0)

    def updateEpsilon(self):
        if self.epsilon *self.decay_rate < EPSILON_MIN:
            self.epsilon = EPSILON_MIN
        else:
            self.epsilon = self.epsilon * self.decay_rate
        return 0

def main():
    env = gym.make('MountainCar-v0')
    env.seed(10)
    total_input = env.observation_space.shape[0]
    total_output = env.action_space.n
    agent = MountainCar(total_input, total_output)
    moving_avg_position = []
    moving_avg_reward = []
    moving_avg_steps = []

    # Run for NUM_EPISODES
    for episode in range(NUM_EPISODES):
        episode_reward = 0
        curr_state = env.reset()
        curr_state = np.reshape(curr_state, [1, total_input])

        for step in range(LEN_EPISODE):
            env.render()
            action = agent.getAction(curr_state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, total_input])

            # Update the policy
            agent.store_values(curr_state, action, reward, next_state, done)
            agent.updateQValues()

            # Record history
            episode_reward += reward

            # Current state for next step
            curr_state = next_state

            # update epsilon
            agent.updateEpsilon()


            if done:
                # Print
                print("Run: " + str(episode+1) + " ,Exp: " + str(agent.epsilon) \
                      + " ,Steps:" + str(step) + " ,Pos:" + str(curr_state[0][0]) \
                      + " ,Reward:" + str(episode_reward))

                reward_history.append(episode_reward)
                moving_avg_reward.append(np.sum(np.array(reward_history[-100:])))
                moving_avg_reward[-1] = moving_avg_reward[-1]/min(100,len(moving_avg_reward))

                # Position History
                position_history.append(curr_state[0][0])
                moving_avg_position.append(np.sum(np.array(position_history[-100:])))
                moving_avg_position[-1] = moving_avg_position[-1]/min(100, len(moving_avg_position))

                # step history
                step_history.append(step)
                moving_avg_steps.append(np.sum(np.array(step_history[-100:])))
                moving_avg_steps[-1] = moving_avg_steps[-1]/min(100, len(moving_avg_steps))

                if episode == 1:
                    model_json = agent.model.to_json()
                    with open("model_problem1_1.json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    agent.model.save_weights("model_problem1_1.h5")
                elif episode == 500:
                    model_json = agent.model.to_json()
                    with open("model_problem1_500.json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    agent.model.save_weights("model_problem1_500.h5")
                break

    # Ploting the Reward
    plt.xlim([0,NUM_EPISODES])
    plt.plot(reward_history,'r-')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Per Episode')
    plt.savefig('Reward_plot.png')
    plt.close()

    # Plotting the Moving Average Reward
    plt.xlim([0,NUM_EPISODES])
    plt.plot(moving_avg_reward,'b-')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Moving Average Reward')
    plt.savefig('Avg_Reward_plot.png')
    plt.close()

    # Plotting the Moving Average Reward
    plt.xlim([0,NUM_EPISODES])
    plt.plot(moving_avg_position,'b-')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.title('Moving Average position')
    plt.savefig('Avg_position_plot.png')
    plt.close()

    # Stats
    print("Episode taken to reach steps 175", np.where(np.array(step_history) < 175)[0][0])
    print("Episode taken to reach steps 150", np.where(np.array(step_history) < 150)[0][0])
    print("Episode taken to reach steps 100", np.where(np.array(step_history) < 100)[0][0])

    print("Episode taken to reach reward of 175", np.where(np.array(reward_history) > -175)[0][0])
    print("Episode taken to reach reward of 150", np.where(np.array(reward_history) > -150)[0][0])
    print("Episode taken to reach reward of 100", np.where(np.array(reward_history) > -100)[0][0])
    print("Minimum steps:", np.amin(np.array(step_history)))


    model_json = agent.model.to_json()
    with open("model_problem1_final.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    agent.model.save_weights("model_problem1_final.h5")


if __name__ == "__main__":
    main()
