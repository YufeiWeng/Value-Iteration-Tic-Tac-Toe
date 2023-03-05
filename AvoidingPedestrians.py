import numpy as np
import random

class Environment:

    def __init__(self):
        self.board = []
        self.car = []
        self.pedestrianL = []
        self.pedestrianR = []
        self.goals = []

    def startWithoutPedestrians(self):
        #initialize
        self.board = np.zeros((7, 8))
        self.goals = np.array([[6,3], [6,4]])
        carStart = random.randint(3,4)
        self.car = np.array([0, carStart])

    def startWithPedestrians(self):
        #initialize
        self.board = np.zeros((7, 8))
        self.goals = np.array([[6,3], [6,4]])
        carStart = random.randint(0,1)
        self.car = np.array([0, carStart])
        pedestrianL = random.randint(0,7)
        pedestrianR = random.randint(0,7)
        self.board[4, pedestrianL] = 2
        self.board[5, pedestrianR] = 2

    def getNextState(self, state, action):
        row, col = state
        #if we don't consider pedestrains
        if not self.pedestrianL and not self.pedestrianR:
            if action == 0:  # move down by 1
                next_state = [row + 1, col]

            elif action == 1:  # move down by 2
                # if row > 4:
                #     print("Can only move 1 step down")
                #     return state
                next_state = [row + 2, col]

            else:  # move sideways and down by 1
                if col == 3:
                    next_state = [row + 1, col + 1]
                else:
                    next_state = [row + 1, col - 1]

            return next_state
        
        else:
            pass
    def getReward(self, state):
        if list(state) in self.goals.tolist():
            return 100
        elif state[0] > 6:
            return -100
        else:
            return -1

    def isTerminal(self, state):
        return list(state) in self.goals.tolist() or state[0] > 6

    def reset(self):
        self.board = np.zeros((7, 8))
        self.car = []
        self.pedestrianL = []
        self.pedestrianR = []
        self.goals = []

    def render(self):
        print(self.board)

    def randomAction(self):
        return random.randint(0, 2)
    
    def visualize_board(self):
        for row in self.board:
            row_str = "|"
            for val in row:
                if val == 0:
                    row_str += "   |"
                elif val == 1:
                    row_str += " C |"
                elif val == 2:
                    row_str += " P |"
            print(row_str)
            print("-" * 33)



# '''
# test implement
# '''
# env = Environment()
# env.startWithoutPedestrians() # or env.startWithPedestrians()

# state = env.car

# while not env.isTerminal(state):
#     arrive = True
#     action = env.randomAction()
#     next_state = env.getNextState(state, action)
#     reward = env.getReward(next_state)

#     env.board[tuple(state)] = 0
    
#     print(f"Action taken: {action}")
#     print(f"Next state: {next_state}")
#     print(f"Reward received: {reward}")
#     if next_state[0] > 6: 
#         print("Car out of boundary! BAD!")
#         arrive = False
#         break

#     env.board[tuple(next_state)] = 1
#     env.visualize_board()
#     print()
#     state = next_state

# if arrive:
#     print("Arrive: ", state)



'''
Q learning
'''
class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, num_actions, state_space):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.num_actions = num_actions
        self.state_space = state_space
        self.q_table = np.zeros((state_space[0], state_space[1], num_actions))

    def select_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state[0], state[1], :])

    def update(self, state, action, next_state, reward, done):
        q_predict = self.q_table[state[0], state[1], action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], :])
        self.q_table[state[0], state[1], action] += self.alpha * (q_target - q_predict)

def train(agent, env, num_episodes, max_steps):
    rewards = []
    for i in range(num_episodes):
        state = env.car
        episode_reward = 0
        for j in range(max_steps):
            action = agent.select_action(state)
            next_state = env.getNextState(state, action)
            reward = env.getReward(next_state)
            done = env.isTerminal(next_state)

            agent.update(state, action, next_state, reward, done)

            episode_reward += reward
            state = next_state

            if done:
                break
        rewards.append(episode_reward)

    return agent.q_table, rewards

"""
without pedestrians
"""
env = Environment()
env.startWithoutPedestrians() # or env.startWithPedestrians()

# Q-learning parameters
alpha = 0.5  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate
num_actions = 3  # move down by 1, move down by 2, move sideways and down by 1
state_space = (7, 8)  # 2D state space

agent = QLearningAgent(alpha, gamma, epsilon, num_actions, state_space)
num_episodes = 100000
max_steps = 10

q_table, rewards = train(agent, env, num_episodes, max_steps)


print("q values: ")
for i in range(q_table.shape[0]):
    for j in range(q_table.shape[1]):
        print(f"Q-values for state ({i},{j}): {q_table[i,j,:]}")


print("Optimal policy:")
print("col3, col4")
for row in range(state_space[0]):
    row_str = "|"
    for col in range(3,5):
        if env.board[row, col] == 2:
            row_str += " P |"
        elif env.board[row, col] == 1:
            row_str += " C |"
        else:
            action = np.argmax(q_table[row, col, :])
            if action == 0:
                row_str += " d1 |"
            elif action == 1:
                row_str += " d2 |"
            else:
                row_str += " sd |"
    print(row_str)
    print("-" * 10)
