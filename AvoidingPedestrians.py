import numpy as np
import random
import pickle


#import note:
#The "other side + down" case is considered as just moving 1 cell, i.e., moving diagonally in one step.

class Environment:

    def __init__(self):
        self.board = []
        self.car = []
        self.pedestrianL = []
        self.pedestrianR = []
        self.goals = []
        self.hit = False

    def startWithoutPedestrians(self):
        #initialize
        self.board = np.zeros((7, 8))
        self.goals = np.array([[6,3], [6,4]])
        carStart = random.randint(3,4)
        self.car = np.array([0, carStart])
        self.hit = False

    def startWithPedestrians(self):
        #initialize
        self.board = np.zeros((7, 8))
        self.goals = np.array([[6,3], [6,4]])
        carStart = random.randint(3,4)
        self.car = np.array([0, carStart])
        pedestrianL = random.randint(0,7)
        pedestrianR = random.randint(0,7)
        self.pedestrianL = [4, pedestrianL]
        self.pedestrianR = [5, pedestrianR]
        self.board[tuple(self.car)] = 1
        self.board[tuple(self.pedestrianL)] = 2
        self.board[tuple(self.pedestrianR)] = 2
        self.hit = False

    def set(self, car, pedestrianL, pedestrianR):
        self.board = np.zeros((7, 8))
        self.goals = np.array([[6,3], [6,4]])
        self.car = np.array(car)
        self.pedestrianL = pedestrianL
        self.pedestrianR = pedestrianR
        self.board[tuple(self.car)] = 1
        self.board[tuple(self.pedestrianL)] = 2
        self.board[tuple(self.pedestrianR)] = 2
        self.hit = False

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
            
            self.board[tuple(state)] = 0
            if next_state[0] < 7:
                self.board[tuple(next_state)] = 1
            return next_state

        else:
            path = []
            # car move
            if action == 0:  # move down by 1
                next_state = [row + 1, col]
                path.append(next_state)

            elif action == 1:  # move down by 2
                next_state = [row + 2, col]
                path.append([row + 1, col])
                path.append(next_state)

            else:  # move sideways and down by 1
                if col == 3:
                    next_state = [row + 1, col + 1]
                    path.append(next_state)
                else:
                    next_state = [row + 1, col - 1]
                    path.append(next_state)
            
            self.board[tuple(state)] = 0
            self.car = next_state
            if next_state[0] < 7:
                self.board[tuple(next_state)] = 1

            # pedestrians move
            if self.pedestrianL[1] > 0:
                #check to avoid overwriting car's position
                if self.board[tuple(self.pedestrianL)] != 1:
                    self.board[tuple(self.pedestrianL)] = 0
                self.pedestrianL[1] -= 1
                self.board[tuple(self.pedestrianL)] = 2
            if self.pedestrianR[1] < 7:
                if self.board[tuple(self.pedestrianR)] != 1:
                    self.board[tuple(self.pedestrianR)] = 0
                self.pedestrianR[1] += 1
                self.board[tuple(self.pedestrianR)] = 2
            
            # check if pedestrians are in the path of the car
            for coord in path:
                if coord[0] == self.pedestrianL[0] and coord[1] == self.pedestrianL[1]:
                    # left pedestrian is hit
                    self.hit = True
                elif coord[0] == self.pedestrianR[0] and coord[1] == self.pedestrianR[1]:
                    # right pedestrian is hit
                    self.hit = True
            
            return next_state

        


    def getReward(self, state):
        if list(state) in self.goals.tolist():
            return 100
        elif state[0] > 6:
            return -100
        elif self.hit:
            return -100    
        else:
            return -5

    def isTerminal(self, state):
        return list(state) in self.goals.tolist() or state[0] > 6 or self.hit

    def reset(self):
        self.board = np.zeros((7, 8))
        self.car = []
        self.pedestrianL = []
        self.pedestrianR = []
        self.goals = []
        self.hit = False

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
# env.startWithPedestrians() # or env.startWithPedestrians()

# state = env.car

# while not env.isTerminal(state):
#     arrive = True
#     action = env.randomAction()
#     if action == 0:
#         move = "downward 1"
#     elif action == 1:
#         move = "downward 2"
#     elif action == 3:
#         move = "diagonal"
#     next_state = env.getNextState(state, action)
#     reward = env.getReward(next_state)

#     env.board[tuple(state)] = 0
    
#     print(f"Action taken: {move}(action {action}), ")
#     print(f"Car Next state: {next_state}")
#     print(f"PL Next state: {env.pedestrianL}")
#     print(f"PR Next state: {env.pedestrianR}")
#     print(f"Reward received: {reward}")
#     if next_state[0] > 6: 
#         print("Car out of boundary! BAD!")
#         arrive = False
#         break

#     # env.board[tuple(next_state)] = 1
#     # env.board[tuple(env.pedestrianR)] = 2
#     # env.board[tuple(env.pedestrianL)] = 2
#     env.visualize_board()
#     print()
#     state = next_state

# if arrive:
#     print("Arrive: ", state)



# '''
# Q learning without Pedestrains
# '''
# class QLearningAgent:
#     def __init__(self, alpha, gamma, epsilon, num_actions, state_space):
#         self.alpha = alpha  # learning rate
#         self.gamma = gamma  # discount factor
#         self.epsilon = epsilon  # exploration rate
#         self.num_actions = num_actions
#         self.state_space = state_space
#         self.q_table = np.zeros((state_space[0], state_space[1], num_actions))

#     def select_action(self, state):
#         if np.random.uniform() < self.epsilon:
#             return np.random.randint(self.num_actions)
#         else:
#             return np.argmax(self.q_table[state[0], state[1], :])

#     def update(self, state, action, next_state, reward, done):
#         q_predict = self.q_table[state[0], state[1], action]
#         if done:
#             q_target = reward
#         else:
#             q_target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], :])
#         self.q_table[state[0], state[1], action] += self.alpha * (q_target - q_predict)

# def train(agent, env, num_episodes, max_steps):
#     rewards = []
#     for i in range(num_episodes):
#         row = random.randint(0,6)
#         col = random.randint(3,4)
#         state = np.array([row, col])
#         episode_reward = 0
#         for j in range(max_steps):
#             action = agent.select_action(state)
#             next_state = env.getNextState(state, action)
#             reward = env.getReward(next_state)
#             done = env.isTerminal(next_state)

#             agent.update(state, action, next_state, reward, done)

#             episode_reward += reward
#             state = next_state

#             if done:
#                 break
#         rewards.append(episode_reward)

#     return agent.q_table, rewards

# """
# without pedestrians
# """
# env = Environment()
# env.startWithoutPedestrians() # or env.startWithPedestrians()

# # Q-learning parameters
# alpha = 0.5  # learning rate
# gamma = 0.9  # discount factor
# epsilon = 0.1  # exploration rate
# num_actions = 3  # move down by 1, move down by 2, move sideways and down by 1
# state_space = (7, 8)  # 2D state space

# agent = QLearningAgent(alpha, gamma, epsilon, num_actions, state_space)
# num_episodes = 100000
# max_steps = 10

# q_table, rewards = train(agent, env, num_episodes, max_steps)


# print("q values: ")
# for i in range(q_table.shape[0]):
#     for j in range(q_table.shape[1]):
#         print(f"Q-values for state ({i},{j}): {q_table[i,j,:]}")


# print("Optimal policy:")
# print("col3, col4")
# for row in range(state_space[0]-1):
#     row_str = "|"
#     for col in range(3,5):
#         if env.board[row, col] == 2:
#             row_str += " P |"
#         elif env.board[row, col] == 1:
#             row_str += " C |"
#         else:
#             action = np.argmax(q_table[row, col, :])
#             if action == 0:
#                 row_str += " d1 |"
#             elif action == 1:
#                 row_str += " d2 |"
#             else:
#                 row_str += " sd |"
#     print(row_str)
#     print("-" * 10)
# print("| P1 | P2 |")






"""
consider pedestrains
"""
class QLearner:

    def __init__(self, alpha, gamma, epsilon, actions):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.q_table = {}

    def getQValue(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state][action]

    def computeValueFromQValues(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return max(self.q_table[state])

    def computeActionFromQValues(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])

    def getAction(self, state):
        return self.computeActionFromQValues(state)

    def update(self, state, action, next_state, reward):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma * self.computeValueFromQValues(next_state))
    
    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

def train(agent, env, episodes):
    for episode in range(episodes):
        # env.reset()
        env.startWithPedestrians()
        #for now we only care about figure 2
        env.set([3,3], [3,5], [4,2])
        state = str(env.board)
        while not env.isTerminal(tuple(env.car)):
            action = agent.getAction(state)
            env.getNextState(tuple(env.car), action)
            next_state = str(env.board)
            reward = env.getReward(tuple(env.car))
            agent.update(state, action, next_state, reward)
            state = next_state
        print("episode: ", episode)


# def test(agent, env):
#     env.reset()
#     env.startWithPedestrians()
#     state = tuple(env.car)
#     while not env.isTerminal(state):
#         action = agent.computeActionFromQValues(state)
#         next_state = tuple(env.getNextState(list(state), action))
#         reward = env.getReward(next_state)
#         state = next_state
#         env.visualize_board
#         if reward == -100:
#             print("The agent hit a pedestrian")
#             break
#         elif reward == 100:
#             print("The agent successfully crossed the road!")
#             break

# set up environment and agent
env = Environment()
env.startWithPedestrians()
actions = [0, 1, 2]
agent = QLearner(alpha=0.5, gamma=0.9, epsilon=0.2, actions=actions)
agent.load_q_table('q_table.pkl')
# train the agent

train(agent, env, episodes=10)
agent.save_q_table('q_table.pkl')


# test the agent
# test(agent, env)
condition = np.zeros((7, 8))
condition[3,3] = 1
condition[3,5] = 2
condition[4,2] = 2

print("Count:", len(agent.q_table.keys()))
print("Figure2: ")
print(str(condition))
# print("Example key: ")
# print(list(agent.q_table.keys())[1])

# Qvalues = agent.q_table[list(agent.q_table.keys())[1]]
if str(condition) in agent.q_table.keys():
    Qvalues = agent.q_table[str(condition)]
    print(Qvalues)
else:
    print("Doesn't exist")