import random
import copy
import matplotlib.pyplot as plt
import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3
gamma = 0.9
epsilon = 0.1
states = []
statestbl = {}
S = {}
win = {}
children = {}

def board_to_string(board):
    """
    Convert a 2D list representing a tic-tac-toe board to a string.
    """
    return "".join(["".join(row) for row in board])


def check_win(board, player):
    """
    Check if a player has won the game on the given board.
    Returns True if the player has won, False otherwise.
    """
    
    # Check rows
    for row in board:
        if row.count(player) == 3:
            return True
    
    # Check columns
    for i in range(3):
        if board[0][i] == player and board[1][i] == player and board[2][i] == player:
            return True
    
    # Check diagonals
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    
    # No win found
    return False

def is_board_filled(board):
        for row in board:
            for item in row:
                if item == '-':
                    return False
        return True

def reward(boardToString):
    """
    return reward of a state
    """
    if boardToString in win.keys():
        return win[boardToString]
    elif is_board_filled([[boardToString[i] for i in range(j * 3, (j + 1) * 3)] for j in range(3)]):
        return 0
    else:
        return 1


def list_valid_states(board,children):
    """
    list all possible states in the game
    """
    valid_states = {}
    #check if terminate
    if check_win(board, 'X'):
        selfChild = {}
        # selfChild[board_to_string(board)] = 0
        win[board_to_string(board)] = -10
        children[board_to_string(copy.deepcopy(board))] = selfChild
        S[board_to_string(board)] = 0
        return valid_states
    if check_win(board, 'O'):
        selfChild = {}
        # selfChild[board_to_string(board)] = 0
        win[board_to_string(board)] = 10
        children[board_to_string(copy.deepcopy(board))] = selfChild
        S[board_to_string(board)] = 0
        return valid_states
    
    num_x = sum(row.count("X") for row in board)
    num_o = sum(row.count("O") for row in board)
    
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if board[i][j] == "-":
                new_board = copy.deepcopy(board)  # Make a copy of the board
                if num_x == num_o:
                    new_board[i][j] = "X"
                    S[board_to_string(new_board)] = 0
                else:
                    new_board[i][j] = "O"
                    
                valid_states[board_to_string(new_board)] = 0
    children[board_to_string(copy.deepcopy(board))] = valid_states
    return valid_states

def valueIter(statestbl, children, epsilon):  
    """
    perform Value iteration
    """
    x = []
    U = statestbl
    Up = copy.deepcopy(statestbl)
    while True:
        delta = 0
        U = copy.deepcopy(Up)
        print("Value of Fig1: ", board_to_string([["X", "-", "X"], ["O", "O", "-"], ["X", "-", "-"]]), U[board_to_string([["X", "-", "X"], ["O", "O", "-"], ["X", "-", "-"]])])
        x.append(U[board_to_string([["X", "-", "X"], ["O", "O", "-"], ["X", "-", "-"]])])
        for state in statestbl.keys():
            child = children[state]
            # if state == board_to_string([["X", "-", "X"], ["O", "O", "O"], ["X", "-", "-"]]):
            #     print("hi")
            # if state == board_to_string([["X", "-", "X"], ["O", "O", "-"], ["X", "-", "-"]]):
            #     print("hi")
            # not empty
            if child:
                max = -99999999
                #pick a acition
                for action in child.keys():
                    # if action == board_to_string([["X", "-", "X"], ["O", "O", "-"], ["X", "-", "-"]]):
                    #     print("hi")
                    sum = 0
                    board2d = [[action[i] for i in range(j * 3, (j + 1) * 3)] for j in range(3)]
                    
                    # if check_win(board2d,"O"):
                    #     max = 10
                    #     break
                    # else:
                    #     for result in children[action]:
                    #         a = 1/len(children[action])
                    #         b = U[result]
                    #         sum += a*b
                    #     if sum > max:
                    #         max = sum


                    if len(children[action]) == 0:
                        max = U[action]
                    else:
                        for result in children[action]:
                            a = 1/len(children[action])
                            b = U[result]
                            sum += a*b
                        if sum > max:
                            max = sum
                newV = reward(state) + gamma*max
                Up[state] = reward(state) + gamma*max
            else:
                Up[state] = reward(state)
            if abs(Up[state] - U[state]) > delta:
                delta = abs(Up[state] - U[state])
        print("current delta is: ", delta)
        
        if delta < epsilon*(1-gamma)/gamma:
            break

    # plt.plot(np.arange(len(x)), x)
    # plt.xlabel("Iterations")
    # plt.ylabel("Value of Figure 1")
    # plt.show()
    return U


# Example usage
board = [["-", "-", "-"], ["-", "-", "-"], ["-", "-", "-"]]
statestbl[board_to_string(copy.deepcopy(board))] = 0
queue = [board_to_string(board)]
while queue:
    state = queue.pop(0)
    board2d = [[state[i] for i in range(j * 3, (j + 1) * 3)] for j in range(3)]  # Convert state to 2D list
    valid_states = list_valid_states(board2d, children)
    for valid_state in valid_states.keys():
        if valid_state not in statestbl.keys():
            statestbl[valid_state] = 0
            queue.append(valid_state)




print("statestbl has: ", len(statestbl))
print("children has: ",len(children))
print("S has: ",len(S))
# print(str([["-", "X", "O"], ["-", "-", "-"], ["-", "-", "-"]]), "inital value is: ", statestbl[board_to_string([["-", "X", "-"], ["-", "-", "-"], ["-", "-", "-"]])])
# print(board_to_string([["X", "O", "X"], ["-", "-", "-"], ["-", "-", "-"]]), "children: ", children[board_to_string([["X", "O", "X"], ["-", "-", "-"], ["-", "-", "-"]])])
# # print(len(children[board_to_string([["X", "O", "-"], ["X", "O", "-"], ["X", "-", "-"]])]))
# print("reward of ", board_to_string([["X", "O", "-"], ["X", "O", "-"], ["X", "-", "-"]]), reward(board_to_string([["X", "O", "-"], ["X", "O", "-"], ["X", "-", "-"]])))
# print("reward of ", board_to_string([["X", "O", "-"], ["X", "O", "-"], ["-", "-", "-"]]), reward(board_to_string([["X", "O", "-"], ["X", "O", "-"], ["-", "-", "-"]])))

####
# print(check_win([["X","O","X"],["X","O","-"],["-","O","-"]],"O"))

print("Value Iter stats!")
UStar = valueIter(S, children, epsilon)
print("Value Iter ends!")

# state = board_to_string([["X","O","X"],["X","O","-"],["-","O","-"]])
state = "X-O-X-OX-"
print("Value: ", state, UStar[state])

# print("Children keys of Fig1: ", board_to_string([["X", "-", "X"], ["O", "O", "-"], ["X", "-", "-"]]), children[board_to_string([["X", "-", "X"], ["O", "O", "-"], ["X", "-", "-"]])])
# # print("Value of max children: ", board_to_string([["X", "-", "X"], ["O", "O", "O"], ["X", "-", "-"]]), UStar[board_to_string([["X", "-", "X"], ["O", "O", "O"], ["X", "-", "-"]])])
# print(len(children[board_to_string([["X", "-", "X"], ["O", "O", "O"], ["X", "-", "-"]])]))
##############################
#Genertator
##############################

class TicTacToe:

    def __init__(self):
        self.board = []
        self.history = []

    def create_board(self):
        for i in range(BOARD_ROWS):
            row = []
            for j in range(BOARD_COLS):
                row.append('-')
            self.board.append(row)
    
    def initDict(self):
        symbols = ['-', 'X', 'O']
        board = []
        for i in range(BOARD_ROWS):
            row = []
            for j in range(BOARD_COLS):
                row.append('-')
            board.append(row)

    def get_random_first_player(self):
        return random.randint(0, 1)

    def fix_spot(self, board, row, col, player):
        board[row][col] = player

    def is_player_win(self, player):
        win = None

        n = len(self.board)

        # checking rows
        for i in range(n):
            win = True
            for j in range(n):
                if self.board[i][j] != player:
                    win = False
                    break
            if win:
                return win

        # checking columns
        for i in range(n):
            win = True
            for j in range(n):
                if self.board[j][i] != player:
                    win = False
                    break
            if win:
                return win

        # checking diagonals
        win = True
        for i in range(n):
            if self.board[i][i] != player:
                win = False
                break
        if win:
            return win

        win = True
        for i in range(n):
            if self.board[i][n - 1 - i] != player:
                win = False
                break
        if win:
            return win
        return False

        for row in self.board:
            for item in row:
                if item == '-':
                    return False
        return True
    # def reward(winner):
    def getReward(self, player):
        if self.is_player_win(player):
                if player == 'X':
                    return -10
                    # self.updateStateValue(-10)
                else:
                    return 10
                    # self.updateStateValue(10)
        else:
            return 1

    def is_board_filled(self):
        for row in self.board:
            for item in row:
                if item == '-':
                    return False
        return True

    def swap_player_turn(self, player):
        return 'X' if player == 'O' else 'O'

    def show_board(self):
        print("Current board in 2d form: ")
        for row in self.board:
            for item in row:
                print(item, end=" ")
            print()
    
    def show_board_to_string(self):
        print("String form: ")
        print(board_to_string(self.board))

    def show_history(self):
        for state in self.history:
            for row in state:
                for item in row:
                    print(item, end=" ")
                print('\n')
            print(',')

    
    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i][j] == '-':
                    positions.append((i+1, j+1))  # need to be tuple
        return positions
    

    def randomMove(self):
        availablePositions = self.availablePositions()
        idx = random.randint(0, len(availablePositions)-1)
        return availablePositions[idx]

    # def updateStateValue(self, reward):
    #     stateHash = str(self.board)
    #     states_value[stateHash] = reward
        


    def start(self):

        self.create_board()
        states = []
        player = 'X'
        num = 0
        print("Start!")
        while True:
            
            
            self.history.append(copy.deepcopy(self.board))

            if player == 'X':
                # print(f"Opponent turn")
                row, col = self.randomMove()
                self.fix_spot(self.board, row - 1, col - 1, player)

            if player == 'O':
                row, col = self.randomMove()
                print("Your turn. You play: ", row, col)
                self.fix_spot(self.board, row - 1, col - 1, player)

            # if num == 0:
            #     self.show_board()
            # self.show_board_to_string()
            # print("R:", self.getReward(player))
            print("\n")
            if player == 'X':
                self.show_board()
                self.show_board_to_string()
                states.append(self.board)

            # checking whether current player is won or not
            if self.is_player_win(player):
                
                if player == 'X':
                    print()
                    print(f"Opponent wins the game!")
                    self.show_board()
                    # self.updateStateValue(-10)
                else:
                    print()
                    print(f"You win the game!")
                    self.show_board()
                    # self.updateStateValue(10)
                break

            # checking whether the game is draw or not
            if self.is_board_filled():
                print()
                print("Match Draw!")
                self.show_board()
                # self.updateStateValue(1)
                break

            # swapping the turn
            player = self.swap_player_turn(player)
            num += 1 
        return states
        # showing the final view of board
        # self.show_board_to_string()
        


# uncomment below to see two random players play the game

# budget = 1
# i = 0
# trajs = []
# while(i < budget):
#     tic_tac_toe = TicTacToe()
#     traj = tic_tac_toe.start()
#     trajs.append(traj)
#     i += 1
#     print(traj)