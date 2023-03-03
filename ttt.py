import random
import copy

BOARD_ROWS = 3
BOARD_COLS = 3
gamma = 0.9
epsilon = 0.1

states_value = {}

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

    def get_random_first_player(self):
        return random.randint(0, 1)

    def fix_spot(self, row, col, player):
        self.board[row][col] = player

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


    def is_board_filled(self):
        for row in self.board:
            for item in row:
                if item == '-':
                    return False
        return True

    def swap_player_turn(self, player):
        return 'X' if player == 'O' else 'O'

    def show_board(self):
        for row in self.board:
            for item in row:
                print(item, end=" ")
            print()
    
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

    def start(self):

        self.create_board()
       
        player = 'X'
        while True:

            reward = 0

            self.history.append(copy.deepcopy(self.board))

            if player == 'X':
               
                print(f"Player opponent turn")


            row, col = self.randomMove()
            if player == 'O':
                print("Your turn. Enter row and column numbers to fix spot: ", row, col)


            # fixing the spot
            self.fix_spot(row - 1, col - 1, player)

            self.show_board()
            print("\n")
            

            # checking whether current player is won or not
            if self.is_player_win(player):
                print(f"Player {player} wins the game!")
                if player == 'X':
                    reward = -10
                else:
                    reward = 10
                break

            # checking whether the game is draw or not
            if self.is_board_filled():
                print("Match Draw!")
                reward = 1
                break

            # swapping the turn
            player = self.swap_player_turn(player)
            reward = 1  

        # showing the final view of board
        print()
        self.show_board()
        
        state = str(self.board)
        #self.show_history()
        ##print(self.board)


# starting the game
tic_tac_toe = TicTacToe()
tic_tac_toe.start()