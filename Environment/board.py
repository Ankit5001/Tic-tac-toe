import numpy as np

class TicTacToeBoard:
    def __init__(self, size=3):
        """
        Initialize the Tic Tac Toe board
        
        Args:
            size (int): Size of the board (default 3x3)
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # Player 1 starts (1 for X, -1 for O)
    
    def reset(self):
        """
        Reset the board to initial state
        
        Returns:
            numpy.array: Reset board state
        """
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        return self.board.flatten()
    
    def make_move(self, position):
        """
        Place a move on the board
        
        Args:
            position (int): Position to place the move (0-8)
        
        Returns:
            bool: Whether the move was valid
        """
        row = position // self.size
        col = position % self.size
        
        # Check if the position is empty
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player *= -1  # Switch players
            return True
        return False
    
    def check_winner(self):
        """
        Check if there's a winner
        
        Returns:
            int: Winning player (1 or -1), or 0 if no winner
        """
        # Check rows
        for row in self.board:
            if np.all(row == row[0]) and row[0] != 0:
                return row[0]
        
        # Check columns
        for col in range(self.size):
            if np.all(self.board[:, col] == self.board[0, col]) and self.board[0, col] != 0:
                return self.board[0, col]
        
        # Check diagonals
        if np.all(np.diag(self.board) == self.board[0, 0]) and self.board[0, 0] != 0:
            return self.board[0, 0]
        
        if np.all(np.diag(np.fliplr(self.board)) == self.board[0, -1]) and self.board[0, -1] != 0:
            return self.board[0, -1]
        
        return 0
    
    def is_board_full(self):
        """
        Check if the board is completely filled
        
        Returns:
            bool: Whether the board is full
        """
        return np.all(self.board != 0)
    
    def get_valid_moves(self):
        """
        Get list of valid moves
        
        Returns:
            list: Indices of empty positions
        """
        return [i for i in range(self.size**2) if self.board[i // self.size, i % self.size] == 0]
    
    def display(self):
        """
        Display the current board state
        """
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print('|'.join([symbols[cell] for cell in row]))
            print('-' * (2 * self.size - 1))