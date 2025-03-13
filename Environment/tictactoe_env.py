import numpy as np
import random
from environment.board import TicTacToeBoard

class TicTacToeEnv:
    def __init__(self):
        """
        Initialize the Tic Tac Toe environment
        """
        self.board = TicTacToeBoard()
        self.game_over = False
        self.winner = None
    
    def reset(self):
        """
        Reset the environment
        
        Returns:
            numpy.array: Initial board state
        """
        state = self.board.reset()
        self.game_over = False
        self.winner = None
        return state
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): Position to place the move
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Validate the action
        if action not in self.get_valid_moves():
            raise ValueError("Invalid move")
        
        # Make the move
        self.board.make_move(action)
        
        # Check game state
        winner = self.board.check_winner()
        is_full = self.board.is_board_full()
        
        # Determine reward and game status
        reward = 0
        done = False
        
        if winner == 1:  # Agent wins
            reward = 10
            done = True
            self.winner = 'agent'
        elif winner == -1:  # Opponent wins
            reward = -10
            done = True
            self.winner = 'opponent'
        elif is_full:  # Draw
            reward = 0
            done = True
            self.winner = 'draw'
        
        self.game_over = done
        
        # Return next state, reward, done flag, and additional info
        return (
            self.board.board.flatten(),  # Next state
            reward,                      # Reward
            done,                        # Game over flag
            {
                'winner': self.winner,
                'board': self.board.board.copy()
            }
        )
    
    def get_valid_moves(self):
        """
        Get list of valid moves
        
        Returns:
            list: Indices of empty positions
        """
        return self.board.get_valid_moves()
    
    def render(self):
        """
        Render the current board state
        """
        self.board.display()
    
    def opponent_move(self):
        """
        Simulate an opponent move
        
        Returns:
            int: Position of opponent's move
        """
        valid_moves = self.get_valid_moves()
        
        # Different opponent strategies can be implemented here
        # 1. Random move
        return random.choice(valid_moves)
        
        # 2. Simple strategic move (can be expanded)
        # for move in valid_moves:
        #     if self._is_winning_move(move):
        #         return move
        # return random.choice(valid_moves)
    
    def _is_winning_move(self, move):
        """
        Check if a move would result in a win
        
        Args:
            move (int): Position to check
        
        Returns:
            bool: Whether the move is a winning move
        """
        # Create a copy of the board
        temp_board = TicTacToeBoard()
        temp_board.board = self.board.board.copy()
        
        # Try the move
        temp_board.make_move(move)
        
        # Check if it results in a win
        return temp_board.check_winner() == -1

# Example usage
if __name__ == "__main__":
    env = TicTacToeEnv()
    
    # Sample game simulation
    state = env.reset()
    done = False
    
    while not done:
        # Agent's turn (random move for demonstration)
        valid_moves = env.get_valid_moves()
        action = random.choice(valid_moves)
        
        state, reward, done, info = env.step(action)
        env.render()
        
        if not done:
            # Opponent's turn
            opp_action = env.opponent_move()
            state, reward, done, info = env.step(opp_action)
            env.render()
    
    print(f"Game over. Winner: {info['winner']}")