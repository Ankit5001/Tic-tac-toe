import tkinter as tk
import numpy as np
from tkinter import messagebox, simpledialog
from environment.tictactoe_env import TicTacToeEnv
from agent.agent import TicTacToeAgent

class TicTacToeGUI:
    def __init__(self, master):
        """
        Initialize the Tic Tac Toe GUI
        
        Args:
            master (tk.Tk): Main window
        """
        self.master = master
        self.master.title("Tic Tac Toe with RL Agent")
        
        # Initialize environment and agent
        self.env = TicTacToeEnv()
        self.agent = TicTacToeAgent()
        
        # Try to load pre-trained model
        try:
            self.agent.load_model()
            print("Loaded pre-trained model")
        except:
            print("No pre-trained model found. Training from scratch.")
        
        # Game state
        self.current_state = None
        self.game_mode = tk.StringVar(value="human_vs_agent")
        
        # Create GUI components
        self.create_widgets()
        
        # Initialize game
        self.reset_game()
    
    def create_widgets(self):
        """
        Create GUI widgets
        """
        # Game Board Frame
        self.board_frame = tk.Frame(self.master, bg='white')
        self.board_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Create 3x3 grid of buttons
        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(
                    self.board_frame, 
                    text='', 
                    font=('Arial', 20, 'bold'),
                    width=5, 
                    height=2,
                    command=lambda r=i, c=j: self.on_button_click(r*3 + c)
                )
                btn.grid(row=i, column=j, sticky='nsew', padx=5, pady=5)
                row.append(btn)
            self.buttons.append(row)
        
        # Configure grid weights
        for i in range(3):
            self.board_frame.grid_rowconfigure(i, weight=1)
            self.board_frame.grid_columnconfigure(i, weight=1)
        
        # Control Frame
        control_frame = tk.Frame(self.master)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Game Mode Selection
        tk.Radiobutton(
            control_frame, 
            text="Human vs Agent", 
            variable=self.game_mode, 
            value="human_vs_agent"
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(
            control_frame, 
            text="Agent vs Agent", 
            variable=self.game_mode, 
            value="agent_vs_agent"
        ).pack(side=tk.LEFT, padx=10)
        
        # Buttons
        tk.Button(
            control_frame, 
            text="Reset Game", 
            command=self.reset_game
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            control_frame, 
            text="Train Agent", 
            command=self.train_agent
        ).pack(side=tk.LEFT, padx=10)
    
    def reset_game(self):
        """
        Reset the game board and state
        """
        # Reset environment
        self.current_state = self.env.reset()
        
        # Clear button texts
        for row in self.buttons:
            for btn in row:
                btn.config(text='', state=tk.NORMAL, bg='SystemButtonFace')
        
        # Start game based on mode
        if self.game_mode.get() == "agent_vs_agent":
            self.agent_move()
    
    def on_button_click(self, position):
        """
        Handle human player's move
        
        Args:
            position (int): Clicked board position
        """
        # Check if move is valid
        if position not in self.env.get_valid_moves():
            return
        
        # Make human move
        self.make_move(position, player='human')
        
        # Agent's turn if in human vs agent mode
        if self.game_mode.get() == "human_vs_agent" and not self.env.game_over:
            self.agent_move()
    
    def make_move(self, position, player):
        """
        Make a move on the board
        
        Args:
            position (int): Board position
            player (str): 'human' or 'agent'
        """
        # Step in environment
        state, reward, done, info = self.env.step(position)
        self.current_state = state
        
        # Update GUI
        row, col = position // 3, position % 3
        symbol = 'X' if player == 'human' else 'O'
        self.buttons[row][col].config(text=symbol)
        
        # Check game end
        if done:
            self.end_game(info['winner'])
    
    def agent_move(self):
        """
        Let the agent make a move
        """
        # Get valid moves
        valid_moves = self.env.get_valid_moves()
        
        # Choose action
        action = self.agent.choose_action(self.current_state, valid_moves)
        
        # Make move
        self.make_move(action, player='agent')
        
        # If in agent vs agent mode, continue
        if (self.game_mode.get() == "agent_vs_agent" and 
            not self.env.game_over):
            self.master.after(500, self.agent_move)
    
    def end_game(self, winner):
        """
        Handle game end
        
        Args:
            winner (str): Game winner
        """
        # Disable buttons
        for row in self.buttons:
            for btn in row:
                btn.config(state=tk.DISABLED)
        
        # Show result
        if winner == 'agent':
            messagebox.showinfo("Game Over", "You Wins!")
        elif winner == 'opponent':
            messagebox.showinfo("Game Over", "Computer Wins!")
        else:
            messagebox.showinfo("Game Over", "It's a Draw!")
    
    def train_agent(self):
        """
        Open training dialog and train the agent
        """
        # Ask for number of training episodes
        episodes = simpledialog.askinteger(
            "Train Agent", 
            "Number of training episodes:", 
            initialvalue=1000, 
            minvalue=100, 
            maxvalue=10000
        )
        
        if episodes:
            # Perform training (you might want to show a progress bar)
            from agent.trainer import TicTacToeTrainer
            
            trainer = TicTacToeTrainer(num_episodes=episodes)
            results = trainer.train()
            
            # Save trained model
            self.agent.save_model()
            
            # Show training results
            messagebox.showinfo(
                "Training Complete", 
                f"Trained for {episodes} episodes\n"
                f"Agent Wins: {results['win_counts']['agent']}\n"
                f"Opponent Wins: {results['win_counts']['opponent']}\n"
                f"Draws: {results['win_counts']['draw']}"
            )

def main():
    root = tk.Tk()
    root.geometry("500x600")
    app = TicTacToeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()