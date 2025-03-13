import sys
import os
import argparse
import logging
import tkinter as tk

# Import custom modules
from gui.tictactoe_gui import TicTacToeGUI
from agent.agent import TicTacToeAgent
from agent.trainer import TicTacToeTrainer
from environment.tictactoe_env import TicTacToeEnv
from utils.helpers import (
    Logger, 
    Visualizer, 
    DataManager, 
    create_project_structure
)

class TicTacToeApp:
    def __init__(self, mode='gui'):
        """
        Main application class for Tic Tac Toe RL Project
        
        Args:
            mode (str): Application mode (gui, train, evaluate)
        """
        # Initialize logging
        self.logger = Logger()
        
        # Set application mode
        self.mode = mode
        
        # Initialize core components
        self.env = TicTacToeEnv()
        self.agent = TicTacToeAgent()
        self.trainer = TicTacToeTrainer()
        
        # Ensure project structure
        self._setup_project()
    
    def _setup_project(self):
        """
        Set up project directory structure and initial configurations
        """
        try:
            # Create necessary directories
            create_project_structure()
            
            # Check and create config if not exists
            self._create_default_config()
            
            self.logger.log_training_metrics({
                "Project Setup": "Successful",
                "Mode": self.mode
            })
        except Exception as e:
            self.logger.logger.error(f"Project setup failed: {e}")
            sys.exit(1)
    
    def _create_default_config(self):
        """
        Create default configuration file if not exists
        """
        config_path = 'config/default_config.json'
        os.makedirs('config', exist_ok=True)
        
        if not os.path.exists(config_path):
            default_config = {
                "agent": {
                    "learning_rate": 0.001,
                    "exploration_rate": 1.0,
                    "exploration_decay": 0.995
                },
                "training": {
                    "episodes": 1000,
                    "batch_size": 32
                }
            }
            
            import json
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
    
    def run_gui(self):
        """
        Launch the Graphical User Interface
        """
        try:
            root = tk.Tk()
            root.title("Tic Tac Toe Reinforcement Learning")
            
            # Set window icon (optional)
            # root.iconbitmap('assets/icon.ico')
            
            # Create and run GUI
            app = TicTacToeGUI(root)
            root.mainloop()
        except Exception as e:
            self.logger.logger.error(f"GUI launch failed: {e}")
            sys.exit(1)
    
    def train_agent(self, config_path=None):
        """
        Train the RL agent
        
        Args:
            config_path (str, optional): Path to training configuration
        """
        try:
            # Load configuration
            if config_path:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {
                    "episodes": 5000,
                    "learning_rate": 0.001,
                    "exploration_rate": 1.0
                }
            
            # Perform training
            training_results = self.trainer.train(
                num_episodes=config.get('episodes', 5000),
                learning_rate=config.get('learning_rate', 0.001)
            )
            
            # Visualize and log results
            Visualizer.plot_training_progress(
                training_results['episode_rewards'], 
                'training_progress.png'
            )
            
            # Save training data
            DataManager.save_training_data({
                'rewards': training_results['episode_rewards'],
                'win_counts': training_results['win_counts']
            })
            
            # Save trained model
            self.agent.save_model()
            
            self.logger.log_training_metrics({
                "Training Complete": True,
                "Total Episodes": len(training_results['episode_rewards']),
                "Final Exploration Rate": training_results.get('final_exploration_rate', 0)
            })
        
        except Exception as e:
            self.logger.logger.error(f"Training failed: {e}")
            sys.exit(1)
    
    def evaluate_agent(self, num_games=100):
        """
        Evaluate agent's performance
        
        Args:
            num_games (int): Number of games to evaluate
        """
        try:
            # Load pre-trained model
            self.agent.load_model()
            
            # Perform evaluation
            evaluation_results = self.trainer.evaluate(num_games)
            
            # Visualize results
            Visualizer.plot_win_rates(
                evaluation_results, 
                'agent_performance.png'
            )
            
            # Log evaluation metrics
            self.logger.log_training_metrics({
                "Evaluation Games": num_games,
                "Results": evaluation_results
            })
            
            # Print results to console
            print("Agent Performance Evaluation:")
            for key, value in evaluation_results.items():
                print(f"{key}: {value}")
        
        except Exception as e:
            self.logger.logger.error(f"Evaluation failed: {e}")
            sys.exit(1)

def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Tic Tac Toe Reinforcement Learning Application"
    )
    
    parser.add_argument(
        '-m', '--mode', 
        choices=['gui', 'train', 'evaluate'], 
        default='gui',
        help='Application mode'
    )
    
    parser.add_argument(
        '-c', '--config', 
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '-g', '--games', 
        type=int, 
        default=100, 
        help='Number of games for evaluation'
    )
    
    return parser.parse_args()

def main():
    """
    Main entry point of the application
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create application instance
    app = TicTacToeApp(mode=args.mode)
    
    # Run based on selected mode
    if args.mode == 'gui':
        app.run_gui()
    elif args.mode == 'train':
        app.train_agent(args.config)
    elif args.mode == 'evaluate':
        app.evaluate_agent(args.games)

if __name__ == "__main__":
    main()