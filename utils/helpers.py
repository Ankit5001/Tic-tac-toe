import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

class Logger:
    """
    Advanced logging utility with multiple log levels and file management
    """
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique log filename
        log_filename = os.path.join(
            log_dir, 
            f"tictactoe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_training_metrics(self, metrics):
        """
        Log training metrics
        
        Args:
            metrics (dict): Training metrics to log
        """
        # Convert metrics to a formatted string
        metrics_str = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
        
        # Log the metrics
        self.logger.info("Training Metrics:\n%s", metrics_str)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)

class Visualizer:
    """
    Advanced visualization utilities for machine learning metrics
    """
    @staticmethod
    def plot_training_progress(rewards, filename='training_progress.png'):
        """
        Create comprehensive training progress visualization
        
        Args:
            rewards (list): List of rewards per episode
            filename (str): Output filename
        """
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # Rewards plot
        plt.subplot(2, 1, 1)
        plt.plot(rewards, label='Episode Rewards')
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # Moving average
        window_size = max(10, len(rewards) // 20)
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(len(moving_avg)), moving_avg, color='red', label=f'{window_size}-Episode Moving Avg')
        plt.legend()
        
        # Cumulative rewards
        plt.subplot(2, 1, 2)
        plt.plot(np.cumsum(rewards), label='Cumulative Rewards', color='green')
        plt.title('Cumulative Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join('logs', filename))
        plt.close()
    
    @staticmethod
    def plot_win_rates(win_counts, filename='win_rates.png'):
        """
        Advanced win rate visualization
        
        Args:
            win_counts (dict): Dictionary of win counts
            filename (str): Output filename
        """
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Pie chart
        plt.subplot(1, 2, 1)
        labels = list(win_counts.keys())
        sizes = list(win_counts.values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Game Outcome Distribution')
        
        # Bar chart
        plt.subplot(1, 2, 2)
        plt.bar(labels, sizes)
        plt.title('Win Counts')
        plt.ylabel('Number of Games')
        
        plt.tight_layout()
        plt.savefig(os.path.join('logs', filename))
        plt.close()

class DataManager:
    """
    Comprehensive data management utility
    """
    @staticmethod
    def save_training_data(data, filename='training_data.json'):
        """
        Save training data with metadata
        
        Args:
            data (dict): Training data to save
            filename (str): Output filename
        """
        os.makedirs('data', exist_ok=True)
        full_path = os.path.join('data', filename)
        
        # Add timestamp to data
        data['timestamp'] = datetime.now().isoformat()
        
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    @staticmethod
    def load_training_data(filename='training_data.json'):
        """
        Load training data with error handling
        
        Args:
            filename (str): Input filename
        
        Returns:
            dict: Loaded training data
        """
        full_path = os.path.join('data', filename)
        
        try:
            with open(full_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

def create_project_structure():
    """
    Create comprehensive project directory structure
    """
    directories = [
        'logs',      # For log files
        'models',    # For saved models
        'data',      # For training data
        'plots',     # For visualization outputs
        'checkpoints',# For model checkpoints
        'config'     # For configuration files
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project structure created successfully.")

# Example usage
if __name__ == "__main__":
    create_project_structure()
    
    # Example logging
    logger = Logger()
    logger.log_training_metrics({
        'episodes': 1000,
        'final_reward': 100,
        'exploration_rate': 0.1
    })
    
    # Example visualization
    sample_rewards = np.random.randn(100).cumsum()
    Visualizer.plot_training_progress(sample_rewards)