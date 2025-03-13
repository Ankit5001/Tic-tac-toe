import numpy as np
from environment.tictactoe_env import TicTacToeEnv
from agent.agent import TicTacToeAgent

class TicTacToeTrainer:
    def __init__(self, num_episodes=1000):
        """
        Initialize training environment
        
        Args:
            num_episodes (int): Number of training episodes
        """
        self.env = TicTacToeEnv()
        self.agent = TicTacToeAgent()
        self.num_episodes = num_episodes
    
    def train(self):
        """
        Train the agent through multiple episodes
        
        Returns:
            dict: Training statistics
        """
        episode_rewards = []
        win_counts = {
            'agent': 0,
            'opponent': 0,
            'draw': 0
        }
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Agent's turn
                valid_moves = self.env.get_valid_moves()
                action = self.agent.choose_action(state, valid_moves)
                next_state, reward, done, info = self.env.step(action)
                
                # Update agent's knowledge
                self.agent.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                # Check game outcome
                if done:
                    if info['winner'] == 'agent':
                        win_counts['agent'] += 1
                    elif info['winner'] == 'opponent':
                        win_counts['opponent'] += 1
                    else:
                        win_counts['draw'] += 1
            
            episode_rewards.append(total_reward)
            
            # Periodic model saving and logging
            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}")
                self.agent.save_model()
        
        # Final training statistics
        return {
            'episode_rewards': episode_rewards,
            'win_counts': win_counts,
            'final_exploration_rate': self.agent.exploration_rate
        }
    
    def evaluate(self, num_games=100):
        """
        Evaluate agent's performance
        
        Args:
            num_games (int): Number of games to evaluate
        
        Returns:
            dict: Performance metrics
        """
        performance_metrics = {
            'wins': 0,
            'losses': 0,
            'draws': 0
        }
        
        for _ in range(num_games):
            state = self.env.reset()
            done = False
            
            while not done:
                valid_moves = self.env.get_valid_moves()
                action = self.agent.choose_action(state, valid_moves)
                state, reward, done, info = self.env.step(action)
            
            if info['winner'] == 'agent':
                performance_metrics['wins'] += 1
            elif info['winner'] == 'opponent':
                performance_metrics['losses'] += 1
            else:
                performance_metrics['draws'] += 1
        
        return performance_metrics

# Usage example
if __name__ == "__main__":
    trainer = TicTacToeTrainer(num_episodes=5000)
    training_results = trainer.train()
    print("Training Results:", training_results)
    
    evaluation_results = trainer.evaluate()
    print("Evaluation Results:", evaluation_results)