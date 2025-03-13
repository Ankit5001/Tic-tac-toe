import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class TicTacToeAgent:
    def __init__(self, 
                 state_size=9,  # 3x3 board
                 action_size=9, 
                 learning_rate=0.001, 
                 exploration_rate=1.0, 
                 min_exploration_rate=0.01,
                 exploration_decay=0.995):
        """
        Initialize the Reinforcement Learning Agent for Tic Tac Toe
        
        Args:
            state_size (int): Dimension of the game state
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate for the neural network
            exploration_rate (float): Initial exploration probability
            min_exploration_rate (float): Minimum exploration probability
            exploration_decay (float): Rate of exploration decay
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural Network for Q-Learning
        self.model = self._build_neural_network()
        
        # Optimizer and Loss Function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Exploration parameters
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        
        # Experience replay buffer
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
    
    def _build_neural_network(self):
        """
        Create a simple neural network for Q-learning
        
        Returns:
            nn.Sequential: Neural network model
        """
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def choose_action(self, state, valid_moves):
        """
        Choose an action using epsilon-greedy strategy
        
        Args:
            state (numpy.array): Current game state
            valid_moves (list): List of valid moves
        
        Returns:
            int: Chosen action
        """
        # Exploration vs Exploitation
        if random.random() < self.exploration_rate:
            return random.choice(valid_moves)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get Q-values from the neural network
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        # Filter Q-values for valid moves
        valid_q_values = q_values[0][valid_moves]
        best_action_index = valid_moves[torch.argmax(valid_q_values).item()]
        
        return best_action_index
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on the experience
        
        Args:
            state (numpy.array): Current state
            action (int): Chosen action
            reward (float): Reward received
            next_state (numpy.array): Next state
            done (bool): Whether the game is finished
        """
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Trigger learning when memory is sufficiently filled
        if len(self.memory) > self.batch_size:
            self._learn()
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
    
    def _learn(self):
        """
        Perform learning from stored experiences
        """
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])
        
        # Compute current Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        
        # Backpropagate and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, filepath='models/tictactoe_agent.pth'):
        """Save the trained model"""
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath='models/tictactoe_agent.pth'):
        """Load a pre-trained model"""
        self.model.load_state_dict(torch.load(filepath))