import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity=10000):
        """
        Initialize Experience Replay Memory
        
        Args:
            capacity (int): Maximum number of experiences to store
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Store an experience in memory
        
        Args:
            state (numpy.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.array): Next state
            done (bool): Whether episode is complete
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences
        
        Args:
            batch_size (int): Number of experiences to sample
        
        Returns:
            list: Batch of experiences
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """
        Get current memory size
        
        Returns:
            int: Number of experiences stored
        """
        return len(self.memory)