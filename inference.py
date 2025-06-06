import torch
from model import CardGameModel
from game import CardGame
import numpy as np

# Load the trained model
model = CardGameModel(input_size=20, output_size=10)
model.load_state_dict(torch.load('program_output/trained_model.pth'))
model.eval()  # Set the model to evaluation mode (disables dropout, batchnorm, etc.)

# Initialize the environment (same game)
env = CardGame()

# Simulate one game and play using the trained model
state = env.reset()
done = False
total_reward = 0

while not done:
    # Get the model's action (card to play)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = model(state_tensor)
    action = q_values.argmax().item()  # Choose the action with the highest Q-value
    
    # Take the action in the environment
    next_state, reward, done = env.step(action)
    
    # Accumulate the reward
    total_reward += reward
    
    # Update the state
    state = next_state

print(f"Total reward for the game: {total_reward}")
