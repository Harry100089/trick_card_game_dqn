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
    
    # Determine which player's hand size is valid
    current_hand_size = len(env.players_hands[0 if env.current_trick % 2 == 0 else 1])
    
    # Ensure action is within the valid range based on current hand size
    action = q_values.argmax().item()
    action = min(action, current_hand_size - 1)  # Make sure the action is within bounds (0 to current_hand_size - 1)
    
    # Take the action in the environment
    next_state, reward, done = env.step(action)
    
    # Accumulate the reward
    total_reward += reward
    
    # Update the state
    state = next_state

print(f"Total reward for the game: {total_reward}")
