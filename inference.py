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

# Simulate n games and play using the trained model
num_games = 1000
total_rewards = []

for game in range(num_games):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Get the model's action (card to play)
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
        q_values = model(state_tensor)
        action = q_values.argmax().item()  # Choose the action with the highest Q-value

        # Ensure the action is valid: action should be within the range of remaining cards
        current_hand = env.players_hands[0 if env.current_trick % 2 == 0 else 1]
        if action >= len(current_hand):
            action = np.random.choice(len(current_hand))  # Choose a valid action randomly

        # Take the action in the environment
        next_state, reward, done = env.step(action)

        # Accumulate the reward
        total_reward += reward

        # Update the state
        state = next_state

    total_rewards.append(total_reward)
    print(f"Game {game + 1}/{num_games} - Total Reward: {total_reward}")

# Calculate the average reward for the 100 games
average_reward = np.mean(total_rewards)
print(f"Average reward over {num_games} games: {average_reward}")
