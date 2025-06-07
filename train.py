import torch
import torch.optim as optim
import numpy as np
from model import CardGameModel
from game import CardGame
import random
from collections import deque

# Hyperparameters
input_size = 20  # 10 cards per player * 2 players
output_size = 10 # 10 possible actions (cards to play)

initial_epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

batch_size = 32
discount_factor = 0.9
learning_rate = 0.001
num_episodes = 10000
replay_buffer = deque(maxlen=10000)

# Initialize the game environment and the model
env = CardGame()
model = CardGameModel(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

def train():
    epsilon = initial_epsilon

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Get the current player's hand size
                current_hand_size = len(env.players_hands[0 if env.current_trick % 2 == 0 else 1])
                action = random.choice(range(current_hand_size))  # Limit action to current hand size
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                # Get the current player's hand size and select the best action within range
                current_hand_size = len(env.players_hands[0 if env.current_trick % 2 == 0 else 1])
                action = q_values.argmax().item()
                action = min(action, current_hand_size - 1)  # Ensure action is within valid bounds

            next_state, reward, done = env.step(action)

            # Store the experience in the replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            # Sample a batch from the replay buffer
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(np.array(actions))
                rewards = torch.FloatTensor(np.array(rewards))
                next_states = torch.FloatTensor(np.array(next_states))

                # Compute the target Q-values
                with torch.no_grad():
                    target_q_values = rewards + discount_factor * model(next_states).max(dim=1)[0] * (1 - torch.FloatTensor(dones))

                # Get the current Q-values
                current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute the loss
                loss = criterion(current_q_values, target_q_values)

                # Update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed")

    # Save the trained model after training
    torch.save(model.state_dict(), "program_output/trained_model.pth")
    print("Model saved to 'program_output/trained_model.pth'")

# Start training
if __name__ == "__main__":
    train()
