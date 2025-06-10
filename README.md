# Card Game AI with Deep Q-Learning  
This project trains a Deep Q-Network (DQN) model to play a simple two-player trick-taking card game optimally. The AI learns to play a custom 10-card trick game using reinforcement learning techniques.  
  
**Features**  
• Custom card game environment simulating 10 tricks between two players  
• Deep Q-Network (DQN) implemented in PyTorch for learning the optimal strategy  
• Experience replay buffer for efficient learning  
• Epsilon-greedy exploration strategy  
• Inference script to evaluate the trained model’s performance  
• Dockerfile for easy environment setup and running training in a container  
  
**Possible Future Checklist**  
• Have models train and play against each other  
• Use different reward structure to force valid actions instead of manually forcing
  
**Personal Notes**  
Build and run Docker container:
docker build -t trickgame_dqn .  
docker run --rm -it trickgame_dqn 
  
Run with container's shell:  
docker run --rm -it trickgame_dqn /bin/bash  
  
Run with bind mount:  
docker run --rm -it -v $(pwd):/app trickgame_dqn /bin/bash  
  
  
  
Model 1 seems to average 6.7173 out of possible [-20, 20] against random  
Model 2 (upped training eps to 10000 instead of 1000) average 7.56694 in [-20, 20]  
Model 3 (added epsilon decay from 1.0 to 0.01 with 0.995x per episode) average 9.38464 in [-20, 20]
