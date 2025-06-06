import numpy as np
import random

# Super basic 2 player trick game
# Standard 52 card deck
# Each player gets 10 cards
# Each turn one player plays a card first then the second player plays
# Higher card wins the trick
# Alternate who goes first each turn
# Most tricks wins the game after 10 rounds

class CardGame:
    def __init__(self):
        self.deck = self.create_deck()
        self.players_hands = self.deal_cards()
        self.tricks_won = [0, 0]
        self.current_trick = 0
        self.done = False

    def create_deck(self):
        ranks = list(range(1, 14))  # Ranks 1-13 (Ace is 1)
        deck = [(rank, suit) for rank in ranks for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']]
        return deck

    def deal_cards(self):
        random.shuffle(self.deck)
        return [self.deck[i*10:(i+1)*10] for i in range(2)]  # Deal 13 cards to each player
    
    def reset(self):
        self.deck = self.create_deck()
        self.players_hands = self.deal_cards()
        self.tricks_won = [0, 0]
        self.current_trick = 0
        self.done = False
        return self.get_state()
    
    def get_state(self):
        # Represent the hand of the player as a vector (10 cards)
        player1_hand = [card[0] for card in self.players_hands[0]]  # Just rank, not suit
        player2_hand = [card[0] for card in self.players_hands[1]]
        return np.array(player1_hand + player2_hand)
    
    def step(self, action):
        if self.done:
            raise Exception("Game over. Please reset.")

        # Get the current player's card
        current_card = self.players_hands[0 if self.current_trick % 2 == 0 else 1].pop(action)
        
        # Determine the winner
        if self.current_trick % 2 == 0:
            # Player 1's turn
            opponent_card = self.players_hands[1].pop(random.choice(range(len(self.players_hands[1]))))
        else:
            # Player 2's turn
            opponent_card = self.players_hands[0].pop(random.choice(range(len(self.players_hands[0]))))
        
        winner = 0 if current_card[0] > opponent_card[0] else 1
        self.tricks_won[winner] += 1
        
        # Update the trick count
        self.current_trick += 1
        
        # Check if the game is over
        self.done = self.current_trick >= 10
        
        reward = 1 if winner == 0 else 0  # Reward player 1 for winning
        return self.get_state(), reward, self.done
