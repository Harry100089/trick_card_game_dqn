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
        # Check the number of cards dealt
        #print(f"Deck size: {len(self.deck)}")
        player1_hand = self.deck[:10]
        player2_hand = self.deck[10:20]
        #print(f"Player 1 hand: {len(player1_hand)} cards, Player 2 hand: {len(player2_hand)} cards")
        return [player1_hand, player2_hand]

    
    def reset(self):
        self.deck = self.create_deck()
        self.players_hands = self.deal_cards()
        self.tricks_won = [0, 0]
        self.current_trick = 0
        self.done = False
        return self.get_state()
    
    def get_state(self):
        # If a player has fewer than 10 cards, pad with zeros (or any other value you want)
        player1_hand = [card[0] for card in self.players_hands[0]]  # Just rank, not suit
        player2_hand = [card[0] for card in self.players_hands[1]]
        
        # Pad with zeros if the hand has fewer than 10 cards
        while len(player1_hand) < 10:
            player1_hand.append(0)  # Padding with 0, can change as needed
        while len(player2_hand) < 10:
            player2_hand.append(0)  # Padding with 0, can change as needed
        
        return np.array(player1_hand + player2_hand)


    
    def step(self, action):
        if self.done:
            raise Exception("Game over. Please reset.")

        # Determine which player's hand to pop from
        current_player_idx = 0 if self.current_trick % 2 == 0 else 1
        current_hand = self.players_hands[current_player_idx]
        
        # Check if action is within valid range
        if action < 0 or action >= len(current_hand):
            raise IndexError(f"Invalid action: {action}. Player has only {len(current_hand)} cards left.")

        # Get the current player's card
        current_card = current_hand.pop(action)
        
        # Determine the opponent's card
        opponent_idx = 1 - current_player_idx  # If 0 -> 1, if 1 -> 0
        opponent_hand = self.players_hands[opponent_idx]
        
        if len(opponent_hand) > 0:
            opponent_card = opponent_hand.pop(random.choice(range(len(opponent_hand))))
        else:
            opponent_card = None  # No more cards to play
        
        # Determine winner
        winner = 0 if current_card[0] > opponent_card[0] else 1 if opponent_card else 0  # If no opponent card, player 1 wins

        self.tricks_won[winner] += 1
        
        # Update the trick count
        self.current_trick += 1
        
        # Check if the game is over
        self.done = self.current_trick >= 10
        
        reward = 1 if winner == 0 else 0  # Reward player 1 for winning
        return self.get_state(), reward, self.done
