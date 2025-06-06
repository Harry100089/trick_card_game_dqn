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

        # Reward: +1 for winning, -1 for losing, 0 for neutral actions
        reward = 1 if winner == 0 else -1
        
        # Game over reward: +10 for winning the game, -10 for losing
        if self.done:
            if self.tricks_won[0] > self.tricks_won[1]:  # Player 1 wins
                reward += 10  # Big reward for winning the game
            else:
                reward -= 10  # Large penalty for losing the game

        return self.get_state(), reward, self.done

