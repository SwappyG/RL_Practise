import numpy as np
import matplotlib.pyplot as plt

from cards import Card, Deck
from policy import Policy, QPolicy, DealerPolicy, StateSpace

class Player(object):
	
	def __init__(self, id, policy = None):
		self.ID = id
		self.policy = policy
		self.hand = BlackJackHand()

	def GetState(self, dealer_card = None):
		return self.policy.GetState( self.hand.value, useable_ace=self.hand.useable_ace, dealer_card=dealer_card )

	def GetAction(self, state):
		return self.policy.GetAction(state)

	def AddCards(self, cards):
		for card in cards:
			self.hand.AddCard(card)

	def GetHandValue(self):
		return self.hand.value

	def GetFirstCard(self):
		return self.hand.GetFirstCard()

	def GetCards(self):
		return self.hand.cards

	def IsBust(self):
		return self.hand.IsBust()

	def DiscardHand(self):
		self.hand.NewHand()

	def ImprovePolicy(self, state, reward):
		self.policy.ImprovePolicy(state, reward)

class BlackJackHand(object):

	def __init__(self, starting_cards = None):
		if starting_cards == None:
			self.cards = []
		else:
			if all([isinstance(card,Card) for card in starting_cards]):
				self.cards = starting_cards
			else:
				raise RuntimeError("starting_cards argument must list of instances of Card")
		
		self.value = 0
		self.useable_ace = False
		self.is_bust = False
		self._UpdateValue()

	def __repr__(self):
		return "cards: {}, value: {}".format(self.cards, self.value)

	def NewHand(self):
		del self.cards
		self.__init__()

	def _UpdateValue(self):
		
		self.value = 0

		# go through all cards in the hand
		for card in self.cards:

			# get the value of the card
			this_val = np.minimum(card.number, 10)

			# if the card is an ace and we have less than 11 total, make it worth 11
			if card.is_ace and (self.value < 11):
				this_val = 11
				self.useable_ace = True

			# add the value to the running total
			self.value += this_val

			# if we have a useable ace and the total goes over 21, make the ace worth 1
			if self.useable_ace and self.value > 21:
				self.useable_ace = False
				self.value -= 10

		# If the hand value is over 21, this hand is bust
		if self.value > 21:
			self.is_bust = True

	def AddCard(self, card):
		if isinstance(card, Card):
			self.cards.append(card)
			self._UpdateValue()

	def RemoveLastCard(self):
		if self.cards == []:
			return False
		else:
			del self.cards[-1]
			self._UpdateValue()

	def GetFirstCard(self):
		if self.cards == []:
			return None
		else:
			return self.cards[0]

	def IsBust(self):
		return self.is_bust

	def RemoveCardByID(self, id):
		for ii, card in enumerate(self.cards):
			if card.ID == id:
				del self.card[ii]
				self._UpdateValue()
				return True

		return False

class BlackJackGame(object):

	IN_PROGRESS = 0
	PLAYER_WIN_HIGH_SCORE = 1
	PLAYER_WIN_DEALER_BUST = 2
	DEALER_WIN_HIGH_SCORE = 3
	DEALER_WIN_PLAYER_BUST = 4
	TIE = 5

	def __init__(self, player_policy = None, dealer_policy = None):
		self.deck = Deck()
		self.player = Player(id=1, policy=player_policy)
		self.player.AddCards(self.deck.DealCards(2))

		self.dealer = Player(id=2, policy=dealer_policy)
		self.dealer.AddCards(self.deck.DealCards(2))
		
		self.outcome = BlackJackGame.IN_PROGRESS
		self.player_stick = False
		self.dealer_stick = False

	def __repr__(self):
		return "p_stick: {}, d_stick: {}, outcome, {}".format(self.player_stick, self.dealer_stick, self.outcome)

	def RestartGame(self):
		self.player.DiscardHand()
		self.dealer.DiscardHand()

		self.player.AddCards(self.deck.DealCards(2, type=[Deck.ACE_50_50, Deck.ONLY_TEN]))
		self.dealer.AddCards(self.deck.DealCards(2, type=[Deck.ONE_TO_TEN, Deck.ONE_TO_TEN]))

		self.outcome = self.IN_PROGRESS
		self.player_stick = False
		self.dealer_stick = False

	def DealToPlayer(self, dealer = False):
		if dealer:
			self.dealer.AddCards(self.deck.DealCards(1))
		else:
			# print self.player.hand
			self.player.AddCards(self.deck.DealCards(1))
			# print self.player.hand

		self.CheckOutcome()
		return self.outcome

	def CheckOutcome(self):

		# If player or dealer are bust
		if self.player.IsBust():
			self.outcome = BlackJackGame.DEALER_WIN_PLAYER_BUST
		elif self.dealer.IsBust():
			self.outcome = BlackJackGame.PLAYER_WIN_DEALER_BUST

		# If game is over with neither being bust
		elif self.player_stick and self.dealer_stick:
			if self.player.hand.value > self.dealer.hand.value:
				self.outcome = BlackJackGame.PLAYER_WIN_HIGH_SCORE
			elif self.player.hand.value < self.dealer.hand.value:
				self.outcome = BlackJackGame.DEALER_WIN_HIGH_SCORE
			else:
				self.outcome = BlackJackGame.TIE
		else:
			self.outcome = BlackJackGame.IN_PROGRESS

def GetStateFromGame(game):

	player_state = game.player.GetPlayerState()
	ace_state = game.player.GetAceState()
	dealer_state = game.dealer.GetFirstCardState()

	return (dealer_state, player_state, ace_state)
	


def GetRewardFromOutcome(outcome):
	if outcome == BlackJackGame.IN_PROGRESS or \
		outcome == BlackJackGame.TIE:
		return 0
	elif outcome == BlackJackGame.PLAYER_WIN_HIGH_SCORE or \
		outcome == BlackJackGame.PLAYER_WIN_DEALER_BUST:
		return 1
	else:
		return -1
	
# Plot the Q function and the current policy
def PlotResults(Q, R):
	
	fig, ax = plt.subplots(2,2)
	
	q_img = ax[0,0].imshow(Q[:,:,0,1] - Q[:,:,0,0	],  cmap='Set2', vmin=-2, vmax=2, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	q_img = ax[1,0].imshow(Q[:,:,1,1] - Q[:,:,1,0], cmap='Set2', vmin=-2, vmax=2, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	
	# q_img = ax[0,0].imshow(Q[:,:,0,HIT],  cmap='Set2', vmin=0, vmax=10, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	# q_img = ax[1,0].imshow(Q[:,:,0,STAY], cmap='Set2', vmin=0, vmax=10, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	# q_img = ax[0,1].imshow(Q[:,:,0,HIT],  cmap='Set2', vmin=0, vmax=10, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	# q_img = ax[1,1].imshow(Q[:,:,1,STAY], cmap='Set2', vmin=0, vmax=10, aspect="auto", extent=[11, 21, 1, 10]) # NO ACE, POS if HIT, neg if STAY
	# p_img = ax[0,1].imshow(policy[:,:,0], cmap='bwr', vmin=0, vmax=1, aspect="auto", extent=[11, 21, 1, 10])	# Useable Ace
	# p_img = ax[1,1].imshow(policy[:,:,1], cmap='bwr', vmin=0, vmax=1, aspect="auto", extent=[11, 21, 1, 10])	# No Useable Ace
	
	ax[0,0].set_ylabel("Dealer States")
	ax[1,0].set_ylabel("Dealer States")
	ax[1,0].set_xlabel("Player States")
	ax[1,1].set_xlabel("Player States")
	
	fig.colorbar(q_img, ax=ax.ravel().tolist())
	# fig.colorbar(p_img, cmap='bwr', ax=ax.ravel().tolist())
	
	fig, ax = plt.subplots(1)
	ax.plot(R)

	# plt.show()	

def TestFunctionality(game, Q, Q_dealer):
	print "Qs: ", Q.values.shape, Q_dealer.values.shape
	print "player:", game.player.hand
	print "dealer:", game.dealer.hand
	state = game.player.GetState( game.dealer.GetFirstCard().number )
	print "player state", state , "(D, P, A)"
	action = game.player.GetAction(state)
	print "player action", action
	print "Dealing\n"	
	game.DealToPlayer(dealer=False)

	print "player:", game.player.hand
	print "dealer:", game.dealer.hand
	print "player state", game.player.GetState( game.dealer.GetFirstCard().number ), "(D, P, A)"
	print game.player.IsBust(), game.outcome

	game.RestartGame()
	print "\nrestarting\n"
	print "Qs: ", Q.values.shape, Q_dealer.values.shape
	print "player:", game.player.hand
	print "dealer:", game.dealer.hand
	state = game.player.GetState( game.dealer.GetFirstCard().number )
	print "player state", state , "(D, P, A)"
	action = game.player.GetAction(state)
	print "player action", action
	print "Dealing\n"	
	game.DealToPlayer(dealer=False)

	print "player:", game.player.hand
	print "dealer:", game.dealer.hand
	print "player state", game.player.GetState( game.dealer.GetFirstCard().number ), "(D, P, A)"
	print game.player.IsBust(), game.outcome

	print "player:", game.player.hand
	print "dealer:", game.dealer.hand
	print "dealer state", game.dealer.GetState()
	print "dealer action", game.dealer.GetAction(game.dealer.GetState())

def main():

	np.set_printoptions(precision=3	)
	NUM_EPISODES = 10000
	PRINT_EVERY = 1000
	ALPHA = 0.01
	EPSILON = 0.99
	STAY = 0
	HIT = 1

	DEALER_STATES = 10 # (A, 2, 3, ... , 8, 9, 10/J/Q/K)
	PLAYER_STATES = 11 # (<11, 12, 13, ... 19, 20, 21)
	ACE_STATES = 2 # (has ace, no ace)

	states = (DEALER_STATES, PLAYER_STATES, ACE_STATES)
	actions = 2

	Q = StateSpace(states, actions, mean=1, var=0.1)
	Q_dealer = StateSpace((2,), actions)

	game = BlackJackGame(player_policy=QPolicy(Q, learn_rate=ALPHA, epsilon=EPSILON, static=False), dealer_policy=DealerPolicy(Q_dealer))
	
	# TestFunctionality(game, Q, Q_dealer)

	weighted_reward = 0
	rewards_list = []
	outcomes = np.zeros(6)
	# Play through a bunch of episodes
	for ii in range(NUM_EPISODES):

		episode = []

		# While the player still wants to draw cards, keep giving them cards
		while not game.player_stick:

			# Get the current state of the game
			dealer_face_up_card = game.dealer.GetFirstCard().number
			curr_state = game.player.GetState( dealer_card=dealer_face_up_card )

			# Grab the action the player will perform
			action = game.player.GetAction( curr_state )

			# If the player wants a new card
			if action:

				# Give the player a card, and get reward based on outcome
				game.DealToPlayer(dealer=False)
				reward = GetRewardFromOutcome(game.outcome)

				# Store the state, action, reward triplet in the episode
				episode.append( (curr_state, action, reward) )

				# If the player went bust, set player_stick to true
				if game.player.IsBust():
					game.player_stick = True	

			# If the player wants to stick, set player_stick to true and exit loop
			else:
				game.player_stick = True
			
		# If the player didn't go bust already, then its the dealer's turn
		if not game.outcome == BlackJackGame.IN_PROGRESS:
			game.dealer_stick = True
		
		# While the dealer still wants cards, keep giving them one
		while not game.dealer_stick:

			# Get the value of the dealer's hand (state)
			curr_dealer_state = game.dealer.GetState()

			# Get the action the dealer will take
			action = game.dealer.GetAction( curr_dealer_state )

			# If the dealer wants a new card, deal him one
			if action:
				game.DealToPlayer(dealer=True)
				if game.dealer.IsBust():
					game.dealer_stick = True
			else:
				game.dealer_stick = True
		
			# Grab the outcome of the game
			game.CheckOutcome()

			reward = GetRewardFromOutcome(game.outcome)
			episode.append( (curr_state, action, reward) )
		
		# Loop through the episode and learn from our decisions
		for S, A, reward in episode:
			state = S + (A,)
			game.player.ImprovePolicy(state, reward)

		final_reward = GetRewardFromOutcome(game.outcome)
		outcomes[game.outcome] += 1
		weighted_reward = (1-ALPHA) * weighted_reward + ALPHA*final_reward
		rewards_list.append((outcomes[1] + outcomes[2])/np.sum(outcomes))

		if ii % PRINT_EVERY == 0:
			# print weighted_reward
			print "win rate: ", (outcomes[1] + outcomes[2])/np.sum(outcomes)
			print "lose rate: ", (outcomes[3] + outcomes[4])/np.sum(outcomes)
			print "tie rate: ", (outcomes[5])/np.sum(outcomes)
			print ""
		game.RestartGame()
		del episode
		episode = []

		

	# 	# a = raw_input()

	P = game.player.policy.state_space.values
	PlotResults(P, rewards_list)

	plt.show()


if __name__=="__main__":
	main()