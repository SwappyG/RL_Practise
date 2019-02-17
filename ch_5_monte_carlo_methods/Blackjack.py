import numpy as np
import matplotlib.pyplot as plt

# --- CONSTANTS ---

# States
ACE_STATES = 2
DEALER_STATES = 10
PLAYER_STATES = 1 + 10
STATES = PLAYER_STATES * DEALER_STATES * ACE_STATES

# Action related constants
ACTION_COUNT = 2
STAY = 0
HIT = 1

# Policy and Reward constants
GAMMA = 1	
EPSILON = 0.99
ALPHA = 0.01

# Result constants
WIN = 1
DRAW = 0
LOSS = -1
LOSE_STATE = PLAYER_STATES - 1

OUTCOMES = 5
LOW_SCORE = 0
HIGH_SCORE = 1
EQUAL_SCORE = 2
PLAYER_BUST = 3
DEALER_BUST = 4


# Runtime constants
CHECK_EVERY = 20000
CHECK_COUNT = 1000
RUN_PERIOD = 200000

# Contains an episode of data (state, action, reward)
class Episode:
	def __init__(self):
		self.state = []
		self.action = []
		self.reward = []

# Converts a hand of cards to corresponding state
def cards_to_state(cards, is_dealer=False):
	
	if is_dealer:
		return cards[0] - 1
	
	else:
		# Get the total of the cards
		sum = np.sum(cards)
		
		# If any card is an ace and sum is less than 12, ace is useable
		a_state = 1 if ( np.any(np.array(cards) == 1) and sum < 12 ) else 0
		
		# If ace is useable, set its value to 11 points
		if a_state:
			sum += 10
		
		# All sums less than 11 are treated as state 0
		# Sums 12-21 are mapped to states 1-10
		# Any sum above 21 is mapped to state 11
		p_state = np.maximum(0, np.minimum(PLAYER_STATES-1, sum - 11))
		
		return p_state, a_state

# Start a new game and deal 2 cards to player and dealer		
def deal_cards(random_start=False):
	
	# If random start flag is set, choose a random state with equal prob
	if random_start:
		d_state = np.random.randint(DEALER_STATES)
		a_state = np.random.randint(ACE_STATES)
		p_state = np.random.randint(PLAYER_STATES - 1 + a_state)
		
		# if has an ace
		if a_state:
		
			# if p_state == 0, ie p_cards sum < 12
			if p_state == 0:
				second_card = np.random.randint(1,11) # random card between 1 and 10
			else:
				second_card = p_state
				
			p_cards = [1, second_card] # Ace and something else
			
		# if no ace
		else:
		
			if p_state == 10:
				raise ValueError
		
			if p_state == 0:
				first_card = np.random.randint(2,10)
				p_cards = [first_card, 11 - first_card]
			else:
				p_cards = [p_state+1, 10]	# p_cards sum = p_state + 11
			
		# exposed card is d_state+1, second card is random card between 1 and 10
		d_cards = [d_state+1, np.random.randint(1,11)]
		
	# Otherwise, choose state with prob based on an infinite deck
	else: 
	
		# create two random card (numbers between 1 and 13) for dealer and player
		d_cards = np.random.randint(1, 14, size=2)
		p_cards = np.random.randint(1, 14, size=2)
		
		# Values of cards 11-13 are all 10
		d_cards[d_cards > 10] = 10
		p_cards[p_cards > 10] = 10
		
		
		# Map the cards to the states
		d_state = cards_to_state(d_cards, is_dealer=True)
		p_state, a_state = 	cards_to_state(p_cards)
	
	return d_cards, p_cards, d_state, p_state, a_state

# Deal a single new card (suit doesnt matter, only value)
def deal_card():
	
	# generate a random number between 1-13; convert 11-13 to 10
	return np.minimum(10, np.random.randint(1, 14))

# Randomly decide to HIT or STAY with equal probability
def random_policy():
	
	# randomly decide to HIT or STAY
	return np.random.choice((STAY,HIT))

# Return the score of the hand given the cards
def get_score(cards, is_dealer=False):
	
	# Check if there's an ace in the hand
	has_ace = np.any( np.array(cards) == 1 )
	
	# Get the sum of cards, with any ace as 11 points
	sum = np.sum(cards) + 10*(has_ace)
	
	# If bust and has an ace, make ace worth 1 pt
	if has_ace and sum > 21:
		sum -= 10
		
	return sum

def get_reward(D, P):

	if (P > 21):	return LOSS, PLAYER_BUST
	elif (D > 21):	return WIN, DEALER_BUST
	elif (D > P):	return LOSS, LOW_SCORE
	elif (P > D):	return WIN, HIGH_SCORE
	elif P == D:	return DRAW, EQUAL_SCORE
	else:						
		raise ValueError # This Should never happen
	
# Run a single episode, either on a policy or with a random policy	
def run_episode(on_policy=False, policy=None, hard_policy=False):
	
	# Deal out cards to start the game
	d_cards, p_cards, d_state, p_state, a_state = deal_cards(random_start=True)
	
	# Store state and action in our episode
	first_state = (d_state, p_state, a_state)
	p_score = get_score(p_cards)
	
	if p_score == 21:
		d_score = get_score(d_cards)
		reward, outcome = get_reward(d_score, p_score)
		return first_state, STAY, reward, outcome
	
	# Use provided policy to get action if hard policy, or with prob EPSILON if soft
	if on_policy and ( np.random.rand() < EPSILON or hard_policy ):
		first_action = policy[first_state]
	else:
		first_action = random_policy()
	
	action = first_action
	
	# While the policy keeps coming up as HIT
	while action == HIT:
	
		# print(f"Cards P/D {p_cards} / {d_cards}, States: {state}, P HIT")
		
		# Draw new cards, update state, and get player score
		p_cards = np.append(p_cards, deal_card())	# Draw a new card
		p_state, a_state = cards_to_state(p_cards)	# Get the new state of the player
		p_score = get_score(p_cards)	# Get the player score
		
		state = (d_state, p_state, a_state)	# Collect the states into a tuple
		
		# if player is in the LOSE_STATE, give -1 reward and end episode
		if p_score > 21:
			# print(f"Cards P/D {p_cards} / {d_cards}, States: {state}, BUST")
			reward = -1
			outcome = PLAYER_BUST
			return first_state, first_action, reward, outcome
		
		# Use provided policy to get action if hard policy, or with prob EPSILON if soft
		if on_policy and ( np.random.rand() < EPSILON or hard_policy ) :
			action = policy[state]
		else:
			action = random_policy()
					
	# print(f"Cards P/D {p_cards} / {d_cards}, States: {p_state, d_state, a_state}, P STAY")
	
	# Get the current score of the dealer
	d_score = get_score(d_cards, is_dealer=True)	
		
	# If the dealer's score is less than 17
	while d_score < 17:
		# print(f"Cards P/D {p_cards} / {d_cards}, States: {state}, D HIT")
		d_cards = np.append(d_cards, deal_card())	# Draw a new card
		d_score = get_score(d_cards, is_dealer=True)	# Update score
		
	# print(f"Cards P/D {p_cards} / {d_cards}")
	
	reward, outcome = get_reward(d_score, p_score)
	
	return first_state, first_action, reward, outcome

# Run a bunch of episodes to check win-rate with provided policy
# If no policy is provided, a random policy will be used as baseline
def check_win_rate(on_policy=False, policy=None):

	wins = 0
	losses = 0
	games = CHECK_COUNT
	
	outcomes = np.zeros(OUTCOMES, dtype=int)
	
	# For number of episodes to run check on
	for t in range(CHECK_COUNT):
	
		# Run an episode on hard policy
		_,_,reward, outcome = run_episode(on_policy=on_policy, policy=policy, hard_policy=True)
		
		# Keep running total of wins and losses
		wins += 1 if reward == WIN else 0
		losses += 1 if reward == LOSS else 0
	
		outcomes[outcome] += 1
	
	return 100.0*wins/games, 100.0*losses/games, outcomes

# Plot the Q function and the current policy
def plot_Q_policy(Q, policy):
	
	fig, ax = plt.subplots(2,2)
	
	q_img = ax[0,0].imshow(Q[:,:,0,HIT] - Q[:,:,0,STAY],  cmap='Set2', vmin=-2, vmax=2, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	q_img = ax[1,0].imshow(Q[:,:,1,HIT] - Q[:,:,1,STAY], cmap='Set2', vmin=-2, vmax=2, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	
	# q_img = ax[0,0].imshow(Q[:,:,0,HIT],  cmap='Set2', vmin=0, vmax=10, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	# q_img = ax[1,0].imshow(Q[:,:,0,STAY], cmap='Set2', vmin=0, vmax=10, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	# q_img = ax[0,1].imshow(Q[:,:,0,HIT],  cmap='Set2', vmin=0, vmax=10, aspect="auto", extent=[11, 21, 1, 10]) # ACE, POS if HIT, neg if STAY
	# q_img = ax[1,1].imshow(Q[:,:,1,STAY], cmap='Set2', vmin=0, vmax=10, aspect="auto", extent=[11, 21, 1, 10]) # NO ACE, POS if HIT, neg if STAY
	p_img = ax[0,1].imshow(policy[:,:,0], cmap='bwr', vmin=0, vmax=1, aspect="auto", extent=[11, 21, 1, 10])	# Useable Ace
	p_img = ax[1,1].imshow(policy[:,:,1], cmap='bwr', vmin=0, vmax=1, aspect="auto", extent=[11, 21, 1, 10])	# No Useable Ace
	
	ax[0,0].set_ylabel("Dealer States")
	ax[1,0].set_ylabel("Dealer States")
	ax[1,0].set_xlabel("Player States")
	ax[1,1].set_xlabel("Player States")
	
	fig.colorbar(q_img, ax=ax.ravel().tolist())
	fig.colorbar(p_img, cmap='bwr', ax=ax.ravel().tolist())
	plt.show()
		
	
# Train a policy using Q function to learn from experience	
def main():
	
	# Initialize Q function and count of times any (S,A) pair has been encountered
	Q = np.random.normal(loc=0.0, scale=0.1, size=(DEALER_STATES, PLAYER_STATES, ACE_STATES, ACTION_COUNT))
	Q[:,0,:,HIT] = 1
	Q[:,-1,0,STAY] = 1
	N = np.zeros_like(Q)
	
	# The policy is the value action (A) for any given state (S) that maximizes Q(S,A)
	policy = np.argmax(Q, axis=3)
	
	outcome_counts = np.zeros(OUTCOMES, dtype=int)
	
	episode_count = 1
	keep_training = True
	
	# Keep training until the user says to stop
	while keep_training:
		
		# print(f"Ran {episode_count} episodes")
		
		# Run an episode on a soft policy
		state, action, reward, outcome = run_episode(on_policy=True, policy=policy)
				
		outcome_counts[outcome] += 1
		
		# Get the starting states and action, increment N(S,A)
		s1,s2,s3 = state
		a = action
		N[s1,s2,s3,a] += 1
		
		# Update Q(S,A) by averaging all G(S,A) seen so far (incremental update)
		# Q[s1,s2,s3,a] += (G - Q[s1,s2,s3,a]) / N[s1,s2,s3,a]
		Q[s1,s2,s3,a] = (1 - ALPHA) * Q[s1,s2,s3,a] + ALPHA * reward
		
		# Update the policy
		policy[s1,s2,s3] = np.argmax( Q[s1,s2,s3,:] )
		
		# Every so often, check current win and lose rate
		if episode_count % CHECK_EVERY == 0:
			
			win_rate, lose_rate, outcomes = check_win_rate(on_policy=True, policy=policy)
			print(f"Ran {episode_count//1000}k eps, W/L: {win_rate:.1f} / {lose_rate:.1f}, O: {outcomes}")

		# Every so often, ask the user if they want to keep training
		if episode_count % RUN_PERIOD == 0:
			
			EPSILON = 1 - 0.9**(episode_count/RUN_PERIOD)
			
			plot_Q_policy(Q, policy)
			
			# If the user presses n, stop, otherwise keep going
			# user_input = input("\nPress n to stop training, or anything else to keep going\n")
			# if user_input == "n":
				# keep_training = False
		
		episode_count += 1
	
		# input("Press a key to continue\n\n")
	
	# Print final w/l rates
	win_rate, lose_rate = check_win_rate(on_policy=True, policy=policy)
	print(f"\n\n-------------------------------\nRan {episode_count} episodes, FINAL W/L: {win_rate:.2f} / {lose_rate:.2f}\n-------------------------------\n")
	plot_Q_policy(Q, policy)
	
if __name__=="__main__":
	main()
	
	
	
	
	
	
	
# for t in reversed(range(len(episode.reward)-1)):
			
	# G = episode.reward[t+1]
	
	# s1,s2,s3 = episode.state[t]
	# a = episode.action[t]
	# N[s1,s2,s3,a] += 1
	# Q[s1,s2,s3,a] += (G - Q[s1,s2,s3,a]) / N[s1,s2,s3,a]
	
	# policy[s1,s2,s3] = np.argmax( Q[s1,s2,s3,:] )