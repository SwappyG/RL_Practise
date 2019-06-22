import numpy as np

class StateSpace(object):
	def __init__(self, states, actions, dyn_func = None, mean = 0, var = 0):
		self.dyn_func = dyn_func
		if dyn_func == None:
			self.values = np.random.normal( mean, var, size=states + (actions,) )
			self.updates = np.zeros(states + (actions,) )
		else:
			self.values = np.random.normal( mean, var, size=states )
			self.updates = np.zeros(states )

	def GetExpectedStateValue(self, state, action):
		return self.dyn_func(self.values, state, action)


	def GetBestAction(self, state):
		best_value = None
		best_action = None
		for action in self.actions:
			value = GetExpectedStateValue(self, state, action)

			if value > best_value or best_value == None:
				best_action = action
				best_value = value

		return best_action

	def UpdateState(self, state, value):
		self.values[state] = value
		self.updates[state] += 1

class Policy(object):

	def __init__(self, state_space, learn_rate = 0.01, epsilon = 0.1, static = True):
		self.state_space = state_space
		self.learn_rate = learn_rate
		self.epsilon = epsilon
		self.improvements = 0
		self.static = static

	def ImprovePolicy(self, state, reward):
		raise NotImplementedError("Any class implementing the Policy class must implement an ImprovePolicy function")

	def GetAction(self):
		raise NotImplementedError("Any class implementing the Policy class must implement a GetAction function")

class QPolicy(Policy):

	def __init__(self, state_space, learn_rate = 0.01, epsilon = 0.1, static=False):
		Policy.__init__(self, state_space, learn_rate, epsilon, static = static)
		self.improvements = 0

	def _UpdateEpsilon(self):
		pass
		# if self.improvements > 10000:
		# 	self.improvements = 0
		# 	self.epsilon = np.minimum(1, self.epsilon+0.1)

	def ImprovePolicy(self, state, reward):

		# If the policy is static, don't change anything and just return
		if self.static:
			return False
		else:

			# Get the current value, and nudge it closer to our reward
			val = self.state_space.values[state]
			val = (1-self.learn_rate)*val + self.learn_rate*reward

			# Update the state with the new value
			self.state_space.UpdateState(state, val)

			# Keep track of the number of policy iterations and update epsilon if needed
			self.improvements += 1
			self._UpdateEpsilon()

			return True

	def GetState(self, player_val, useable_ace=None, dealer_card=None):
		player_state = np.maximum( player_val - 11,  0 )
		ace_state = 1 if useable_ace else 0
		dealer_state = np.minimum( dealer_card, 10) - 1
		return (dealer_state, player_state, ace_state)

	def GetAction(self, state):

		# Check if we're doing a random action based on epsilon
		do_random = np.random.rand() > self.epsilon

		# If we are, grab a random action from our choices
		if do_random:
			actions = len(self.state_space.values[state])
			return np.random.choice(actions)

		# If we're not doing random, pick the best action for our state
		else:
			return np.argmax(self.state_space.values[state])

class VPolicy(object):

	def __init__(self, state_space):
		self.state_space = state_space

	def GetAction(self, state):
		return self.state_space.GetBestAction(state)


class DealerPolicy(Policy):

	HIT = 1
	STAY = 0

	def __init__(self, state_space):
		Policy.__init__(self, state_space, static=True)
		self.state_space = state_space

	def GetState(self, value, **kwargs):
		return DealerPolicy.STAY if (value > 17) else DealerPolicy.HIT

	def GetAction(self, state):
		if state == DealerPolicy.HIT:
			return 1
		else:
			return 0
