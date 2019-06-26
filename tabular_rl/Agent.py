class Agent(object):

	def __init__(self, policy, start_state, **kwargs):
		self.policy = policy
		self.ID = kwargs.get("ID", np.random.randint(2**32-1))
		self.training = kwargs.get("training", True)
        self.curr_state = start_state

    # Use our policy to get the action
	def GetAction(self, S):
		return policy.GetAction(S)

	def ImprovePolicy(self, exp_packet):

        # Only improve policy if training flag is set
		if self.training:

            # Grab elements out of our exp_packet
			S_list, A_list, R_list, n = exp_packet.Get()

            # grab the current value of the state
			V = self.policy.vals[S_list[0], A_list[0]]

            # Calculate the new target based on the exp_packet
			G = self.policy.GetTargetEstimate(exp_packet)

            # Increment towards the new target based on learning rate
            new_val = (1-self.alpha) * V + self.alpha * G

            # Update the value of the state and return True
			self.policy.UpdateState(S_list[0], A_list[0], new_val)
			return True

        # Training flag not set, return false
        else:
			return False

    def UpdateCurrentState(self, new_state):
        self.curr_state = new_state
