import numpy as np

class Card(object):

	def __init__(self, suit=None, value=None):
		self.ID = np.random.randint(2**31)
		
		if not suit == None:
			self.suit = suit
		else:
			self.suit = np.random.randint(4)
		
		if not value == None:
			if value < 1 or value > 13:
				raise ValueError("Value of card must be between 1 and 13, inclusive")
			self.number = value
		else:
			self.number = np.random.randint(low=1, high=13+1)

		# Set flag if number is 1
		self.is_ace = (self.number == 1)

	def __repr__(self):
		return "{}".format(self.number)

class Deck():

	RANDOM = 0
	ACE_50_50 = 1
	ONE_TO_TEN = 2
	ONLY_TEN = 3

	def __init__(self):
		pass

	def DealCard(self, type = None):
		# If the caller wants a 50/50 chance of ACE
		if type == Deck.ACE_50_50:
			if np.random.rand() > 0.5:
				return Card(value=1)
			else:
				return Card( value = np.random.randint(low=1, high=10+1) )

		# If the caller wants equal chance of the 10 possible values
		elif type == Deck.ONE_TO_TEN:
			return Card( value = np.random.randint(low=1, high=10+1) )

		# If the caller wants a guarenteed 10
		elif type == Deck.ONLY_TEN:
			return Card(value=10)

		# If the caller wants a random card from Ace to King
		else:
			return Card( value = np.random.randint(low=1, high=13+1) )

	def DealCards(self, num, type=None):
		
		cards = []
		
		# Check if type is given and is correct length
		if not type == None:
			if not len(type) == num:
				raise ValueError("If not None, type must be a list of len num: {}".format(num))

			for ii in range(num):
				cards.append( self.DealCard(type=type[ii]) )
		
		else:
			for ii in range(num):
				cards.append( self.DealCard() )
				
		return cards