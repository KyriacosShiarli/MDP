import numpy as np
import math

def discretise(quantity,disc_vector):
	#Returns the bin in which the quantity falls according to the discretisation vector
	typ = len(disc_vector.shape)
	if typ == 1: # 1D discretisation vector of icreasing values
		size = disc_vector.shape[0]
		for i in range(size):
			#assert disc_vector[i]<=disc_vector[i+1] # 1D Discretisation vector should be increasing
			if i == 0 and quantity < disc_vector[i]:
				return i
			elif i != (size-1) and quantity >= disc_vector[i] and quantity < disc_vector[i+1]:
				return i
			elif i == (size - 1):
				return i
	elif typ==2: #2D discretisation to be used for angles
		for i,j in enumerate(disc_vector):
			if quantity >= j[0] and quantity < j[1]:
				return i
class DiscModel(object): # Discretisation for non uniform polar discretisation
	def __init__(self):
		distance = np.linspace(0,3,9) # Nine bins whatever the case
		linear = np.array([0,0.1,0.35]) # Linear velocity bins
		#target = np.linspace(0,2*math.pi,8) # Targer orientation bins
		angular = np.array([-0.1,0,0.1]) # Angular velocity bins
		angle = np.linspace(-math.pi,math.pi,17)[0:16] # Angle to persons bins
		actions_linear = [0,0.5,0.25]
		self.bin_info = [angle,distance,angular,linear]
		self.dist_bins_per_angle = [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]
		self.get_dims()
		self.statesPerAngle()
	def get_dims(self):
		self.dims = []
		for j in self.bin_info:self.dims.append( len(j))
		print self.dims
	def statesPerAngle(self):
		#calculates number of possible states after an orientation is known for all orientiations and returns
		# a vector. This is a direct result of the fact that we have variable length discretisations at different
		# orientations in order to save s
		self.states_per_angle = [] # number of states for each orientation
		print self.states_per_angle
		for i in range(self.dims[0]):
			self.states_per_angle.append(self.dist_bins_per_angle[i] * np.prod(self.dims[2::]))

	def quantityToBins(self,quantity_vector):
		assert len(quantity_vector) == len(self.bin_info)
		bins = []
		for n,k in enumerate(quantity_vector):
			if n == 1: 
				disc_vector = self.bin_info[n][:self.dist_bins_per_angle[bins[0]]]
			else: disc_vector = self.bin_info[n]
			bins.append(discretise(k,disc_vector))
		return bins
	def binsToQuantity(self,bins,sample = False):
		quantity = []
		if sample==True:
			for n,k in enumerate(bins):
				if k != (len(self.bin_info[n]) - 1):
					quantity.append(np.random.uniform(self.bin_info[n][k],self.bin_info[n][k+1]))
				else:
					diff =self.bin_info[n][k] - self.bin_info[n][k-1]  
					quantity.append(np.random.uniform(self.bin_info[n][k], self.bin_info[n][k]+diff))
		else:
			for n,k in enumerate(bins):
				quantity.append(self.bin_info[n][k])
		return quantity
	def binsToState(self,bin_numbers):
		state = sum(self.states_per_angle[0:int(bin_numbers[0])])
		if bin_numbers[1]>self.dist_bins_per_angle[bin_numbers[0]]-1: print "Illegal state based on model";return None
		for i in range(1,len(bin_numbers)):
			if i !=len(bin_numbers)-1:
				state+= bin_numbers[i] * np.prod(self.dims[i+1::])
			else:
				state+=bin_numbers[i]
		return state
		#Determine where you have ones.
	def stateToBins(self,state):
		#first determine direction.
		bins = []
		for i,dim in enumerate(self.states_per_angle):
			state -=  dim 
			if state < 0:
				state = state + dim
				bins.append(i)
				break
		for i in range(2,len(self.dims)):
			bins.append(math.floor(state/np.prod(self.dims[i::])))
			state = state%np.prod(self.dims[i::])
		bins.append(state)
		if bins[1]>self.dist_bins_per_angle[bins[0]] - 1:
			print "Mistake in calculation. REview Function"
			print bins
			return None
		else:
			return bins
	def quantityToState(self,quantity):
		bins = self.quantityToBins(quantity)
		return self.binsToState(bins)

