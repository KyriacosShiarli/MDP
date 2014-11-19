import numpy as np
import math
import RFeatures as rf

def discretiseQuantity_binary(quantity,disc_vector):
	# discretises quantity according to a discretisation vector and outputs a
	# binary vector of the same size as disc vector, indicating the binary
	typ = len(disc_vector.shape)
	print typ
	if typ == 1: # 1D discretisation vector of icreasing values
		size = disc_vector.shape[0]
		out_vector = np.zeros(size)
		for i in range(size):
			assert disc_vector[i]<=disc_vector[i+1] # 1D Discretisation vector should be increasing
			if i == 0 and quantity < disc_vector[i]:
				out_vector[i] = 1
				return out_vector
			elif i != (size-1) and quantity >= disc_vector[i] and quantity < disc_vector[i+1]:
				out_vector[i+1] = 1
				return out_vector
			elif i == (size - 1):
				out_vector[i] = 1
				return out_vector
	elif typ==2: #2D discretisation to be used for angles
		out_vector = np.zeros(disc_vector.shape[0])
		for i,j in enumerate(disc_vector):
			if quantity >= j[0] and quantity < j[1]:
				out_vector[i] = 1
		return out_vector

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
def parseVector(binary_vector,dims):
	assert len(binary_vector) == sum(dims)
	parsed = []
	prev = 0
	for i in dims:
		parsed.append(binary_vector[prev:i+prev])
		prev = i+prev
	return parsed


class Model(object):
	def __init__(self):
		distance = np.linspace(0,6.3,9) # Nine bins whatever the case
		linear = np.array([0,0.1,0.35]) # Linear velocity bins
		target = np.linspace(0,2*math.pi,8) # Targer orientation bins
		angular = np.array([-0.1,0,0.1]) # Angular velocity bins
		angle = np.linspace(0,2*math.pi,16) # Angle to persons bins
		actions_linear = [0,0.1,0.2]
		actions_angular = [0,0.1,0.2]
		self.actions = {"linear":actions_linear,"angular":actions_angular}
		self.bin_info = [angle,distance,target,angular,linear]
		self.dist_bins_per_angle = [9,9,6,6,3,3,3,3,3,3,3,3,6,6,9,9]
		self.get_dims()
		self.statesPerAngle()

	def get_dims(self):
		self.dims = []
		for j in self.bin_info:self.dims.append( len(j))
	def statesPerAngle(self):
		#calculates number of possible states after an orientation is known for all orientiations and returns
		# a vector. This is a direct result of the fact that we have variable length discretisations at different
		# orientations in order to save s
		self.states_per_angle = [] # number of states for each orientation
		print self.states_per_angle
		for i in range(self.dims[0]):
			self.states_per_angle.append(self.dist_bins_per_angle[i] * np.prod(self.dims[2::]))
	def enumerateState_binary(self,binary_vector):
		parsed = parseVector(binary_vector,self.dims)
		idx = np.zeros(len(parsed))
		for i,j in enumerate(parsed):
			idx[i] = np.nonzero(j)[0]
			if len(np.nonzero(j)) > 1: print "Error two ones in one feature"	
		state = sum(self.states_per_angle[0:int(idx[0])])
		for i in range(1,len(idx)):
			if i !=len(idx)-1:
				state+= idx[i] * np.prod(self.dims[i+1::])
			else:
				state+=idx[i]
		return state
	
	def quantityToBins(self,quantity_vector):
		assert len(quantity_vector) == len(self.bin_info)
		bins = []
		for n,k in enumerate(quantity_vector):
			if n == 1: 
				disc_vector = self.bin_info[n][:self.dist_bins_per_angle[bins[0]]]
			else: disc_vector = self.bin_info[n]
			bins.append(discretise(k,disc_vector))
		return bins

	def binsToQuantity(self,bins):
		quantity = []
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

def kinematics(state,action):
	state[1] = state[1]+action[0]*10
	state[2] = state[2]+action[1]*5

	state[4] = state[4]+action[0]*10
	state[2] = state[3]+action[1]*5
	return state

def buildTransitionFunction(model):
	tot_states =sum(model.states_per_angle)
	tot_actions = len(model.actions["linear"])*len(model.actions["angular"])
	transition_f = np.zeros([tot_actions,tot_states,tot_states])
	for i in range(tot_states):
		for j in range(tot_actions):
			bins = model.stateToBins(i)
			quantity = model.binsToQuantity(bins)
		 	angular_idx = int(math.floor(j/len(model.actions["linear"])))
		 	linear_idx = int(j%len(model.actions["linear"]))
			action = [model.actions["linear"][linear_idx],model.actions["angular"][angular_idx]]
			next_quantity = kinematics(quantity,action)
			next_bins = model.quantityToBins(next_quantity)
			next_state = model.binsToState(next_bins)
			transition_f[j,i,next_state] += 1
	row_sums = np.sum(transition_f,axis=2)
	#TODO Make nans failsafe
	print np.amax(row_sums)
	transition_f = transition_f/row_sums[:,np.newaxis]
	
	return transition_f
		#print i,bins
def buildFeatureFunction(model):
	tot_states = sum(model.states_per_angle)
	for i in range(tot_states):
		quantity = model.binsToQuantity(model.stateToBins(i))
		features = rf.staticGroupSimple(7,quantity[1],quantity[0],quantity[2])
		if i == 0: feature_f = np.zeros([tot_states,len(features)])
		feature_f[i,:] = features
	return feature_f



m = Model()
#tran = buildTransitionFunction(m)
feat = buildFeatureFunction(m)
for i in feat:
	print i


