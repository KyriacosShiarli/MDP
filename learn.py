from dataload import *
from Discretisation import *
from forwardBackward import *
from Model import *
#Initialise discretisation model (theres only one at the mwoment)
disc_model = DiscModel()
# initialise weights
w = [0.3,0.1]
#Initialise transition and reward model
groupModel = Model(disc_model,w)
#Load data ->
print "Loading data"
examples = extract_info(disc_model)
#Run forward Backward algorithm for MaxEnt IRL
isayso = True
z_states = None
n=1
while isayso:
	sequence = np.arange(13)
	np.random.shuffle(sequence)
	for example in sequence:
		action_probs,z_states =backward(groupModel.transition_f,groupModel.reward_f,examples[example]["end_state"],0.5,z_states)
		state_frequencies= forward(action_probs,groupModel.transition_f,examples[example]["start_state"],examples[example]["steps"],)
		cumulative = 0
		print "SumStateFrieq", np.sum(state_frequencies)
		for n,i in enumerate(state_frequencies):
			cumulative += i*groupModel.feature_f[n]
		print "Old W", groupModel.w 
		gamma = 0.1
		#groupModel.w = groupModel.w +0.5*(examples[example]["feature_sum"]-cumulative)
		groupModel.w=groupModel.w * np.exp(gamma*(examples[example]["feature_sum"]-cumulative))
		groupModel.reward_f = groupModel.buildRewardFunction()
		print "Feature Sum",examples[example]["feature_sum"]
		print "Cumulative",cumulative
		print "Gradient", examples[example]["feature_sum"]-cumulative
		print "new W" , groupModel.w
	n+=1