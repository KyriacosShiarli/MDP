
from Discretisation import *
from forwardBackward import *
from Model import Model
from dataload import *
import matplotlib.pyplot as plt

def extractStatistics(examples,model):
	#performs forward backwward calculation for a reward function and extracts different kinds of statistics used to
	#evaluate the algorithm
	z_states = None
	feature_diff = 0
	feature_diff_avg=0
	likelihoods = np.zeros(len(examples))
	for num,example in enumerate(examples):
		action_probs,z_states =caus_ent_backward(model.transition_backward,model.transition_f,model.reward_f,example["end_state"],example["steps"],10,z_states)
		state_frequencies= smart_forward(action_probs,model.transition_forward,model.transition_f,example["start_state"],example["steps"])
		cumulative = 0
		log_policy = np.log(action_probs)
		# Gradient Computation - - - - - - - - - - - 
		for idx,i in enumerate(state_frequencies):
			cumulative += i*model.feature_f[idx]
		feature_diff +=example["feature_sum"]-cumulative
		#feature_diff_avg +=(example["feature_sum"]-cumulative)/example["steps"]
		# Action matching computation Should be adjusted for path length
		for n in xrange(example["steps"]-1):
			likelihoods[num] += log_policy[example["action_numbers"][n+1],example["state_numbers"][n]]
	feature_diff/=len(examples)
	return feature_diff,np.sum(likelihoods)
#Initialise discretisation model (theres only one at the mwoment)
disc_model = DiscModel()
# initialise weights
w = [-1.03497617,-0.04944345,-0.0291642,-9.21098972,-18.18630619,-1.47331647,-4.51922346,-7.44012957,-4.80524282,-0.06006514,-0.06949205,-1,-1.0064372]
#Initialise transition and reward model
model = Model(disc_model,w)
#Load data ->
print "Loading data"
examples = extract_info(disc_model,70)
#Run forward Backward algorithm for MaxEnt IRL
isayso = True
iteration=0
gamma = 0.02
batch = True
repeat = 30
grad_array= []
sum_grad_array= []
valid_array = []
sum_valid_array = []
z_actions = None
train = examples[:-5]
test = examples[-5:]
likelihoods_train_all = []
likelihoods_test_all = []
for iteration in xrange(repeat):
	#np.random.shuffle(sequence)
	grad_train,likelihood_train = extractStatistics(train,model)
	grad_array.append(grad_train)
	sum_grad_array.append(np.sum(np.absolute(grad_train)))
	likelihoods_train_all.append(likelihood_train)

	grad_test,likelihood_test = extractStatistics(test,model)
	valid_array.append(grad_test)
	sum_valid_array.append(np.sum(np.absolute(grad_test)))
	likelihoods_test_all.append(likelihood_test)
	

	if iteration%10 == 0:
		gamma=gamma/2
	print "Old W", model.w 	
	print "Gradient", grad_train
	print "Cross Validation Difference", grad_test
	model.w=model.w *np.exp(-gamma*grad_train)
	print "Train Likelihood"
	print "Test Likelihood"
	print "new W" , model.w
	model.reward_f = model.buildRewardFunction()
	print "THIS IS N",iteration

f = open("data3.py","w")
f.write("\nsum_grad=%s"%str(sum_grad_array))
f.write("\nsum_valid=%s"%str(sum_valid_array))
f.write("\nlik_train=%s"%str(likelihoods_train_all))
f.write("\nlik_test=%s"%str(likelihoods_test_all))
f.write("\nw_final=%s"%str(model.w))
	#x = np.arange(2)
	#f, axarr = plt.subplots(2, sharex=True)
	#axarr[0].scatter(x,np.around(grad_array,3),c = "r")
	#axarr[1].scatter(x,np.around(valid_array,3),c = "g")	
	#plt.show()