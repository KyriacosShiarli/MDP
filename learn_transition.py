from discretisationmodel import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataload import loadFile,loadFile2
def learn_tran():
	examples = loadFile2()
	examples2 = loadFile()
	tot_examples = examples + examples2
	disc = DiscModel()
	unorm_transition_f = np.zeros([disc.tot_actions,disc.tot_states,disc.tot_states])
	for example in tot_examples:
		for i in range(len(example["states"])-1):
			state = disc.quantityToState(example["states"][i])
			action = disc.actionToIndex(example["actions"][i])
			next = disc.quantityToState(example["states"][i+1])
			unorm_transition_f[action,state,next]+=1
	return unorm_transition_f

def loss_augmentation(amount):
	examples = loadFile()
	disc = DiscModel()
	loss_aug = np.ones([disc.tot_states])*amount
	for example in examples:
		for step in range(len(example["states"])):
			state = disc.quantityToState(example["states"][step])
			action = disc.actionToIndex(example["actions"][step])
			loss_aug[state] = 1
	return loss_aug

'''
examples_bad = loadFile2()
examples_good = loadFile()
disc = DiscModel()
feature_sums_good =np.zeros([13,len(examples_good)])
feature_sums_bad =np.zeros([13,len(examples_bad)])
for n,example in enumerate(examples_good):
	for i in range(100):
		feature_sums_good[:,n]+=disc.quantityToFeature(example["states"][i])

for n,example in enumerate(examples_bad):
	for i in range(100):
		feature_sums_bad[:,n]+=disc.quantityToFeature(example["states"][i])


fig = plt.figure()
ax = fig.add_subplot(111)

x = range(1,14)
for i in feature_sums_bad.T:
	h1 = ax.scatter(x,i,color = "red",alpha = 0.2)
for i in feature_sums_good.T:
	h2 = ax.scatter(x,i,color = "green",alpha = 0.2)
red_patch = mpatches.Patch(color='red', label='Bad',alpha = 0.5)
greed_patch = mpatches.Patch(color='green', label='Good',alpha = 0.5)
plt.legend([h1,h2],["bad","good"],bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
plt.show()
'''










