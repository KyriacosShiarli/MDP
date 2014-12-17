import numpy as np
import math
import string
from discretisationmodel import *
from RFeatures import *
import matplotlib.pyplot as plt

class ExampleStruct(object):
	def __init__(self):
		self.states = []
		self.actions = []
		self.state_numbers = []
		self.action_numbers = []
		self.trajectory_number = 0
		self.start = 0
		self.goal = 0
		self.feature_sum = 0
		self.state_action_counts = []
		self.info = ["distance","angle"]


def loadFile():
	f = open("data/training_samples.txt",'r')
	examples = []
	while True:
		line =f.readline()
		if line == '':
			break
		ex = ExampleStruct()
		for i in line.replace(' ','').split('||')[1:]:
			al = i.split('|')
			if map(float,al[0].split(','))[1]<50:
				ex.states.append(map(float,al[0].split(',')))
				ex.actions.append(map(float,al[1].split(',')))
		ex.states = np.array(ex.states)
		ex.actions = np.array(ex.actions)
		examples.append(ex)
	return examples

def loadFile2():
	f = open("data/training_samples-bad_behavior-withOrientations.txt",'r')
	examples = []
	while True:
		line =f.readline()
		if line == '':
			break
		ex = ExampleStruct()
		for i in line.replace(' ','').split('||')[1:]:
			al = i.split('|')
			if map(float,al[0].split(','))[1]<50:
				ex.states.append(map(float,al[0].split(',')))
				ex.states[len(ex.states)-1].pop(2)
				ex.actions.append(map(float,al[1].split(',')))
		ex.states = np.array(ex.states)
		ex.actions = np.array(ex.actions)
		examples.append(ex)

	return examples

def loadFile3():
	f = open("data/UPO/Experiments Folder/2014-11-17 11.08.31 AM/training_samples-closest_person.txt",'r')
	examples = []
	order = []
	while True:
		line =f.readline()
		if line == '':
			break
		ex = ExampleStruct()
		new = line.split(".bag")
		ex.trajectory_number = int(new[0].replace('traj',''))
		for i in new[1].replace(' ','').split('||')[1:]:
		#for i in new[].replace(' ','').split('||')[1:]:
			al = i.split('|')
			if map(float,al[0].split(','))[1]<50:
				ex.states.append(map(float,al[0].split(','))[:4])
		#		ex["states"][len(ex["states"])-1].pop(2)
				ex.actions.append(map(float,al[1].split(',')))
		ex.states = np.array(ex.states)
		ex.actions = np.array(ex.actions)
		examples.append(ex)
	def getKey(item):
		return item.trajectory_number
	examples = sorted(examples,key = getKey)
	return examples

def extract_info(disc_model,num_samples,examples_type = "good"):
	if examples_type == "good":
		examples = loadFile() 
	else:
		examples = loadFile2()
	
	for example in examples:
		example.state_action_counts = np.zeros([disc_model.tot_actions,disc_model.tot_states])
		if num_samples =="Full":
			length = len(example.states)
		else:
			length = num_samples
		example.feature_sum = 0;example.state_numbers = [];example.action_numbers = [];example.steps =length

		for i,state in enumerate(example.states[:length]):
			state_number = disc_model.quantityToState(state)
			example.state_numbers.append(state_number)
			example.action_numbers.append(disc_model.actionToIndex(example.actions[i]))
			
			example.state_action_counts[example.action_numbers[i],state_number]+=1
			if i == 0:
				example.start = state_number
			if i == length-1:
				example.goal = state_number
			example.feature_sum += disc_model.quantityToFeature(state,example.actions[i])
	return examples
if __name__ =="__main__":
	examples = loadFile3()
	
	
	d = DiscModel()
	e = extract_info(d,70,"good")
	for ex in e:
		print ex.feature_sum
	#print examples
	#x=5
	#m = DiscModel()
	#examples = extract_info(m,70,"bad")
	#plt.imshow(dist)d
	#plt.show()
	#for example in examples:
	#	print example["feature_sum"]
	#summ = 0 
	#for i,j in enumerate(examples[0]["states"]):
	#	out = staticGroupSimple(j,examples[0]["actions"][i],duration = 0.1)
	#	out2 = examples[0]["states"][i+1]
	#	print "STATE=",j
	#	print "Action",examples[0]["actions"][i]
	#	print out
	#	print out2



