import numpy as np
import math
import string
from discretisationmodel import *
from RFeatures import *
import matplotlib.pyplot as plt
import itertools
from functions import *

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

def action_filter(actions):
	for n,action in enumerate(actions):
		if n ==0: 
			if np.absolute(action[0]) > math.pi and np.absolute(actions[n+1][0]) < math.pi:
				action[0] = (0+actions[n+1][0])/2
			elif np.absolute(action[0]) > math.pi and np.absolute(actions[n+1][0]) > math.pi:
				action[0] = 0
			if np.absolute(action[1]) > 10 and np.absolute(actions[n+1][1]) < 10:
				action[1] = (0+actions[n+1][1])/2
			elif np.absolute(action[1]) > 10 and np.absolute(actions[n+1][1]) > 10:
				action[1] = 0
		elif n != len(actions)-1:
			
			if np.absolute(action[0]) > math.pi and np.absolute(actions[n+1][0]) < math.pi:
				action[0] = (actions[n-1][0]+actions[n+1][0])/2
			elif np.absolute(action[0]) > math.pi and np.absolute(actions[n+1][0]) > math.pi:
				action[0] = (actions[n-1][0]+actions[n-2][0])/2


			if np.absolute(action[1]) > 10 and np.absolute(actions[n+1][1]) < 10:
				action[1] = (actions[n-1][1]+actions[n+1][1])/2
			elif np.absolute(action[1]) > 10 and np.absolute(actions[n+1][1]) > 10:
				action[1] = (actions[n-1][1]+actions[n-2][1])/2
				
		else:
			if np.absolute(action[0]) > math.pi:
				action[0] = (actions[n-1][0]+actions[n-2][0])/2

			if np.absolute(action[1]) > 10:
				action[1] = (actions[n-1][1]+actions[n-2][1])/2
	return actions			

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
				ex.states.append(map(float,al[0].split(',')[:2]))
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
				ex.states.append(map(float,al[0].split(',')[:2]))
				#ex.states[len(ex.states)-1].pop(2)
				ex.actions.append(map(float,al[1].split(',')))
		ex.states = np.array(ex.states)
		ex.actions = np.array(ex.actions)
		examples.append(ex)

	return examples

def loadFile3(name):
	def examples_from_stream(f):
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
				if map(float,al[0].split(','))[1]<50000:
					ex.states.append(map(float,al[0].split(','))[:2])
			#		ex["states"][len(ex["states"])-1].pop(2)
					ex.actions.append(map(float,al[1].split(',')))
			ex.states = np.array(ex.states)
			temp = subsample(ex.states[:,0],4)
			temp = np.vstack([temp,subsample(ex.states[:,1],4)])			
			ex.states = np.transpose(temp)
			ex.actions = np.array(ex.actions)
			ex.actions = action_filter(ex.actions)
			temp = subsample(ex.actions[:,0],4)
			temp = np.vstack([temp,subsample(ex.actions[:,1],4)])
			ex.actions = np.transpose(temp)
			examples.append(ex)
			ex.states.shape
			ex.actions.shape
		def getKey(item):
			return item.trajectory_number
		examples = sorted(examples,key = getKey)
		plt.show()
		return examples		
	examples_p1 = examples_from_stream(open(name +"training_samples-person1.txt",'r'))
	x = range(examples_p1[2].states.shape[0])
	examples_p2 = examples_from_stream(open(name +"training_samples-person2.txt",'r'))
	#examples_target = examples_from_stream(open(name +"training_samples-person3.txt",'r'))
	#plt.scatter(x,examples_p1[2].states[:,0],color = "red")
	for example1, example2 in itertools.izip_longest(examples_p1,examples_p2):
		example1.states =  c_o_m([example1.states,example2.states])
		#example1.states = np.vstack(example1.states,examples_target.states)
	

		#example1.com =  np.array(com)
		#example1.states[:,0] =np.array(map(angle_full_to_half, avg_ang))
		#example1.states[:,1] = (example1.states[:,1]+example2.states[:,1])/2
		#example1.states[:,0] = (example1.states[:,0]+example2.states[:,0])/2
	
	#plt.scatter(x,examples_p1[2].states[:,0],color = "green")
	#plt.scatter(x,examples_p2[2].states[:,0],color = "blue")
	#plt.scatter(x,map(angle_half_to_full,examples_p1[2].states[:,0]),color = "yellow")
	#plt.scatter(x,examples_p1[2].com[0,:],color = "black")
	#plt.show()
	return examples_p1


def extract_info(disc_model,num_samples,examples_type = "good"):
	if examples_type == "good":
		examples = loadFile3("data/UPO/Experiments Folder/2014-11-17 11.08.31 AM/")
		examples2 = loadFile3("data/UPO/Experiments Folder/2014-11-28 01.22.03 PM/") 
		examples = examples[:12]+examples2[:12]
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
	#examples = loadFile3("data/UPO/Experiments Folder/2014-11-17 11.08.31 AM/")
	#for example in examples:
	#	print "TRAJECTORY NUMBER------------->", example.trajectory_number
		
	
	
	d = DiscModel()
	e = extract_info(d,100,"good")
	su = 0
	for ex in e:
		su+= ex.feature_sum
	print su
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



