import numpy as np
import math
import string
from Discretisation import *
from RFeatures import *
from kinematics import *
def loadFile():
	f = open("training_samples.txt",'r')
	examples = []
	while True:
		line = f.readline()
		if line == '':
			break
		ex = {"states":[],
			  "actions":[]		
			}
		x = line.split("]]")
		for i in x[0:-1]:
			y = i.replace(' ','')
			y = i.replace('[[','').replace('[','').replace(']','').split(',')[1::]
			y = map(float,y)
			ex["states"].append(y[0:4])
			ex["actions"].append(y[4:6])
		examples.append(ex)
	return examples
def loadFile2():
	f = open("data/training_samples.txt",'r')
	examples = []
	while True:
		line =f.readline()
		if line == '':
			break
		ex = {
			"states":[],
			"actions":[]
			}
		for i in line.replace(' ','').split('||')[1:]:
			al = i.split('|')
			ex["states"].append(map(float,al[0].split(',')))
			ex["actions"].append(map(float,al[1].split(',')))
		examples.append(ex)
	return examples
def extract_info(disc_model,num_samples):
	examples = loadFile2()
	for example in examples:
		if num_samples =="Full":
			length = len(example["states"])
		else:
			length = num_samples
		example["feature_sum"] = 0
		example["state_numbers"] = []
		example["action_numbers"] = []
		example["steps"] =length
		for i,state in enumerate(example["states"][:length]):
			state_number = disc_model.quantityToState(state)
			example["state_numbers"].append(state_number)
			example["action_numbers"].append(disc_model.actionToIndex(example["actions"][i]))
			if i == 0:
				example["start_state"] = state_number
			if i == length-1:
				example["end_state"] = state_number
			example["feature_sum"] += disc_model.quantityToFeature(state)
	return examples
if __name__ =="__main__":
	m = DiscModel()
	examples = extract_info(m,200)
	summ = 0 
	for i,j in enumerate(examples[0]["states"]):
		out = staticGroupSimple(j,examples[0]["actions"][i],duration = 0.1)
		out2 = examples[0]["states"][i+1]
		print "STATE=",j
		print "Action",examples[0]["actions"][i]
		print out
		print out2



