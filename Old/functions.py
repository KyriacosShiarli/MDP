import sys
import math
import numpy as np

class grid_environment():
  def __init__(self,size,maximum,minimum,other):
    assert size<6
    letters = ['a','b','c','d','e'] ; self.letters = letters[:size] 
    numbers = ['1','2','3','4','5'] ; self.numbers = numbers[:size]
    self.states = []
    for l in self.letters:
      for n in self.numbers:
        self.states.append(l+n)
    self.maximum = maximum
    self.minimum = minimum
    self.other = other
    self.actions =['north','south','east','west']
    self.reward_function = self.build_reward_function()
  def build_reward_function(self):
    reward_function = {}
    for s in self.states:
      if s =='a3':
        reward_function[s]=self.maximum
      elif s == 'b3':
        reward_function[s]=self.minimum
      else:
        reward_function[s]=self.other
    return reward_function
  def query_transition_function(self,state,action):
    wall_vertical = 0
    wall_horizontal = 0
    letters = self.letters
    numbers = self.numbers
    index1 = letters.index(state[0])
    if index1 == 0: wall_vertical = 1
    elif index1 == len(letters)-1: wall_vertical = -1
    
    index2 = numbers.index(state[1])  
    if index2 == 0: wall_horizontal = 1
    elif index2 == len(numbers)-1: wall_horizontal = -1
      
    next_state = {}
    stay = 0;
    if action == 'north':
      if wall_vertical != 1:
        next_state[letters[index1 - 1]+state[1]] = 0.8
      else:
        stay = 0.8
      if wall_horizontal == 0: next_state[state[0]+numbers[index2-1]] = 0.1 ; next_state[state[0]+numbers[index2+1]] = 0.1
      elif wall_horizontal == 1:
        stay += 0.1 ;
        next_state[state[0]+numbers[index2+1]] = 0.1
      elif wall_horizontal == -1:
        stay += 0.1
        next_state[state[0]+numbers[index2-1]] = 0.1
    elif action == 'south':
      if wall_vertical != -1: next_state[letters[index1 + 1]+state[1]] = 0.8
      else: stay = 0.8
      if wall_horizontal == 0: next_state[state[0]+numbers[index2-1]] = 0.1 ; next_state[state[0]+numbers[index2+1]] = 0.1
      elif wall_horizontal == 1: stay += 0.1 ; next_state[state[0]+numbers[index2+1]] = 0.1
      elif wall_horizontal == -1: stay += 0.1 ; next_state[state[0]+numbers[index2-1]] = 0.1
    elif action == 'east':
      if wall_horizontal != -1: next_state[state[0]+numbers[index2+1]] = 0.8
      else: stay = 0.8
      if wall_vertical == 0: next_state[letters[index1-1]+state[1]] = 0.1 ; next_state[letters[index1+1]+state[1]] = 0.1
      elif wall_vertical == 1: stay += 0.1 ; next_state[letters[index1 + 1]+state[1]] = 0.1
      elif wall_vertical == -1: stay += 0.1 ; next_state[letters[index1 - 1]+state[1]] = 0.1
    elif action == 'west':
      if wall_horizontal != 1: next_state[state[0]+numbers[index2-1]] = 0.8
      else: stay = 0.8
      if wall_vertical == 0: next_state[letters[index1-1]+state[1]] = 0.1 ; next_state[letters[index1+1]+state[1]] = 0.1
      elif wall_vertical == 1: stay += 0.1 ; next_state[letters[index1 + 1]+state[1]] = 0.1
      elif wall_vertical == -1: stay += 0.1 ; next_state[letters[index1 - 1]+state[1]] = 0.1
    next_state[state] = stay
    return next_state

class agent():
  def __init__(self,environent):
    self.environment = environment
    self.value_function = self.initialise_value()
    self.policy = {}
  def initialise_value(self):
    value_function = {}
    print 'states', self.environment.states
    for s in self.environment.states:
      value_function[s] = 0
    value_function['a3'] = self.environment.maximum
    value_function['b3'] = self.environment.minimum
    return value_function
  def value_of_action(self,action,state):
    next_states = self.environment.query_transition_function(state,action)
    value = self.environment.reward_function[state]
    for possible_state,probability in next_states.iteritems():
      value += probability * self.value_function[possible_state]
    return value
  def value_iteration(self):
      for i in range(10):
        print 'ITERATION', i 
        for state in self.environment.states:
          if state !='a3' and state !='b3':
            action_values = {}
            for act in self.environment.actions:
              action_values[act] = self.value_of_action(act,state)
            key,val = max_from_dict(action_values)
            self.value_function[state]=val
            self.policy[state] =key
        print self.value_function

def max_from_dict(dictionary):
  k = list(dictionary.keys())
  v = list(dictionary.values())
  m = max(v)
  return k[v.index(m)],m

environment = grid_environment(3,100,-100,-3)
ag = agent(environment)
ag.value_iteration()
print ag.policy
  
