from Grid import *
import matplotlib.pyplot as plt

timing = []
number = 10
numStates = np.zeros(number)
y1 = np.zeros(number)
y2 = np.zeros(number)
ar = np.arange(10,50,5)
for i,j in enumerate(ar):
	layout = [j,j]
	obstacle = j*j/3
	goal = j*j - 1
	env = grid_environment(layout,obstacle,goal)
	freq,t = forward_backard(env.transition_f,env.reward_f,0,goal,10)
	timing.append(t)
	numStates[i] = j*j
	y1[i] = t["Backward"]
	y2[i] = t["Forward"]
print y2

plt.subplot(2,1,1)
plt.scatter(numStates,y1)
plt.subplot(2,1,2)
plt.scatter(numStates,y2,c = 'r')
plt.show()