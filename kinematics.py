import numpy as np
import math
# AXIS INFORMATION:
#                                      | +x
#									   |
#									   |
#								+	   |    -
#									   |
#									   |
#                     +y ------------------------------------ - y
#									   |
##									   |
#								+	   |    -
#									   |
#									   | - x
									   
# ------------------------------------------------------------------------------------------------------


def staticGroupSimple(state,action,duration = 0.1):
	def degug():
		#action = [angular,linear]
		print "dx",dx
		print "dy",dy
		print "dx_tar",dx_tar
		print "dy_tar",dy_tar
		print "STATE------>",state
		print "out------>",out
		print "action----->", action
	x_vel_init = state[3]*math.cos(state[2])
	y_vel_init = state[3]*math.sin(state[2])
	x_vel_fin = action[1]*math.cos(action[0])
	y_vel_fin = action[1]*math.sin(action[0])
	dx = (x_vel_init+ x_vel_fin)*duration/2
	dy = (y_vel_init+ y_vel_fin)*duration/2
	dx_tar = state[1]*math.cos(state[0])
	dy_tar = state[1]*math.sin(state[0])
	diff_x = dx_tar - dx
	diff_y = dy_tar - dy
	if (action[1]==0 and state[1]==0) or (diff_x==0 and diff_y==0):
		orient =state[0] - action[0]*duration
	else:
		if (diff_x>0 and diff_y>0) or (diff_x >0 and diff_y<0):
			ang =math.atan((diff_y)/(diff_x))
		elif diff_x<0 and diff_y>0:
			ang = math.pi + math.atan((diff_y)/(diff_x))
		elif diff_x <0 and diff_y<0:
			ang = math.atan((diff_y)/(diff_x)) - math.pi
		elif diff_x ==0 and diff_y<0:
			ang = -math.pi/2
		elif diff_x == 0 and diff_y>0:
			ang = math.pi/2
		elif diff_x >0 and diff_y ==0:
			ang = 0
		elif diff_x <0 and diff_y==0:
			ang = -math.pi
		orient =ang - action[0]*duration
	while orient > math.pi or orient < -math.pi:
		orient -=math.copysign(2*math.pi,orient) 
	dist = math.sqrt((diff_x)**2 + (diff_y)**2)
	out = np.zeros(len(state))
	out[0] = orient ; out[1] = dist; out[2] = action[0] ; out[3] = action[1]
	return out

def staticGroupSimple2(state,action,duration = 0.03):
	def degug():
		#action = [angular,linear]
		print "dx",dx
		print "dy",dy
		print "dx_tar",dx_tar
		print "dy_tar",dy_tar
		print "STATE------>",state
		print "out------>",out
		print "action----->", action
	x_vel_fin = action[1]*math.cos(action[0])
	y_vel_fin = action[1]*math.sin(action[0])
	dx =(x_vel_fin)*duration
	dy = (y_vel_fin)*duration
	dx_tar = state[1]*math.cos(state[0])
	dy_tar = state[1]*math.sin(state[0])
	diff_x = dx_tar - dx
	diff_y = dy_tar - dy
	if (action[1]==0 and state[1]==0) or (diff_x==0 and diff_y==0):
		orient =state[0] - action[0]*duration
	else:
		if (diff_x>0 and diff_y>0) or (diff_x >0 and diff_y<0):
			ang =math.atan((diff_y)/(diff_x))
		elif diff_x<0 and diff_y>0:
			ang = math.pi + math.atan((diff_y)/(diff_x))
		elif diff_x <0 and diff_y<0:
			ang = math.atan((diff_y)/(diff_x)) - math.pi
		elif diff_x ==0 and diff_y<0:
			ang = -math.pi/2
		elif diff_x == 0 and diff_y>0:
			ang = math.pi/2
		elif diff_x >0 and diff_y ==0:
			ang = 0
		elif diff_x <0 and diff_y==0:
			ang = -math.pi
		orient =ang - action[0]*duration
	while orient > math.pi or orient < -math.pi:
		orient -=math.copysign(2*math.pi,orient) 
	dist = math.sqrt((diff_x)**2 + (diff_y)**2)
	out = np.zeros(len(state))
	out[0] = orient ; out[1] = dist; #out[2] = action[0] ; out[3] = action[1]
	return out