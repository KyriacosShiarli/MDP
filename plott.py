import numpy as np
import data_30iterLearnedTransition70teps
import matplotlib.pyplot as plt
def plot_from_file(filename):
	f = open(filename,"r")
	for i in range(5):
		line = f.readline()
		exec line
	iterations = len(sum_grad)

	x = np.arange(iterations)
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(x,np.around(sum_grad,3),c = "r",label = "Train Gradient")
	axarr[0].plot(x,np.around(sum_valid,3),c = "g",label = "Test Gradient")
	axarr[0].set_ylabel("Gradient")
	axarr[0].set_title(filename)
	h2, = axarr[1].plot(x,np.around(lik_train,3),c = "r",label = "Train")
	h, = axarr[1].plot(x,np.around(lik_test,3),c = "g",label = "Test")	
	plt.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
	axarr[1].set_ylabel("-Log(Lik)")
	axarr[1].set_xlabel("Iterations")
	plt.show()

plot_from_file("data_30iternormalTransition70steps.py")
plot_from_file("data_30iterLearnedTransition70teps.py")
plot_from_file("data2.py")