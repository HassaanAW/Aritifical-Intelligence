import numpy as np
from numpy import *
#import matplotlib.pyplot as plt
from scipy.special import expit
import sys
import time 
import matplotlib.pyplot as plt

# Code for plotting graph of Execution time and LR
# times = [58.2, 54.6, 54.2, 50.8]
# lr = [0.0002, 0.002, 0.005, 0.01]
# fig,plots = plt.subplots()
# plots.plot(lr, times)
# plots.set(xlabel = 'Learning Rate', ylabel='Time Taken to Train (5 epochs)', title = 'Learning Rate vs Execution Time')
# plots.grid()
# fig.savefig("plot.png")

# Code for plotting graph of Accuracy and LR
# acc = [78.3, 72.3, 70.8 ]
# lr = [0.002, 0.005, 0.01]
# fig,plots = plt.subplots()
# plots.plot(lr, acc)
# plots.set(xlabel = 'Learning Rate', ylabel='Accuracy (%) on Test Data', title = 'Learning Rate vs Accuracy')
# plots.grid()
# fig.savefig("plot_acc.png")

# Delete netWeights.txt before training Data as otherwise data will be appended to previous data and results will be in accurate

# matrix of size 60000x784
def plotter(time, acc, lr):
	fig,plots = plt.subplots()
	plots.plot(time, acc)
	plots.set(xlabel = 'Time', ylabel='Accuracy (%)', title = 'Time vs Accuracy on LR =' + str(lr) )
	plots.grid()
	fig.savefig(str(lr)+".png")

def cost_entropy(actual,obtained):
	change = np.sum(-actual * np.log(obtained))
	return change

def sigmoid(x):
	return expit(x)
    #return 1/(1+np.exp(-x))

def deriv_sigmoid(z):
	return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
	exp_z = np.exp(z)
	return exp_z / exp_z.sum(axis = 1, keepdims=True)

def get_labels(nameone):
	file1 = open(nameone, 'r')
	data = file1.read()
	labels = np.fromstring(data, dtype = int, sep = '\n')
	return labels
	

def get_data(nameone, nametwo):
	file1 = open(nameone, 'r')
	data = file1.read()
	data = data.replace('[','')
	data = data.replace(']','')
	matrix = np.fromstring(data, dtype = int, sep = ' ')
	file1.close()
	matrix = matrix.reshape(-1, 784)

	file2 = open(nametwo, 'r')
	data = file2.read()
	labels = np.fromstring(data, dtype = int, sep = '\n')
	file2.close()
	
	return matrix, labels

def normalize(data):
	for i in range(data.shape[0]):
		get_mean = np.mean(data[i])
		get_dev = np.var(data[i])
		data[i] = (data[i]-get_mean)/get_dev
	return data

def make_one_hot(labels):
	one_hot = np.eye(10)[labels]
	return one_hot

def get_indexes(obtained):

	max_index = obtained.argmax()
	return max_index

def get_accuracy(obtained, actual):
	matches = 0
	count = 0
	for i in range(len(actual)):
		count = count + 1
		if(obtained[i] == actual[i]):
			matches = matches + 1
	acc = (matches/count)*100
	return acc, matches

def store_weights(mat1, mat2):
	matt =  mat1.reshape(1,-1) # 23520
	l1 = matt.tolist()
	file1 = open('netWeights.txt', 'a+')
	for each in l1:
		file1.write(str(each))
	file1.close	
	
	# second_weight = second_weight.reshape(1,-1)
	# np.savetxt('netWeights', second_weight)
	matt2 = mat2.reshape(1,-1) # 300
	l2 = matt2.tolist()
	file1 = open('netWeights.txt', 'a+')
	for each in l2:
		file1.write(str(each))
	file1.close

def get_weights(weights):
	file1 = open(weights, 'r')
	data = file1.read()
	concat = ''
	for char in data:
		if(char != ']'):
			concat = concat + char
		else:
			concat = concat + ']'
			break
	second = ''
	for i in range(len(concat), len(data)):
		second = second + data[i]
	file1.close()
	concat = concat.replace('[','')
	concat = concat.replace(']','')
	concat = concat.replace(',','')
	first_matrix = np.fromstring(concat, dtype = float, sep = ' ')
	first_matrix = first_matrix.reshape(784,30)
	#print(first_matrix.shape)

	second = second.replace('[','')
	second = second.replace(']','')
	second = second.replace(',','')
	second_matrix = np.fromstring(second, dtype = float, sep = ' ')
	second_matrix = second_matrix.reshape(30, 10)
	#print(second_matrix.shape)
	#print(second_matrix)
	return first_matrix, second_matrix




def main():

	prog_type = sys.argv[1]
	if(prog_type == "train"):
		data_file = sys.argv[2]
		label_file = sys.argv[3]
		rate = sys.argv[4]
		rate = float(rate)


		train_file = data_file
		train_label = label_file

		print("Loading Data set")
		matrix, labels = get_data(train_file, train_label) # 60000x784
		print("Loading complete")
	 
		one_hot =  make_one_hot(labels) # 60000x10
	
		first_layer = 784
		hidden_layer = 30
		output = 10

		first_weight = 2 * np.random.rand(first_layer,hidden_layer) -1 #784x30
		second_weight = 2 * np.random.rand(hidden_layer,output) -1 #30x10
		#matrix = normalize(matrix).reshape(60000, 784)

		time_arr=[]
		acc_arr =[]
		start = time.time() # measure training time
		for epoch in range(5):
			max_arr = []
			for image in range(matrix.shape[0]): # image[i] = 1x784
				first_comb = np.dot(matrix[image], first_weight) # first linear combination of layer one 1x30
				#print(first_comb)
				first_output = sigmoid(first_comb).reshape(1,30) # output of sigmoid 1x30
				# print(first_output)

				second_comb = np.dot(first_output, second_weight) # second linear combination of layer two 1x10
				second_output = sigmoid(second_comb).reshape(1,10) # output of second activation 1x10
				# print(second_output)
				# This finishes our first forward propogation
				max_index = get_indexes(second_output)
				max_arr.append(max_index)
				# print(max_index)
				# accuracy = get_accuracy(max_index, labels)
				# Back Propogate from here
				diff = second_output - one_hot[image] # get difference between actual values and predicted values after initial forward propogation. 60000x10
				# diff 1x10
				diff = diff.reshape(1,10)
				delta = np.dot(first_output.T, diff).reshape(30,10) # delta = 30x10

				deriv_sig = deriv_sigmoid(first_comb).reshape(1,30) #1x30
				delta_two = np.dot(diff, second_weight.T).reshape(1,30) # resultant = 1x30
				resultant = deriv_sig * delta_two # Multiply the result obtained from derivative of sigmoid function with first linear comb as input and the dot of difference matrix and second set of weights
				# 1x30 ^
				final = np.dot(matrix[image].reshape(1,784).T, resultant).reshape(784,30) # 784x30

				# Now apply gradient descent algorithm
				# weights = weights - @ * Partial J(W)/ Partial (Wj)

				first_weight = first_weight - (rate*final)
				second_weight = second_weight - (rate*delta)

				#change = cost_entropy(one_hot, second_output)
			accuracy, matches = get_accuracy(max_arr,labels)
			print("Epoch",epoch + 1)
			print("Correctly Classified", matches, "/", len(labels), "images")
			print("Accuracy:", accuracy, "%")
			print("Error:", 100-accuracy, "%")
			end = time.time()
			time_arr.append(end-start)
			acc_arr.append(accuracy)

		plotter(time_arr, acc_arr, rate)
		# print(time_arr)
		# print(acc_arr)
		#print("Execution time with rate", rate, ":", end-start, "s")
			# print("--------------------")
		# save weights now
		# np.savetxt('netWeights.txt',first_weight, delimiter = '\n') #784x30
		store_weights(first_weight, second_weight)
	


	elif(prog_type == "test"):
		data_file = sys.argv[2]
		label_file = sys.argv[3]
		weight_file = sys.argv[4]

		first_layer = 784
		hidden_layer = 30
		output = 10
	
		weight_one, weight_two = get_weights(weight_file)
		Test_file = data_file
		Test_label_file = label_file
		Test_matrix, Test_labels = get_data(Test_file, Test_label_file) # 10000x784
		
		test_one_hot = make_one_hot(Test_labels)
		max_test = []
		for image in range(Test_matrix.shape[0]):
			first_comb = np.dot(Test_matrix[image], weight_one) # first linear combination of layer one 1x30
			#print(first_comb)
			first_output = sigmoid(first_comb).reshape(1,30) # output of sigmoid 1x30
			# print(first_output)

			second_comb = np.dot(first_output, weight_two) # second linear combination of layer two 1x10
			second_output = sigmoid(second_comb).reshape(1,10) # output of second activation 1x10
			# print(second_output)
			# This finishes our first forward propogation
			max_index = get_indexes(second_output)
			max_test.append(max_index)
		new_accuracy, matches = get_accuracy(max_test,Test_labels)
		print("On Test Data")
		print("Correctly Classified", matches, "/", len(Test_labels), "images")
		print("Accuracy:", new_accuracy, "%")
		print("Error:", 100-new_accuracy, "%")


main()







