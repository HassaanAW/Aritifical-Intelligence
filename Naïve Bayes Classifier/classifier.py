import numpy as np
import sys

def Gather_info(name_one):
	file1 = open(name_one, 'r') 
	count = 0
	normal = 0
	abnormal = 0
	Lines = file1.readlines()
	normal_data = []
	abnormal_data = []

	for i in range(len(Lines)):
		count = count + 1
		Lines[i] = Lines[i].strip()
		if(Lines[i][0] == '1'):
			normal = normal + 1
			normal_data.append( list( Lines[i].split(',') ) )
		elif(Lines[i][0] == '0'):
			abnormal = abnormal + 1
			abnormal_data.append( list(Lines[i].split(',')) )
	# print(normal_data)
	# print(abnormal_data)

	# We have gathered all the information here and placed it into appropriate lists and arrays. 
	# There are two final arrays of size (40x23) representing Normal and Abnormal data
	Normal_array = np.array(normal_data)
	Normal_array = Normal_array.reshape(normal,23)
	Abnormal_array = np.array(abnormal_data)
	Abnormal_array = Abnormal_array.reshape(abnormal,23)

	Normal_prior = normal/count
	Abnormal_prior = abnormal/count

	print("Starting Training on", count, "plots")
	return  Normal_prior, Abnormal_prior, Normal_array, Abnormal_array
	# print(Normal_array.shape)
	# print(Abnormal_array.shape)

def Test_info(name_two):
	file1 = open(name_two, 'r')
	count = 0
	Lines = file1.readlines()
	Listform  = []
	for i in range(len(Lines)):
		count = count + 1
		Lines[i] = Lines[i].strip()
		Listform.append( list( Lines[i].split(',') ) )

	Test_array = np.array(Listform)
	Test_array = Test_array.reshape(count, 23)

	# get true results
	actual_results = []
	for i in range(count):
		actual_results.append(Test_array[i][0])
	Test_data = np.delete(Test_array, 0, 1)
	# This does not have the results in it
	print("Testing on", count, "plots")
	return actual_results, Test_data, count


def Normal_cal( normal_arr ):
	#print(normal_arr.shape)
	true_tests= []
	false_tests = []
	for col in range(1,23):
		ones = 0
		zeros = 0
		for row in range(40):
			if normal_arr[row][col] == '1':
				ones = ones + 1
			elif normal_arr[row][col] == '0':
				zeros = zeros + 1
		true_tests.append(ones)
		false_tests.append(zeros)
	# print(true_tests)
	# print(false_tests)

	true_given_Normal = np.array(true_tests)
	false_given_Normal = np.array(false_tests)
	
	# converted both into probabilities 

	true_given_Normal = true_given_Normal/40
	false_given_Normal = false_given_Normal/40

	return true_given_Normal, false_given_Normal

def Abnormal_cal( abnormal_arr ):
	#print(abnormal_arr.shape)
	true_tests= []
	false_tests = []
	for col in range(1,23):
		ones = 0
		zeros = 0
		for row in range(40):
			if abnormal_arr[row][col] == '1':
				ones = ones + 1
			elif abnormal_arr[row][col] == '0':
				zeros = zeros + 1
		true_tests.append(ones)
		false_tests.append(zeros)
	# print(true_tests)
	# print(false_tests)

	true_given_Abnormal = np.array(true_tests)
	false_given_Abnormal = np.array(false_tests)

	# converted both into probabilities 
	true_given_Abnormal = true_given_Abnormal/40
	false_given_Abnormal = false_given_Abnormal/40

	return true_given_Abnormal, false_given_Abnormal

def Normal_class(Normal_prior, true_given_Normal, false_given_Normal, Test_data, Test_count):
	perc_matrix = []
	for rows in range(Test_count):
		multiply = 1
		for col in range(22):
			if Test_data[rows][col] == '1':
				multiply = multiply * true_given_Normal[col]
			elif Test_data[rows][col] == '0':
				multiply = multiply * false_given_Normal[col]
		final = multiply * Normal_prior
		perc_matrix.append(final)
	return perc_matrix

def Abnormal_class(Abnormal_prior, true_given_Abnormal, false_given_Abnormal, Test_data, Test_count):
	perc_matrix = []
	for rows in range(Test_count):
		multiply = 1
		for col in range(22):
			if Test_data[rows][col] == '1':
				multiply = multiply * true_given_Abnormal[col]
			elif Test_data[rows][col] == '0':
				multiply = multiply * false_given_Abnormal[col]
		final = multiply * Abnormal_prior
		perc_matrix.append(final)
	return perc_matrix

def Compare_both(Normal_matrix, Abnormal_matrix, Test_count):
	Predicted = []
	for i in range(Test_count):
		if(Normal_matrix[i] > Abnormal_matrix[i]):
			Predicted.append('1')
		elif(Normal_matrix[i]< Abnormal_matrix[i]):
			Predicted.append('0')
	return Predicted

def Final_Comp(Predicted, results, Test_count):
	error = 0
	correct = 0
	for i in range(Test_count):
		if(Predicted[i] == results[i]):
			correct = correct + 1
		else:
			error = error + 1
	Accuracy = (correct / Test_count) * 100
	print("Total Accuracy:", Accuracy, "%")
	
	return Accuracy

def main():
	try:
		name_one = sys.argv[1]+ ".txt"
		name_two = sys.argv[2]+ ".txt"
	except:
		pass

	print("------------------------------------------")
	Normal_prior, Abnormal_prior, Normal_array, Abnormal_array = Gather_info(name_one)
	print("Training Complete")
	print(".")
	print(".")
	print(".")
	true_given_Normal, false_given_Normal = Normal_cal(Normal_array)
	true_given_Abnormal, false_given_Abnormal = Abnormal_cal(Abnormal_array)
	# we have now extracted all information from our training data. This includes:
	# -> Normal prior and abnormal prior
	# -> Probababilites of each test with test = 1 and test = 0 given Normal and Abnormal
	results, Test_data, Test_count = Test_info(name_two)
	Normal_matrix = Normal_class(Normal_prior, true_given_Normal, false_given_Normal, Test_data, Test_count)
	Abnormal_matrix = Abnormal_class(Abnormal_prior, true_given_Abnormal, false_given_Abnormal, Test_data, Test_count)
	Predicted = Compare_both(Normal_matrix, Abnormal_matrix, Test_count)
	Accuracy = Final_Comp(Predicted, results, Test_count)
	print("------------------------------------------")





main()
	


