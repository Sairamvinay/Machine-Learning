import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


TRAIN_SAMPLES = 292	#training data number of samples
TEST_SAMPLES = 100 #test data number of samples
ALL_SAMPLES = 392  #NUMBER OF ALL SAMPLES

# Problem 1: helper function to remove the missing value records
def remove_missing():

	features = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','name']	#need feature names for accessing columns
	df = pd.read_csv("auto-mpg.data",names = features,sep = '\s+')	#read the data file which has only spaces as seperator token
	df = df[df["horsepower"] != '?']	#to clean up the missing values in the data set
	df["horsepower"] = df["horsepower"].astype(float)	#convert to float so that no error occurs while computation
	return df

#Problem 1: to split and regroup the data set based on dividing the mpg values into 4 diff categories acc to thresholds;
def split_ranges(df):
	'''
	Logic is to sort data frame based on mpg values, then divide all into 4 quarter of original data frame for categories 
	and then obtain the final data frame with added cat column
	'''
	
	df = df.sort_values(by = ["mpg"])	#sort all the mpg values so that it is easy to split into categories
	num_samples = len(df["mpg"])	#need it to split up the samples into bins
	
	print("The thresholds for each quarter are: ")

	print("minimum value is %f" % min(df["mpg"]))	#the minimum value of mpg

	df_first_qtr = df.iloc[:int(num_samples * 0.25)]	#the first threshold is found and the first bin is saved
	print("first quartile: ",df_first_qtr["mpg"].iloc[-1])
	df_first_qtr["cat"] = "low"
	
	df_second_qtr = df.iloc[int(num_samples * 0.25):int(num_samples * 0.5)] #the second threshold is found and the second bin is saved
	print("second quartile: ",df_second_qtr["mpg"].iloc[-1])
	df_second_qtr["cat"] = "med"
	
	df_third_qtr = df.iloc[int(num_samples * 0.5):int(num_samples * 0.75)]	#the third threshold is found and the third bin is saved
	print("third quartile: ",df_third_qtr["mpg"].iloc[-1])
	df_third_qtr["cat"] = "high"
	
	df_fourth_qtr = df.iloc[int(num_samples * 0.75):]	#the fourth bin is saved
	df_fourth_qtr["cat"] = "vhigh"

	print("maximum value is %f" % max(df["mpg"]))	#the maximum value of mpg
	

	#Showing equally sized bins
	print("Equally sized bins:\n")
	print("First bin size: ",len(df_first_qtr))
	print("Second bin size: ",len(df_second_qtr))
	print("Third bin size: ",len(df_third_qtr))
	print("Fourth bin size: ",len(df_fourth_qtr))


	dfnew = pd.concat([df_first_qtr,df_second_qtr,df_third_qtr,df_fourth_qtr])	#join these bins by rows into new dataframe; have the new category column included
	dfnew = dfnew.sort_index()	#to get back the old order of our data frame (sort by index)
	return dfnew

#Problem 2: generates the scatter matrix plot for each of features (except mpg and name)
def scatter_generate(df):

	print("Problem 2 Solution\n")
	colors = {'low':'red','med':'blue','high':'green','vhigh':'yellow'}	#need to identify the colors
	c = []
	for i,cat in enumerate(df["cat"]):
		c.append(colors[cat])	#get the color encoding for that particular label
		


	dfcopy = df.drop(["cat","mpg"],axis = 1)	#drop off the mpg column so as to get the plot

	pd.plotting.scatter_matrix(dfcopy,alpha = 1,c = c,figsize = (8,8),diagonal = 'hist')
	plt.show()

#Problem 4 helper: A grapher method to graph each of the predictions based on the linear regression solver for different degrees
def grapher(zeroW,firstW,secondW,thirdW,testX,testY,feature):

	fig,ax = plt.subplots()	#create multiple subplots

	ax.scatter(testX[:,0], testY, c='black',label = "data points")	#scattered data
	X = np.linspace(min(testX),max(testX),TEST_SAMPLES)	#Create a line space of ordered values
	X = X.reshape((TEST_SAMPLES,1))	#reshape to get 2D array
	ones = np.ones(TEST_SAMPLES).reshape((TEST_SAMPLES,1))	#all ones vector
	zeroPred = np.matmul(zeroW.T,ones.T)	#for zero degree predictions zeropred = W.T * 1
	firstPred =	np.matmul(firstW.T,np.concatenate((ones,X),axis = 1).T)		#firstpred = firstW[0] + (firstW[1] * X)
	twoPred = np.matmul(secondW.T,np.concatenate((ones,X,X**2),axis = 1).T)	#secondW[0] + (secondW[1] * X) + (secondW[2] * (X**2))
	threePred = np.matmul(thirdW.T,np.concatenate((ones,X,X**2,X**3),axis = 1).T) #thirdW[0] + (thirdW[1] * X) + (thirdW[2] * (X ** 2)) + (thirdW[3] * (X**3))

	ax.plot(X,zeroPred.T,color = 'red',label = "zero degree")
	ax.plot(X,firstPred.T,color = 'blue',label = "first degree")
	ax.plot(X,twoPred.T,color = "green",label = "second degree",linestyle = '--')
	ax.plot(X,threePred.T,color = "orange",label = "third degree")
	ax.legend(loc='upper right')
	plt.xlabel(feature)
	plt.ylabel("MPG")
	plt.title("MPG vs " + feature)
	plt.show()


#Problem 3: A Linear Regression solver for different degrees. It calculates weights accordingly to different weights
def Linear_regressionSolver(feature_train = None,feature_test = None,Y_train = None,Y_test = None,degree = 1,reportMSE = True):

	'''
	First, the X matrix (X_totrain for the training data and X_totest for the testing data) is 
	constructed according to the degree provided

	feature_train,feature_test,Y_train and Y_test are all pandas data frames.
	'''

	X_totrain = np.ones(len(feature_train)).reshape((len(feature_train),1))	#create ones vector according to the shape
	X_totest = np.ones(len(feature_test)).reshape((len(feature_test),1))

	if degree == 0:
		pass	#do nothing to modify the X matrix as per equation W = inv(X_t X) * X_t * Y

	elif degree == 1:

		
		X_totrain = np.concatenate((X_totrain,feature_train),axis = 1)	#add the feature vector as another column
		X_totest = np.concatenate((X_totest,feature_test),axis = 1)

	elif degree == 2:

		Xsquaretrain = feature_train ** 2
		
		Xsquaretest = feature_test ** 2
		
		X_totrain = np.concatenate((X_totrain,feature_train,Xsquaretrain),axis = 1) #add the feature vector and its square as another columns
		X_totest = np.concatenate((X_totest,feature_test,Xsquaretest),axis = 1)

	elif degree == 3:

		Xsquaretrain = feature_train ** 2
		Xcubetrain = feature_train ** 3

		Xsquaretest = feature_test ** 2
		Xcubetest = feature_test ** 3

		X_totrain = np.concatenate((X_totrain,feature_train,Xsquaretrain,Xcubetrain),axis = 1)	#add the feature vector and its square and its cube as another columns
		X_totest = np.concatenate((X_totest,feature_test,Xsquaretest,Xcubetest),axis = 1)

	else:
		print("Error: Wrong degree entered")	#doesnt work for non-negative degrees beyond 3


	Weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_totrain.T,X_totrain)),X_totrain.T),Y_train)	# W = inv(X_t X) * X_t * Y
	Weights = Weights.reshape((Weights.shape[0],1))	#make it a vector (2D array)
	
	trainpred = np.matmul(Weights.T,X_totrain.T)	#get the train predictions
	testpred = np.matmul(Weights.T,X_totest.T)	#get the test prediction

	trainpred = np.array(trainpred.T).ravel()	#make it a 1D vector
	testpred = np.array(testpred.T).ravel()	

	if Y_test.empty:
		return testpred,Weights 	#if the true Ytest labels are not provided or empty: then return the predictions and weights


	#if MSE is to be reported for the problem

	if reportMSE:
		trainMSE = np.mean((trainpred - Y_train) ** 2)
		testMSE = np.mean((testpred - Y_test) ** 2)
		print("The training Mean square error for the %sth order solver is %s"%(degree,trainMSE))
		print("The testing  Mean square error for the %sth order solver is %s\n"%(degree,testMSE))
	
		
	
	return testpred,Weights

#Problem 5 solution
def regressAll(df_all):

	print("Problem 5 Solution\n")
	dfcopy = df_all.drop(labels = ["cat","mpg","name"],axis = 1)	#drop it so that no need to predict with these unneccessary values

	X = dfcopy.to_numpy()	#makes it a numpy Nd array of all features

	trainX,testX = X[:TRAIN_SAMPLES],X[TRAIN_SAMPLES:]
	Y = df_all["mpg"]
	trainY = Y[:TRAIN_SAMPLES]
	testY = Y[TRAIN_SAMPLES:]

	print("For regression against all features")
	zerothTestPred,_ = Linear_regressionSolver(feature_train = trainX, feature_test = testX, Y_train = trainY, Y_test = testY, degree = 0)
	firstTestPred,_  = Linear_regressionSolver(feature_train = trainX, feature_test = testX, Y_train = trainY, Y_test = testY, degree = 1)
	secondTestPred,_ = Linear_regressionSolver(feature_train = trainX, feature_test = testX, Y_train = trainY, Y_test = testY, degree = 2)


#Problem 4: this method regresses against each feature seperately and calls grapher accordingly
def regressEach(df_all):
	
	print("Problem 4 Solution\n")
	columns = df_all.columns.tolist()	#get all columns
	columns.remove("cat")	#drop all the unneccessary columns
	columns.remove("name")
	columns.remove("mpg")
	
	Y = df_all["mpg"]
	trainY = Y[:TRAIN_SAMPLES]
	testY = Y[TRAIN_SAMPLES:]
	
	print("For regression against each feature independently")
	for col in columns:

		print("For the %s feature\n"%col)
		
		X = df_all[col]


		trainX = np.array(X[:TRAIN_SAMPLES]).reshape((TRAIN_SAMPLES,1))
		testX  = np.array(X[TRAIN_SAMPLES:]).reshape((TEST_SAMPLES,1))
		
		#ignore the predictions as we want only the weights here
		#call for each degree
		
		_,zerothW= Linear_regressionSolver(feature_train = trainX,feature_test = testX, Y_train = trainY, Y_test = testY,degree = 0)
		_,firstW =  Linear_regressionSolver(feature_train = trainX,feature_test = testX, Y_train = trainY, Y_test = testY,degree = 1)
		_,secondW = Linear_regressionSolver(feature_train = trainX,feature_test = testX, Y_train = trainY, Y_test = testY,degree = 2)
		_,thirdW =  Linear_regressionSolver(feature_train = trainX,feature_test = testX, Y_train = trainY, Y_test = testY,degree = 3)
		
		grapher(zerothW,firstW,secondW,thirdW,testX,testY,col)	#invoke grapher each time
		print('\n')


#Problem 6: Trains and predicts using the Logistic Regression SKLearn model
#scale_down is for Qn 7 (see method below) and only_model for Qn 8

def LogRegression(df_all,scale_down = False,only_model = False):
	
	
	clf = LogisticRegression()
	if only_model:
		return clf 		#returns only the model without fitting; helpful for question 8


	Y_train,Y_test = df_all["cat"][:TRAIN_SAMPLES],df_all["cat"][TRAIN_SAMPLES:]
	dfcopy = df_all.drop(labels = ["cat","mpg","name"],axis = 1)

	X = dfcopy.to_numpy()	#get Nd array of all features
	
	#if needed to scale_down all values

	if scale_down == True:
		scaler = MinMaxScaler()
		scaler.fit(X)
		X = scaler.transform(X)
		print("After Scaling\n")


	X_train,X_test = X[:TRAIN_SAMPLES],X[TRAIN_SAMPLES:]

	clf.fit(X_train,Y_train)

	print("Logistic Regression Algorithm results\n")
	print("Training Accuracy is %f%%"%(clf.score(X_train,Y_train) * 100.00))
	print("Testing Accuracy is %f%%\n"%(clf.score(X_test,Y_test) * 100.00))


#Problem 7: it is same as problem but with scaling.
def ScaleRegress(df_all):
	
	print("Problem 7 Solution\n")
	LogRegression(df_all,scale_down = True)	#call Log Reg with scaling


#Problem 8: Test one sample feature on the entire data set
def testSample(df,X_test):

	print("Problem 8 Solution\n")
	Y_train_Lin = df["mpg"]	#keep the outputs seperate
	Y_train_Cat = df["cat"]
	dfcopy = df.drop(labels = ["cat","mpg","name"],axis = 1)

	X_train = dfcopy.to_numpy()
	mpgpred,_ = Linear_regressionSolver(feature_train = X_train.reshape((ALL_SAMPLES,7)), feature_test = np.array(X_test).reshape((1,7)),
		Y_train = Y_train_Lin,Y_test = pd.DataFrame(),degree = 2,reportMSE = False)
	

	print("MPG Prediction is %f"%mpgpred)
	clf = LogRegression(dfcopy,only_model = True)
	clf.fit(X_train,Y_train_Cat)
	
	print("Cat Prediction is ",clf.predict(np.array(X_test).reshape(1, 7))[0])


#The main function to perform all the tasks.
def main():
	
	print("Problem 1 Solution\n")
	df = remove_missing()	#remove the missing features; the missing value records
	df = split_ranges(df)	#the threshold break down to do the classification; Problem 1
	scatter_generate(df)	#the scatter plot matrix to be generated ; Problem 2
	df = shuffle(df,random_state = 0) #need to shuffle the data prior to split
	regressEach(df)	#Problem 4
	regressAll(df)		#Problem 5
	print("Problem 6 Solution\n")
	LogRegression(df)	#Problem 6
	ScaleRegress(df)	#Problem 7
	testSample(df,[4,400,150,3500,8,81,1])	#Problem 8
	

if __name__ == '__main__':
	main()
