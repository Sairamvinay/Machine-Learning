import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# helper function to remove the missing value records; Problem 1
def remove_missing():

	features = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','name']
	df = pd.read_csv("auto-mpg.data",names = features,sep = '\s+')	#read the data file which has only spaces as seperator token
	df = df[df["horsepower"] != '?']	#to clean up the missing values in the data set
	df["horsepower"] = df["horsepower"].astype(float)
	return df

#to break the mpg values into 4 diff categories acc to thresholds; Problem 1
def break_ranges(df):
	
	
	df = df.sort_values(by = ["mpg"])	#sort all the mpg values so that it is easy to split into categories
	num_samples = len(df["mpg"])	#need it to break up the samples by 4
	
	print("The thresholds for each quarter are: ")

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

	#Equally sized bins

	print(len(df_first_qtr))
	print(len(df_second_qtr))
	print(len(df_third_qtr))
	print(len(df_fourth_qtr))


	dfnew = pd.concat([df_first_qtr,df_second_qtr,df_third_qtr,df_fourth_qtr])	#join these bins into new dataframe
	dfnew = dfnew.sort_index()	#to get back the old order of our data frame

	return dfnew

#generates the scatter matrix plot; Problem 2
def scatter_generate(df):


	colors = {'low':'red','med':'blue','high':'green','vhigh':'yellow'}	#need to identify the colors
	c = []
	for i,cat in enumerate(df["cat"]):
		c.append(colors[cat])	#get the color encoding for that particular label
		


	dfcopy = df.drop(["cat","mpg"],axis = 1)	#drop off the mpg column so as to get the plot

	pd.plotting.scatter_matrix(dfcopy,alpha = 1,c = c,figsize = (8,8),diagonal = 'hist')
	plt.show()

#for problem 4
def grapher(zeroW,firstW,secondW,thirdW,testX,testY,feature):

	fig,ax = plt.subplots()

	ax.scatter(testX[:,0], testY, c='black',label = "data points")
	X = np.linspace(min(testX),max(testX),100)
	X = X.reshape((100,1))
	ones = np.ones(100).reshape((100,1))
	zeroPred = np.matmul(zeroW.T,ones.T)
	firstPred =	np.matmul(firstW.T,np.concatenate((ones,X),axis = 1).T)		#firstW[0] + (firstW[1] * X)
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


#Problem 3
def Linear_regressionSolver(feature_train = None,feature_test = None,Y_train = None,Y_test = None,degree = 1):


	X_totrain = np.ones(len(feature_train)).reshape((len(feature_train),1))
	X_totest = np.ones(len(feature_test)).reshape((len(feature_test),1))

	if degree == 0:
		pass

	elif degree == 1:

		
		X_totrain = np.concatenate((X_totrain,feature_train),axis = 1)
		X_totest = np.concatenate((X_totest,feature_test),axis = 1)

	elif degree == 2:

		Xsquaretrain = feature_train ** 2
		
		Xsquaretest = feature_test ** 2
		
		X_totrain = np.concatenate((X_totrain,feature_train,Xsquaretrain),axis = 1)
		X_totest = np.concatenate((X_totest,feature_test,Xsquaretest),axis = 1)

	elif degree == 3:
		Xsquaretrain = feature_train ** 2
		Xcubetrain = feature_train ** 3

		Xsquaretest = feature_test ** 2
		Xcubetest = feature_test ** 3

		X_totrain = np.concatenate((X_totrain,feature_train,Xsquaretrain,Xcubetrain),axis = 1)
		X_totest = np.concatenate((X_totest,feature_test,Xsquaretest,Xcubetest),axis = 1)

	else:
		print("Error:Wrong degree entered")


	Weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_totrain.T,X_totrain)),X_totrain.T),Y_train)
	Weights = Weights.reshape((Weights.shape[0],1))
	print("weights are for the degree ",degree,": ",Weights)

	trainpred = np.matmul(Weights.T,X_totrain.T)
	testpred = np.matmul(Weights.T,X_totest.T)

	trainpred = np.array(trainpred.T).ravel()
	testpred = np.array(testpred.T).ravel()

	#print(testpred)
	trainMSE = np.mean((trainpred - Y_train) ** 2)
	testMSE = np.mean((testpred - Y_test) ** 2)


	print("The training Mean square error for the %sth order solver is %s"%(degree,trainMSE))
	print("The testing  Mean square error for the %sth order solver is %s"%(degree,testMSE))
	print("\n")

	return Weights

#Problem 5 solution
def regressAll(df_all):

	dfcopy = df_all.drop(labels = ["cat","mpg","name"],axis = 1)

	X = dfcopy.to_numpy()

	trainX,testX = X[:292],X[292:]
	Y = df_all["mpg"]
	trainY = Y[:292]
	testY = Y[292:]

	print("For regression against all features")
	zerothTestPred = Linear_regressionSolver(feature_train = trainX, feature_test = testX, Y_train = trainY, Y_test = testY, degree = 0)
	firstTestPred  = Linear_regressionSolver(feature_train = trainX, feature_test = testX, Y_train = trainY, Y_test = testY, degree = 1)
	secondTestPred = Linear_regressionSolver(feature_train = trainX, feature_test = testX, Y_train = trainY, Y_test = testY, degree = 2)


#Problem 4 solution
def regressEach(df_all):
	columns = df_all.columns.tolist()

	columns.remove("cat")
	columns.remove("name")
	columns.remove("mpg")
	
	Y = df_all["mpg"]
	trainY = Y[:292]
	testY = Y[292:]
	
	print("For regression against each feature independently")
	for col in columns:

		print("For the %s feature\n"%col)
		
		X = df_all[col]


		trainX = np.array(X[:292]).reshape((292,1))
		testX  = np.array(X[292:]).reshape((100,1))
		
	

		zerothW= Linear_regressionSolver(feature_train = trainX,feature_test = testX, Y_train = trainY, Y_test = testY,degree = 0)
		firstW =  Linear_regressionSolver(feature_train = trainX,feature_test = testX, Y_train = trainY, Y_test = testY,degree = 1)
		secondW = Linear_regressionSolver(feature_train = trainX,feature_test = testX, Y_train = trainY, Y_test = testY,degree = 2)
		thirdW =  Linear_regressionSolver(feature_train = trainX,feature_test = testX, Y_train = trainY, Y_test = testY,degree = 3)
		grapher(zerothW,firstW,secondW,thirdW,testX,testY,col)
		print('\n')

#Problem 6 solution
def LogRegression(df_all):
	
	
	clf = LogisticRegression()
	Y_train,Y_test = df_all["cat"][:292],df_all["cat"][292:]
	dfcopy = df_all.drop(labels = ["cat","mpg","name"],axis = 1)

	X = dfcopy.to_numpy()
	X_train,X_test = X[:292],X[292:]
	clf.fit(X_train,Y_train)

	print("Logistic Regression Algorithm results")
	print("Training Accuracy is %f%%"%(clf.score(X_train,Y_train) * 100.00))
	print("Testing Accuracy is %f%%"%(clf.score(X_test,Y_test) * 100.00))


def ScaleRegress(df_all):
	dfcopy = df_all.drop(labels = ["name"],axis = 1)
	#work on it later tomm

def main():
	df = remove_missing()	#remove the missing features; the missing value records
	df = break_ranges(df)	#the threshold break down to do the classification; Problem 1
	#scatter_generate(dfnew)	#the scatter plot matrix to be generated ; Problem 2
	

	df = shuffle(df,random_state=0) #need to shuffle the data prior to split
	
	#regressEach(df)	#Problem 4
	#regressAll(df)		#Problem 5
	LogRegression(df)	#Problem 6
	ScaleRegress(df)#do min max normalizing and work on it


	

if __name__ == '__main__':
	main()
