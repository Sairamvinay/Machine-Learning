
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