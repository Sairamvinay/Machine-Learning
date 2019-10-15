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

