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
		print("Error: Wrong degree entered")

	print(X_totest)
	Weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_totrain.T,X_totrain)),X_totrain.T),Y_train)
	Weights = Weights.reshape((Weights.shape[0],1))
	print("weights are for the degree ",degree,": ",Weights)

	trainpred = np.matmul(Weights.T,X_totrain.T)
	testpred = np.matmul(Weights.T,X_totest.T)

	trainpred = np.array(trainpred.T).ravel()
	testpred = np.array(testpred.T).ravel()

	# print("Train predictions are: ",trainpred)
	# print("Test predictions are: ",testpred)

	trainMSE = np.mean((trainpred - Y_train) ** 2)
	#testMSE = np.mean((testpred - Y_test) ** 2)


	print("The training Mean square error for the %sth order solver is %s"%(degree,trainMSE))
	#print("The testing  Mean square error for the %sth order solver is %s"%(degree,testMSE))
	print("\n")
	return testpred
	#return Weights