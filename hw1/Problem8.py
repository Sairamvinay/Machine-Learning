def testSample(df,X_test):


	Y_train_Lin = df["mpg"]
	Y_train_Cat = df["cat"]
	dfcopy = df.drop(labels = ["cat","mpg","name"],axis = 1)

	X_train = dfcopy.to_numpy()
	# print(X_train.shape)
	mpgpred = Linear_regressionSolver(feature_train = X_train.reshape((392,7)), feature_test = np.array(X_test).reshape((1,7)),Y_train = Y_train_Lin,degree = 2)
	print("MPG Pred is %f"%mpgpred)
	clf = LogRegression(dfcopy,only_model = True)
	clf.fit(X_train,Y_train_Cat)
	print("Cat Pred is ",clf.predict(np.array(X_test).reshape(1, 7))[0])


