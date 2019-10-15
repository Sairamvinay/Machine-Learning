#Problem 6 solution
def LogRegression(df_all,scale_down = False,only_model = False):
	
	
	clf = LogisticRegression()
	if only_model:
		return clf


	Y_train,Y_test = df_all["cat"][:292],df_all["cat"][292:]
	dfcopy = df_all.drop(labels = ["cat","mpg","name"],axis = 1)

	X = dfcopy.to_numpy()
	

	if scale_down == True:
		scaler = MinMaxScaler()
		scaler.fit(X)
		X = scaler.transform(X)


	X_train,X_test = X[:292],X[292:]

	clf.fit(X_train,Y_train)

	#Ytrainpred = clf.predict(X_train)
	#Ytestpred  = clf.predict(X_test)
	print("Logistic Regression Algorithm results")
	print("Training Accuracy is %f%%"%(clf.score(X_train,Y_train) * 100.00))
	print("Testing Accuracy is %f%%"%(clf.score(X_test,Y_test) * 100.00))
