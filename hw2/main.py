import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers

#These Global variables define the basic variables used in our code

FILENAME = "yeast.data"	#the filename which can be changed as per user's preference
NUM_FEATURES = 8		#the number of features used for ANN
CYT_NODE = 0			#the output node which refers to CYT in output layer
ALL_WEIGHTS_CYT = []	#array to store all weights connecting to CYT output node in output layer 
ALL_BIAS_CYT = []		#array to store all bias connecting to CYT output node in output layer 
EPOCHS = 300			#default number of epochs
BATCH_SIZE = 1		#default batch size to train by 1 sample for each batch
NUM_CLASSES = 10		#the number of output nodes in ANN
LEARNING_RATE = 0.01	#the default learning rate for SGD
ENCODINGS = dict({"CYT":0,"NUC":1,"MIT":2,"ME3":3,"ME2":4,"ME1":5,"EXC":6,"VAC":7,"POX":8,"ERL":9})	#a dictionary to take care of the encodings


np.random.seed(999)	#so that all the weights are initialized to same random weights in every run


# a callback class to extract Weights and bias connected to the CYT output node in output layer at end of each epoch
#uses the global arrays to take care of the extraction
class getWeights(keras.callbacks.Callback):
	#override on_epoch_end method to obtain weights every epoch end while training
	def on_epoch_end(self, batch, logs={}):
		
		output_layer = self.model.layers[-1]	#get the output (last signalled by -1) layer
		weights = output_layer.get_weights()[0]	#get weights from previous hidden layer activation nodes to CYT node
		bias = output_layer.get_weights()[1]	#get bias from previous hidden layer activation nodes to CYT node

		#for CYT alone
		CYTweights = [weights[hidden_node_prev_layer][CYT_NODE] for hidden_node_prev_layer in range(len(weights))]	#get an array of 3 weights which connects prev hidden layer to CYT Node (0 since our encoding implies so)
		CYTbias = bias[CYT_NODE]	#get corresponding bias
		ALL_WEIGHTS_CYT.append(CYTweights)	#append our weights and bias for graphing later
		ALL_BIAS_CYT.append(CYTbias)
		

# Grapher method to plot the train and testing error per iteration
def graphErrors(X,trainError,testError):

	plt.plot(X,trainError.tolist(),label = "Training Error",color = "red")
	plt.plot(X,testError.tolist(),label = "Testing Error",color = "blue")
	plt.xlabel("Epochs")
	plt.ylabel("Training/Testing Error")
	plt.title("Epochs vs Error")
	plt.legend()
	plt.show()

# Grapher method to plot the weights and bias per iteration
def graphWeights(X,Y_weights,Y_bias):
	
	#did this since matrix has weights in form [[W_11, W12,W13 for epoch i]] with dim EPOCHS by 3 (3 prev_hidden layer nodes)
	#to make extraction easy, we need W_11, W_12 and W_13 for every sample as a seperate array
	#so transpose makes it easier since we get now [[W_11 for every epoch i] [W_12 for every epoch i] [W_13 for every epoch i]]
	matrix = np.array(Y_weights).T 		
	weight1 = matrix[0]	#get W_11 array
	weight2 = matrix[1]	#get W_12 array
	weight3 = matrix[2]	#get W_13 array
	bias = Y_bias	#the bias as it is

	plt.plot(X,weight1,label = "Weight from node 1 from Hidden layer 2",color = 'red')
	plt.plot(X,weight2,label = "Weight from node 2 from Hidden layer 2",color = 'blue')
	plt.plot(X,weight3,label = "Weight from node 3 from Hidden layer 2",color = "green")
	plt.plot(X,bias,label = "Bias from Hidden Layer 2",color = "yellow")

	plt.xlabel("Epochs")
	plt.ylabel("Weight/Bias")

	plt.title("Epochs vs Weights/Bias")

	plt.legend()

	plt.show()


# a simpler helper method to parse the filename and 
#remove sequence right at the beginning (never used)
#returns a data frame object

def process_data(filename):

	df = pd.read_csv(filename,sep = '\s+',names = ["Sequence Name","mcg","gvh","alm","mit","erl","pox","vac","nuc","class"])
	df = df.drop(["Sequence Name"],axis = 1)	#remove sequence name since it is redundant
	return df

# a method to find and remove outliers using IF alone
#returns the modified data frame with removed outliers
def find_outliers(df):
	
	dfcopy = df.drop(["class"],axis = 1)	#take out class for now since we need to find outlier only using features
	num_rows = dfcopy.shape[0]		#need this to perform calculations
	Y = df["class"]		#need Y seperately since we need to remove corresponding class values for that outlier sample
	clf1 = IsolationForest(random_state = 0)	#apply IF on the data set for outlier detection
	clf1.fit(dfcopy)		
	predictions1 = clf1.predict(dfcopy)	#obtain array of isoutlier or not predictions
	indices = np.where(predictions1 == -1)[0]	#extract of all outlier samples index

	print("Indices of -1 predictions: ",indices)
	print("For Isolation Forest: Percentage of outliers:",100.00 * predictions1.tolist().count(-1)/num_rows)

	#drop out those outliers
	dfcopy = dfcopy.drop(index = indices,axis = 0)
	Y = Y.drop(index = indices,axis = 0)


	clf2 = OneClassSVM(random_state = 0)
	clf2.fit(dfcopy)
	predictions2 = clf2.predict(dfcopy)
	
	print("For One Class SVM: Percentage of outliers:",100.00 * predictions2.tolist().count(-1)/num_rows) #check for percentage for this, 50%??

	#rejoin the output vector back to the old dataframe
	dfcopy["class"] = Y
	print("After dropping: data frame shape is ",dfcopy.shape)	#verify shape after removal

	return dfcopy


#a simple function to print all the weights and bias of the array [weight,bias]
#print only CYT_NODE values
def printWeights(weights_bias):

	print("After Training: CYT node weights are:")
	weights = weights_bias[0]	#get the weights arrays
	bias = weights_bias[1]	#get the bias
	#get CYT_NODE weights alone
	weights0 = weights[0][CYT_NODE]
	weights1 = weights[1][CYT_NODE]
	weights2 = weights[2][CYT_NODE]
	biasval = bias[CYT_NODE]
	print("Weight from node 1 from Hidden layer 2: ",weights0)
	print("Weight from node 2 from Hidden layer 2: ",weights1)
	print("Weight from node 3 from Hidden layer 2: ",weights2)
	print("Bias from Hidden Layer 2:",biasval)

#do the train_test_split
def split_data(X,Y):
	X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=0.34, random_state=0)
	return X_train,X_test,Y_train,Y_test

#this extracts X and Y seperately from data frame
#it also does one hot encoding for Y after doing encoding
def getXYdata(df):

	Y = np.array(df["class"])
	X = df.drop(["class"],axis = 1)	#seperate X and Y
	Y = Y.tolist()
	print("X shape is :",X.shape)
	print("Y shape is :",len(Y))
	

	#first find encoding for each class
	for i,classification in enumerate(Y):
		Y[i] = ENCODINGS[classification]



	Y = keras.utils.to_categorical(Y,num_classes = NUM_CLASSES)	#get it into one-hot encoded data
	return X,Y
	

#the method which implements neural network and returns the model which can be fit later
def ANN(hidden_layers = 2, hidden_neurons = 3,hidden_activation = "sigmoid",final_activation = "sigmoid",loss = "mean_squared_error",reset_weights = False):
	
	
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(hidden_neurons,input_dim = NUM_FEATURES,activation = hidden_activation))	#input layer and first hidden layer created
	
	#hidden_layers - 1 because we have one hidden layer already, need one lesser than num of hidden layers to add
	for num in range(hidden_layers - 1):
		model.add(keras.layers.Dense(hidden_neurons,activation = hidden_activation))
	

	model.add(keras.layers.Dense(NUM_CLASSES,activation = final_activation))

	#get a SGD model with LEARNING RATE
	sgd = optimizers.SGD(lr=LEARNING_RATE)
	
	#a bool to reset weights if needed, helpful for P4()
	if reset_weights:
		#create zeros at first
		weights_OP = [np.zeros((hidden_neurons,NUM_CLASSES)),np.zeros(NUM_CLASSES)]	
		weights_HL2 = [np.zeros((hidden_neurons,hidden_neurons)),np.zeros(hidden_neurons)]
		weights_HL1 = [np.zeros((NUM_FEATURES,hidden_neurons)),np.zeros(hidden_neurons)]
		
		weights_OP[0][:,CYT_NODE] = 1	#set only CYT_NODE weights to 1 as we only use these
		weights_OP[1][CYT_NODE] = 1	#set the bias for that node
		
		weights_HL2[0][:,0] = 1	#set only first node in 2nd hidden layer weights to 1 as we only use these
		weights_HL2[1][0] = 1	#set the bias for that node

		
		
		#reset all the weights
		model.layers[0].set_weights(weights_HL1)	
		model.layers[1].set_weights(weights_HL2)
		model.layers[2].set_weights(weights_OP)

		
	
	model.compile(loss = loss,optimizer = sgd,metrics=['accuracy'])

	return model


def P2(dfnew):

	print("Outlier removal is done:")	#we pass only the cleaned data frame (removed outliers)

	X,Y = getXYdata(dfnew)	#get X, Y
	
	X_train,X_test,Y_train,Y_test = split_data(X,Y)	#split the data
	
	weightGetter = getWeights()	#a class init call to create object of getWeights callback object
	
	
	model = ANN()	#get the model
	hist = model.fit(X_train,Y_train,epochs = EPOCHS,batch_size = BATCH_SIZE, callbacks = [weightGetter],validation_data = (X_test,Y_test))	#validate on test and get testing and training acc while training
	#callback takes care of getting weights
	
	train_acc,test_acc = hist.history['acc'],hist.history["val_acc"]	#get training/testing acc per epoch
	
	graphWeights(range(1,EPOCHS + 1),ALL_WEIGHTS_CYT,ALL_BIAS_CYT)	#graph only the weights and bias
	graphErrors(range(1,EPOCHS + 1), np.ones(len(train_acc)) - train_acc, np.ones(len(test_acc)) - test_acc)	#get the 1-acc graphs for each
	

#performs ANN on whole data set
def P3(df):
	
	print("No outlier removal now:")
	X,Y = getXYdata(df)
	print("No splitting of data")

	model = ANN()
	hist = model.fit(X,Y,epochs = EPOCHS,batch_size = BATCH_SIZE)	#fit the whole data set

	printWeights(model.layers[-1].get_weights())	#output (last) layer weights after training
	
	train_acc =  1 - hist.history["acc"][-1]	#training error = 1- acc formula
	print("Training error: ",train_acc)
	


def GridSearch(df,hidden_activation = "sigmoid",final_activation = "sigmoid",loss = "mean_squared_error"):
	
	print("Grid Search on the hyperparameters")
	hidden_layers = [1,2,3]
	hidden_neurons = [3,6,9,12]
	
	X,Y = getXYdata(df)	#get the X,Y data
	X_train,X_test,Y_train,Y_test = split_data(X,Y)
	testing_errors = []	#a matrix to take care of the testing error
	matrix = []	#a matrix of tuple of form (HL,HN) to record positions 

	#run 12 different models of ANN to find the best : Grid Search is performed here
	for HL in hidden_layers:

		HL_HNpair = []
		testing_ERR_HL = []
		
		for HN in hidden_neurons:
			
			model = ANN(hidden_layers = HL,hidden_neurons = HN,hidden_activation = hidden_activation,final_activation = final_activation,loss = loss)
			model.fit(X_train,Y_train,epochs = EPOCHS,batch_size = BATCH_SIZE)
			testing_err = 1 - model.evaluate(X_test,Y_test)[1]	#get the 1-acc error
			testing_ERR_HL.append(testing_err)	#get the first row values
			HL_HNpair.append((HL,HN))	#get the first row values



		matrix.append(HL_HNpair)	#first row added
		testing_errors.append(testing_ERR_HL)	#first row added


	print("For the matrix:\n",matrix)
	print("\nThe testing errors matrix are:\n",testing_errors)
	return matrix,testing_errors

	
def P5(df_cleaned):
	LEARNING_RATE = 0.001
	BATCH_SIZE = 10
	GridSearch(df_cleaned)
	LEARNING_RATE = 0.01
	BATCH_SIZE = 1

def P6(df,sample):

	model = ANN()	#get the ANN model
	X,Y = getXYdata(df)
	model.fit(X,Y,epochs = EPOCHS,batch_size = BATCH_SIZE)	#fit the whole model X and Y
	yhat = model.predict_classes(sample)[0]	#predict only for the sample provided

	#find the corresponding encoding for the class predicted (as an integer label)
	for k in ENCODINGS:
		if yhat == ENCODINGS[k]:
			print("Type of the Unknown Sample is ",k)
			break

#trains only one epoch and one batch and this is to verify for Q4
def P4():
	
	X = np.array([[0.52,0.47,0.52,0.23,0.55,0.03,0.52,0.39]])
	y = np.array([[0,0,1,0,0,0,0,0,0,0]])
	
	model = ANN(reset_weights = True)

	model.fit(X,y,epochs = 1,batch_size = 1)
	weights_biasOP = model.layers[-1].get_weights()	#get last layer weights
	weights_biasHL2 = model.layers[1].get_weights()	#get 2nd hidden layer weights
	print(weights_biasOP[0][:,CYT_NODE]," are the weights for Output layer first output node from 2nd hidden layer nodes")
	print(weights_biasOP[1][CYT_NODE]," is the bias for Output layer first output node from 2nd hidden layer nodes")
	print(weights_biasHL2[0][:,CYT_NODE]," are the weights for 2nd hidden layer first node from 1st hidden layer nodes")
	print(weights_biasHL2[1][CYT_NODE]," is the bias for 2nd hidden layer first nodes from 1st hidden layer nodes")


#perform Q7 tasks
def P7(df):

	#does grid search using the new hyperparameters
	matrix,testing_errors = GridSearch(df,hidden_activation = "relu",final_activation = "softmax",loss = "binary_crossentropy")
	matrix = np.array(matrix)
	testing_errors = np.array(testing_errors)
	X,Y = getXYdata(df)
	X_train,X_test,Y_train,Y_test = split_data(X,Y)

	a,b = np.unravel_index(np.argmin(testing_errors),testing_errors.shape)	#this get the index (r,c) version for argmin of errors
	hidden_layers,hidden_neurons = matrix[a][b]	#get the particular configuration
	print("Best model for hidden layers:",hidden_layers," and hidden neurons:",hidden_neurons) #check the same
	model = ANN(hidden_activation = "relu",final_activation = "softmax",loss = "binary_crossentropy",hidden_neurons = hidden_neurons,hidden_layers = hidden_layers)	#get the model first
	hist =  model.fit(X_train,Y_train,epochs = EPOCHS,batch_size = BATCH_SIZE,validation_data = (X_test,Y_test))	#fit with this name
	train_acc,test_acc = hist.history['acc'],hist.history["val_acc"]	#get the error arrays
	graphErrors(range(1,EPOCHS + 1), np.ones(len(train_acc)) - train_acc, np.ones(len(test_acc)) - test_acc)	#graph it



def main():
	
	df = process_data(FILENAME)
	df_cleaned = find_outliers(df)	#P1 call
	
	P2(df_cleaned)
	P3(df)
	P4()
	P5(df_cleaned)
	P6(df_cleaned,np.array([[0.52,0.47,0.52,0.23,0.55,0.03,0.52,0.39]]))
	P7(df_cleaned)
	

	


if __name__ == '__main__':
	main()
