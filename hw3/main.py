from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import make_scorer,mean_squared_error,roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
from sklearn.feature_selection import SelectFromModel as SFM
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize,MultiLabelBinarizer,StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import warnings
from collections import Counter
import operator
#used warnings so that the warnings are not shown while I use classifiers
warnings.filterwarnings("ignore")

#basic global variables which are constantly used in the program
FILENAME = 'ecs171.dataset.txt'	#the filename we are working with
EXPRESSION_START = 6	#the col number from where the gene expressions start from
NUM_ITER = 500	#the number of iterations for the bootstrapping method
NUM_SAMPLES_RAND = 100	#the random number of samples to pick as a subset of the original number of samples for the bootstrapping
PCA_TSNE_COMP = 2	#the number of components for PCA and tSNE (fixed at since we do 2D visualization)
RANDOM_STATE = 999	#a random state used throughout the problem for consistency
CONFIDENCE_VAL = 0.95 #the confidence interval score



#the scorer function for the lasso to pick the best alpha using the minimal MSE value
def non_negate_MSE(y,ypred,**kwargs):
	return mean_squared_error(y,ypred,**kwargs)

#a simple method to read in files and return pandas dataframe
def read_files(file = FILENAME):

	df = pd.read_csv(file,header = 0,sep='\s+')
	df = df.drop(df.columns[len(df.columns)-1], axis=1)	#drop last column
	df = df.dropna()	#drop null samples if any
	return df


#method to create a lasso model: alpha is default at the best alpha value from P1 (this is mainly for P2)
def LassoModel(alpha = 0.0001):
	clf = Lasso(random_state = RANDOM_STATE,alpha = alpha)
	return clf



#a grid search method which does grid search on the alphas to find best alpha for num_splits fold and returns
#the best model and the number of coefficients used
def grid_search(X,Y,num_splits = 5):

	myMetric = make_scorer(non_negate_MSE,greater_is_better = False)	#need the scorer to rank the best alpha
	clf = Lasso(random_state = RANDOM_STATE)	#CLASSIFIER
	alpha_regress = [1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01,1.e+02] #range of alpha to do grid search on
	param_grid = dict(alpha = alpha_regress)
	grid = GridSearchCV(estimator = clf,param_grid = param_grid,scoring = myMetric,cv = num_splits)
	grid_result = grid.fit(X,Y)
	best_model = grid_result.best_estimator_	#get best model

	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	coeff_used = np.sum(best_model.coef_!= 0)
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
	    print("%f (%f) with: %r" % (mean, stdev, param))

	return best_model,coeff_used

#the method for P1
#find the best alpha for Lasso and find the number of non-zero coefficients used
def P1(df):

	print("PROBLEM 1")
	X = df.iloc[:,EXPRESSION_START:]	#pick all the gene expressions as input features
	Y = df["GrowthRate"].to_numpy()	#the output to predict: growthrate
	_,coeff_used = grid_search(X,Y)	#perform grid search; default is for 5 fold
	print("Number of coefficients used ",coeff_used)	#report output


def P2(df,num_samples = None):

	print("PROBLEM 2 and 3")
	scores = []

	Y = df["GrowthRate"].to_numpy()	#Output to predict
	X = df.iloc[:,EXPRESSION_START:]	#the input features: gene expressions
	X_mean = X.mean(axis = 0)	#the mean value for the gene expressions: returns the mean expression value for every expression
	X_mean = X_mean.to_numpy().reshape(1,-1)
	Y_mean = [Y.mean(axis = 0)]	#a single value of the mean growth rate
	
	#BOOTSTRAPPING METHOD PERFORMED HERE
	for i in range(NUM_ITER):
		X_sampled,Y_sampled = resample(X,Y,n_samples = num_samples)	#pick random subset every time
		
		clf = LassoModel()
		clf.fit(X_sampled,Y_sampled)	#train lasso on the subset
		pred = clf.predict(X_mean)	#predicted value
		acc = mean_squared_error(Y_mean,pred)	#get the MSE score
		scores.append(acc)	#collect all the scores

	#the confidence interval to find
	alpha = 0.95
	perc = ((1.0-alpha)/2.0) * 100
	low = max(0.0, np.percentile(scores, perc))
	perc = (alpha+((1.0-alpha)/2.0)) * 100
	high = min(1.0, np.percentile(scores, perc))
	print('%.1f confidence value: the Generalization error is mostly between %.5f value and %.5f value' % (alpha*100, low, high))

	plt.xlabel("Mean squared error values")
	plt.ylabel("Number of iterations with same MSE value")
	plt.title("Confidence Interval using Bootstrapping with NUM_ITER = " + str(NUM_ITER))
	plt.hist(scores)
	plt.show()

#a method to graph ROC plots which have plots for num_splits different plots
def graphROC(Xred,Yclass,predictor,YcolName,num_splits = 5,title = "regular feature selection"):
	fold = KFold(n_splits = num_splits,random_state = RANDOM_STATE)
	foldno = 0
	TRUES = []	#for collecting true values
	PREDS = []	#for collecting predicted values
	for train_index,test_index in fold.split(Xred,Yclass):
		print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
		X_train, X_test = Xred[train_index], Xred[test_index]
		Y_train, Y_test = Yclass[train_index], Yclass[test_index]
		
		predictor.fit(X_train,Y_train)
		ypred = predictor.predict(X_test)
		foldno += 1	#keep count of the number of folds
		fpr,tpr,_ = roc_curve(Y_test.ravel(),ypred.ravel())	#get the FPR and the TPR values
		AUCscore = roc_auc_score(Y_test.ravel(), ypred.ravel(),average = "micro")	#Find AUC score
		PREDS.append(ypred.ravel())	#collect all predictions
		TRUES.append(Y_test.ravel())	#collect all true values
		plt.plot(fpr,tpr,label = "Fold "+str(foldno) + " AUC Score = " + str(round(AUCscore,2)))	#plot each fold
		

	FPR_av,TPR_av,_ = roc_curve(np.concatenate(TRUES),np.concatenate(PREDS))	#get the Average FPR and TPR
	AUCscoreav = roc_auc_score(np.concatenate(TRUES),np.concatenate(PREDS))	#get the micro average AUC score
	print("Average AUC score for the ROC curve is %.2f"%AUCscoreav)
	plt.plot(FPR_av,TPR_av,label = "Mean Curve with Average AUC score = " + str(round(AUCscoreav,2)))	#plot micro average curve also
	plt.plot([0, 1], [0, 1], 'k--',lw = 2,label = "No skill line AUC Score = 0.5")	#plot no skill line also: (always 0.5 as area)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title("ROC curves for predicting " + YcolName + " using " + title)
	plt.legend()
	plt.show()


#a method to graph ROC plots which have plots for num_splits different plots
#same methodology as graphROC() but for PR curves
def graphPR(Xred,Yclass,predictor,YcolName,num_splits = 5,title = "regular feature selection",plotBase = False):
	fold = KFold(n_splits = num_splits,random_state = RANDOM_STATE)
	foldno = 0
	TRUES = []
	PREDS = []
	for train_index,test_index in fold.split(Xred,Yclass):
		print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
		X_train, X_test = Xred[train_index], Xred[test_index]
		Y_train, Y_test = Yclass[train_index], Yclass[test_index]
		
		predictor.fit(X_train,Y_train)
		ypred = predictor.predict(X_test)
		foldno += 1
		precision,recall,_ = precision_recall_curve(Y_test.ravel(),ypred.ravel())	#get precision, recall values
		AUPRCscore = average_precision_score(Y_test.ravel(), ypred.ravel(),average = "micro")	#get micro average AUPRC score for each fold
		PREDS.append(ypred.ravel())
		TRUES.append(Y_test.ravel())
		plt.plot(recall,precision,label = "Fold "+str(foldno) + " AUPRC Score = " + str(round(AUPRCscore,2)))
		

	precision_av,recall_av,_ = precision_recall_curve(np.concatenate(TRUES),np.concatenate(PREDS))
	AUPRCscoreav = average_precision_score(np.concatenate(TRUES),np.concatenate(PREDS))
	print("Average AUPRC score for the PR curve is %.2f"%AUPRCscoreav)	#plot micro average plot
	plt.plot(recall_av,precision_av,label = "Mean Curve with Average AUPRC score = " + str(round(AUPRCscoreav,2)))
	
	if plotBase:
		plt.plot([0, 1], [0.5, 0.5], 'k--',label = "Baseline PR curve with AUPR score = 0.5")

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title("PR curves for predicting " + YcolName + " using " + title)
	plt.legend()
	plt.show()

#method to perform feature selection using featSelectMethod provided 
#it also does grid_search to use best model for each case
#it returns the reduced X,Y and also the OneVsRestSVM classifier for predictions
def FeatureSelect(df,YcolName,featSelectMethod = "SFM",printNumCoeff = False):

	
	X = df.iloc[:,EXPRESSION_START:]	#get all expressions
	Y = df[YcolName].to_numpy()	#the output we want to predict
	Yclass = label_binarize(Y,list(set(Y)))	#get the binarized value
	best_model,coeff_used = grid_search(X,Yclass)	#get best model and alpha
	if (printNumCoeff):
		print("Number of coefficients used ",coeff_used)


	best_model.fit(X,Yclass)	#fit best model
	predictor = OneVsRestClassifier(SVC(C = 1,kernel = 'linear',probability = True))
	Xred = None
	if featSelectMethod == "SFM":
		selector = SFM(best_model,prefit = True)
		Xred = selector.transform(X)

	elif featSelectMethod == "PCA":
		selector = PCA(n_components = PCA_TSNE_COMP,random_state = RANDOM_STATE)
		Xred = selector.fit_transform(X)

	elif featSelectMethod == "tSNE":
		selector = TSNE(n_components=PCA_TSNE_COMP,random_state = RANDOM_STATE)
		Xred = selector.fit_transform(X)

	return Xred,Yclass,predictor
	

#performs Q4 requirements
#graphs ROC and PR for each output after performing feature selection
def P4(df):
	outputs = ["Strain","Medium","Stress","GenePerturbed"]
	for col in outputs[:]:
		print("Problem 4 for column %s"%col)
		Xred,Yclass,predictor = FeatureSelect(df,col,featSelectMethod = "SFM",printNumCoeff = True)
		graphROC(Xred,Yclass,predictor,col,num_splits = 5)
		graphPR(Xred,Yclass,predictor,col,num_splits = 5)


	

#does Q5
#the medium and stress combined problem to predict
def P5(df):
	Y_med = df["Medium"]
	Y_stress = df["Stress"]
	Y = [(med,stress) for med,stress in zip(Y_med,Y_stress)]	#to get combined output
	X = df.iloc[:,EXPRESSION_START:]	#get expressions X
	Yclass = MultiLabelBinarizer().fit_transform(Y)	#need to get muli label encoder
	best_model,_ = grid_search(X,Yclass) #get best model
	best_model.fit(X,Yclass)	#fit the best model

	selector = SFM(best_model,prefit = True)
	Xred = selector.transform(X)	#reduced X value

	predictor = OneVsRestClassifier(SVC(C = 1,kernel = 'linear',probability = True))

	graphROC(Xred,Yclass,predictor,"Combined=Medium_Stress",num_splits = 10)
	graphPR(Xred,Yclass,predictor,"Combined=Medium_Stress",num_splits = 10,plotBase = True)

#a helper method to get graphs for PCA and TSNE
def plot_graph(comp1,comp2,title):

	plt.scatter(comp1,comp2,alpha = 0.8,color = "red")
	plt.xlabel("Component 1")
	plt.ylabel("Component 2")
	plt.title(title)
	plt.show()

#calls plot_graph() 2 times to get PCA and TSNE component
def P6(df):

	X = df.iloc[:,EXPRESSION_START:]
	scaler = StandardScaler()
	Xscaled = scaler.fit_transform(X)	#scaled input since values are differently spaced out; so wanted to scale them to a particular range of values
	pca = PCA(n_components = PCA_TSNE_COMP,random_state = RANDOM_STATE)
	tSNE = TSNE(n_components=PCA_TSNE_COMP,random_state = RANDOM_STATE)
	pc_comp = np.array(pca.fit_transform(Xscaled))
	tsne_comp = np.array(tSNE.fit_transform(Xscaled))

	pc1,pc2 = pc_comp.T[0],pc_comp.T[1] #to get each component conveniently
	tsne1,tsne2 = tsne_comp.T[0],tsne_comp.T[1]	#to get each component conveniently
	

	plot_graph(pc1,pc2,"PCA for gene expressions")
	plot_graph(tsne1,tsne2,"tSNE for gene expressions")


#the Q7 solution: finds PCA and TSNE value and it graphs each time for getting the mean AUC and AUPRC score
def P7(df):
	outputs = ["Strain","Medium","Stress","GenePerturbed"]
	for col in outputs[:]:
		print("Problem 7 for column %s"%col)
		XredPCA,YclassPCA,predictorPCA = FeatureSelect(df,col,featSelectMethod = "PCA")
		XredtSNE,YclasstSNE,predictortSNE = FeatureSelect(df,col,featSelectMethod = "tSNE")
		graphROC(XredPCA,YclassPCA,predictorPCA,col,num_splits = 10,title = "PCA")
		graphROC(XredtSNE,YclasstSNE,predictortSNE,col,num_splits = 10,title = "tSNE")
		graphPR(XredPCA,YclassPCA,predictorPCA,col,num_splits = 10,title = "PCA")
		graphPR(XredtSNE,YclasstSNE,predictortSNE,col,num_splits = 10,title = "tSNE")


def main():
	start = time()
	df = read_files()
	P1(df)
	P2(df,NUM_SAMPLES_RAND)
	P4(df)
	P5(df)
	P6(df)
	P7(df)
	end = time()
	taken = (end - start)/60.00
	print("Time taken %.5f minutes"%taken)

if __name__ == '__main__':
	main()








