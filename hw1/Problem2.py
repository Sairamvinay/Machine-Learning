#generates the scatter matrix plot; Problem 2
def scatter_generate(df):


	colors = {'low':'red','med':'blue','high':'green','vhigh':'yellow'}	#need to identify the colors
	c = []
	for i,cat in enumerate(df["cat"]):
		c.append(colors[cat])	#get the color encoding for that particular label
		


	dfcopy = df.drop(["cat","mpg"],axis = 1)	#drop off the mpg column so as to get the plot

	pd.plotting.scatter_matrix(dfcopy,alpha = 1,c = c,figsize = (8,8),diagonal = 'hist')
	plt.show()