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