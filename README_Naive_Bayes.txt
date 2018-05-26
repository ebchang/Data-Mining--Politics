The input is going to be a file. 

I have put in the two files in the same folder called politics.csv and politics_Trimmed.csv 

The politics.csv file is the original file contained all 10,000 negative examples (Class: 0) and 300 positive examples (Class: 1)

To run Naive Bayes: 

Create Variables:

within the main function:

dataset = loadCSV(filename) Note: To discretize-> dataset = discretize(loadCSV(filename)

to run type in main(filename)

This will print out an accuracy

the function: counter(ROC_list) will give the TP, TN, FP, FN and return the FPR and TPR to find the Area under the ROC curve. 