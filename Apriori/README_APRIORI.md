The inputs to the apriori algorithm is going to be a file. 

### Note: You will need to change the file names to the correct location and need candidates, data set, and C1 
create a variables:

candidates = list(find_candidates(read_data1(filename)))
dataset = read_data1(filename)

To run the apriori algorithm you can run 

l, support_data = apriori(dataset, minsupport)

To Generate Rules:

rules = generateRules(l, support_data, min_confidence) 

The output will give you frozen sets with the items that are frequently found by the minsupport and confidence. 

The Results are in the Apriori Document.docx
