import math 
import matplotlib.pyplot as plt
import numpy as np

def fold_data_training(dat):
    num_folds = 10
    subset_size = round(len(dat)/num_folds)
    for i in range(num_folds):
        testing_this_round = dat[i*subset_size:][:subset_size]
        training_this_round = dat[:i*subset_size] + dat[(i+1)*subset_size:]
#        training_this_round = np.array(training_this_round)
#        testing_this_round = np.array(testing_this_round)
    # train using training_this_round
    # evaluate against testing_this_round
    # save accuracy
    return training_this_round
    
def fold_data_testing(dat):
    num_folds = 10
    subset_size = round(len(dat)/num_folds)
    for i in range(num_folds):
        testing_this_round = dat[i*subset_size:][:subset_size]
        training_this_round = dat[:i*subset_size] + dat[(i+1)*subset_size:]
#        training_this_round = np.array(training_this_round)
#        testing_this_round = np.array(testing_this_round)
    # train using training_this_round
    # evaluate against testing_this_round
    # save accuracy
    return testing_this_round


def read (filename):
    raw = open(filename)
    hold = raw.readline()
    dat = raw.readlines()
    raw.close()
    for i in range(len(dat)):
        dat[i] = dat[i].split(',')
        dat[i].pop(0) # for the training set, skip attribute
        del dat[i][0] # Takes out the name of the people looks at features
        for j in range(len(dat[i])):
            dat[i][j] = int(float(dat[i][j].strip()))
    return dat

def discretize (dat):
    mins = [99999] * len(dat[0]) # adjust if necessary
    maxs = [-99999] * len(dat[0]) # adjust if necessary
    for datum in dat:
        for i in range(len(datum)):
            if datum[i] < mins[i]:
                mins[i] = datum[i]
            if datum[i] > maxs[i]:
                maxs[i] = datum[i]
    for i in range(len(mins)):
        if mins[i] != 0 or maxs[i] != 1: 
            mid = (mins[i] + maxs[i]) / float(2)
            for j in range(len(dat)):
                if dat[j][i] < mid:
                    dat[j][i] = 0
                else:
                    dat[j][i] = 1
    return dat 
    
# takes a set of data points and a list of used (unavailable) features
# lower branches of the tree will have a long list of used features
# returns the index of the feature with the greatest information gain
def pickBestFeature (dat, used):
    # build a histogram of the number of data points per label in the set
    labels = {} # empty histogram
    for datum in dat:
        label = datum[-1] # the label of the data point
        # if the histogram hasn't seen the label yet, add it
        if label not in labels: 
            labels[label] = 0
        labels[label] += 1 # increment number of instances of the label
    e_all = 0 # stores the sum of the entropy of the set of points
    for label in labels:
        p = labels[label] / float(len(dat)) # compute the probability of the label
        e_all -= p * math.log(p, 2) # subtract the probability of the label * the log_2 of the probability from the sum
    gains = [] # stores the information gain of each feature
    for i in range(len(dat[0]) - 1): # over all features, stopping short of the label
        if i not in used: # don't consider already used features
            pos = {} # empty histogram of the labels of the feature-positive points
            num_p = 0 # number of feature-positive points
            neg = {} # empty histogram of the labels of the feature-negative points
            num_n = 0 # number of feature-negative points
            # over each data point
            for datum in dat:
                label = datum[-1] # the label of the data point
                if datum[i] == 1: # if data point is feature-positive
                    # if the feature-positive histogram hasn't seen the label yet, add it
                    if label not in pos:
                        pos[label] = 0
                    pos[label] += 1 # increment number of instances of the label
                    num_p += 1 # increment number of instances of feature-positive points
                else: # if data point is feature-negative
                    # if the feature-negative histogram hasn't seen the label yet, add it
                    if label not in neg:
                        neg[label] = 0
                    neg[label] += 1 # increment number of instances of the label
                    num_n += 1 # increment number of instances of feature-negative points
            e_pos = 0 # stores the sum of the entropy of the feature-positive set of points
            for label in pos:
                p = pos[label] / float(num_p) # compute the probability of the label
                e_pos -= p * math.log(p, 2) # subtract the probability of the label * the log_2 of the probability from the sums
            e_pos *= num_p / float(len(dat)) # weight the entropy (normalize it)
            e_neg = 0 # stores the sum of the entropy of the feature-negative set of points
            for label in neg:
                p = neg[label] / float(num_n) # compute the probability of the label
                e_neg -= p * math.log(p, 2) # subtract the probability of the label * the log_2 of the probability from the sums
            e_neg *= num_n / float(len(dat)) # weight the entropy (normalize it)
            gains.append([e_all - (e_pos + e_neg), i]) # store the information gain and the index of the feature
#            print(gains)
    best = gains[0] # the current best candidate feature
    for i in range(1, len(gains)):
        if gains[i][0] > best[0]: # compare features against the best candidate
            best = gains[i] # new best candidate
    return best[1]# return the best candidate "int"


def splitOnFeature (dat, idx):
    pos = []
    neg = []
    for datum in dat: #dat is a list of lists
        if datum[idx] == 1:
            pos.append(datum)
        else:
            neg.append(datum)
    return [pos, neg] 
  
def majorityClass (dat):
    labels = {}
    for datum in dat:
        label = datum[-1]
        if label not in labels:
            labels[label] = 0
        labels[label] += 1
    majority = None
    for label in labels:
        if majority == None or labels[label] > labels[majority]:
            majority = label
    return majority    
    
class node:
    def __init__ (self):
        self.feat = None
        self.classed = None
        self.l_child = None
        self.r_child = None
        
    def setFeature (self, feat, pos_c, neg_c):
        self.feat = feat
        self.l_child = node()
        self.l_child.classed = pos_c
        self.r_child = node()
        self.r_child.classed = neg_c
        
    def classify (self, datum):
        if not self.l_child and not self.r_child:
            return self.classed
        elif datum[self.feat] == 1:
            return self.l_child.classify(datum)
        else:
            return self.r_child.classify(datum)
    
    def stringifyHelper (self, depth):
        if not self.l_child and not self.r_child:
            print ('    ' * depth + 'Classed: ' + str(self.classed))
        else:
            print ('    ' * depth + 'Split on feature: ' + str(self.feat))
        if self.l_child:
            print ('    ' * depth + '  If positive for feature:')
            self.l_child.stringifyHelper(depth + 1)
        if self.r_child:
            print ('    ' * depth + '  If negative for feature:') 
            self.r_child.stringifyHelper(depth + 1)
            
    def stringify (self):
        print ('Tree:')
        self.stringifyHelper(0)
        
def build_tree (dat, depth):  
    root = node()
    nodes = []
    dats = []
    used = [] 
    nodes.append(root)
    dats.append(dat)
    used.append([])
    for i in range(0, depth):
        new_nodes = []
        new_dats = []
        new_used = [] 
        for j in range(len(nodes)):
            feat = pickBestFeature(dats[j], used[j])
            split = splitOnFeature(dats[j], feat)
            if len(split[0]) > 0 and len(split[1]) > 0:  
                nodes[j].setFeature(feat, majorityClass(split[0]), majorityClass(split[1]))
                new_nodes.append(nodes[j].l_child)
                new_nodes.append(nodes[j].r_child) 
                new_dats.append(split[0]) 
                new_dats.append(split[1]) 
                new_used.append(used[j] + [feat])
                new_used.append(used[j] + [feat]) 
        nodes = new_nodes
        dats = new_dats
        used = new_used  
    return root     
    
def printCSV (dat):
    for datum in dat:
        row = str(datum[0])
        for i in range(1, len(datum)):
            row += ',' + str(datum[i])
        print (row)

        
def run (depth):
    print ('Training stage (depth ' + str(depth) + '):')
    dat = fold_data_training(read(r'/Users/evanchang/Desktop/politics.csv'))
#    dat = fold_data_training(discretize(read(r'/Users/evanchang/Desktop/politics_Trimmed.csv')))
    model = build_tree(dat, depth) 
    model.stringify()
    correct = 0
    zero_class_train =0
    one_class_train = 0
    FP_train = 0
    FN_train = 0
    wrong_list_train = []
    count = len(dat)
    for i in range(count): 
        if model.classify(dat[i]) == dat[i][-1]:
            correct += 1
    for j in range(count):
        if model.classify(dat[j]) == 0 and dat[j][-1] == 0: #calculates true negative
            zero_class_train += 1
        elif model.classify(dat[j]) == 1 and dat[j][-1] == 1: #calcs true positive
            one_class_train +=1
        elif model.classify(dat[j]) == 1 and dat[j][-1] == 0: #calcs False positive
            FP_train+=1
            wrong_list_train.append((dat[j][-1], model.classify(dat[j])))
        else:
            FN_train +=1  #cals False Negative
            wrong_list_train.append((dat[j][-1], model.classify(dat[j]))) #shows True then predicted
    print (str(correct) + ' out of ' + str(count) + ' classed correctly (' + str(round(correct / float(count), 3) * 100) + '%).')
    print(str(zero_class_train) + " " + str(one_class_train) + " " + str(wrong_list_train))
    print ('Testing stage:')
    dat = fold_data_testing(read(r'/Users/evanchang/Desktop/politics.csv'))
 #   dat = fold_data_testing(discretize(read(r'/Users/evanchang/Desktop/politics_Trimmed.csv')))
    correct = 0
    zero_class_test = 0
    one_class_test = 0
    FP_test=0
    FN_test =0
    wrong_list_test = []
    count = len(dat)
    for i in range(count):
        if model.classify(dat[i]) == dat[i][-1]:
            correct += 1
    for j in range(count):
        if model.classify(dat[j]) == 0 and dat[j][-1] == 0:
            zero_class_test += 1
        elif model.classify(dat[j]) == 1 and dat[j][-1] == 1:
            one_class_test +=1
        elif model.classify(dat[j]) == 1 and dat[j][-1] ==0:
            FP_test+=1
            wrong_list_test.append((dat[j][-1], model.classify(dat[j])))
        else:
            FN_test +=1
            wrong_list_test.append((dat[j][-1], model.classify(dat[j])))
    print (str(correct) + ' out of ' + str(count) + ' classed correctly (' + str(round(correct / float(count), 3) * 100) + '%).')
    print(str(zero_class_test) + " " + str(one_class_test) + " " + str(wrong_list_test))

    x = FP_test/(FP_test +zero_class_test)
    y = one_class_test/(one_class_test +FN_test)

    plt.plot(x,y)
    plt.show()

    auc = np.trapz(y,x)
    return auc 
    
