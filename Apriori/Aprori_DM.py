# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:51:20 2016

@author: evanchang
"""
#from collections import defaultdict

def read (filename):
    raw = open(filename)
    hold = raw.readline()
    dataset = raw.readlines()
    id_hold = {}
    raw.close()
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(',')
#    print(dataset)
#    for d in range(len(dataset)):
#        if int(dataset[d][2]) > 0:
#            dataset[d][2] = 1
#        else:
#            dataset[d][0] = 0
    #makes a dictionary of keys :[items]
    for t in range(len(dataset)):
        id_hold.setdefault(dataset[t][0],[]).append(dataset[t][1])
    dataset = list(id_hold.values())
    return dataset

def read_data1(filename):
    raw = open(filename)
    hold = raw.readline()
    dataset = raw.readlines()
    id_hold = {}
    raw.close()
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(",")
    for t in range(len(dataset)):
        id_hold.setdefault(dataset[t][0],[]).append(dataset[t][1:])
#    print(dataset)
#    dataset = id_hold.values()
    return dataset

def find_candidates(dataset):
    #Will find the candidates of one. It will go through the list of lists
    #and find the objects in the dataset creating candidate sets of 1
    freq_sets = []
    for c in dataset:
        for freq in c:
            if [freq] not in freq_sets:
                freq_sets.append([freq])
    freq_sets.sort()
    return map(frozenset,freq_sets) #map it out to be a key of a dict. 

#C1 = find_candidates(read("/Users/evanchang/Desktop/10000_dataset.txt"))
#C1 = find_candidates(read_data1("/Users/evanchang/Desktop/trimmed.csv"))
#candidates = list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed.csv")))
#candidates = list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed_totaldonate.csv")))
candidates = list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed_totaldonate_zero.csv")))
#candidates = list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed_obamadonate.csv")))
#candidates = list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed_obamadonate_zero.csv")))
#"/Users/evanchang/Desktop/trimmed_totaldonate.csv"
#"/Users/evanchang/Desktop/trimmed_obamadonate.csv"
#dataset = read_data1("/Users/evanchang/Desktop/trimmed.csv")
#dataset = read_data1("/Users/evanchang/Desktop/trimmed_totaldonate.csv")
dataset = read_data1("/Users/evanchang/Desktop/trimmed_totaldonate_zero.csv")
#dataset = read_data1("/Users/evanchang/Desktop/trimmed_obamadonate.csv")
#dataset =read_data1("/Users/evanchang/Desktop/trimmed_obamadonate_zero.csv")
#len_candidates = len(list(find_freqsets_data1(read("/Users/evanchang/Desktop/10000_dataset.txt"))))
#len_candidates = len(list(find_freqsets_data1(read("/Users/evanchang/Desktop/10000_dataset.txt"))))
D = map(set, dataset)
D_list = len(list(map(set,dataset)))

def find_freq(D, D_list, candidates, min_support):
    "returns all freq itemsets that meet min_support level"
    dict_holder = {}
    for item in D:
#        print(item)
        for can in candidates:
#            print(can)
            if can.issubset(item):
                dict_holder.setdefault(can, 0) 
            #will count how many times an object appears in a mapped dataset
                dict_holder[can] += 1
#    print(dict_holder)
    num_items = float(D_list)
    freq_sets = []
    support_data = {}
#    max_freq = 0
    for key in dict_holder:
        support = dict_holder[key]/num_items
#        print(support)
        #Check to see if the frequency of the object "key" over all transactions
        #Giving up the support that can be checked against the min_support
        if support >= min_support:
            freq_sets.insert(0,key)
        #If the object meets min_support then add it to a dict and set at 0
#        elif support < min_support:
#            max_freq = max_freq +1
        #If infrequent then add one to the max_freq_set counter
        support_data[key] = support
    return freq_sets, support_data #, max_freq


def joint_set(freq_sets, k):
    #Find the joint transactions based on the freq_sets given from find_freq
    joint_freq = []
    lenlk = len(freq_sets)
    for i in range(lenlk):
        for j in range(i +1, lenlk):
            l1 = list(freq_sets[i])[:k -2]
            l2 = list(freq_sets[j])[:k -2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                joint_freq.append(freq_sets[i] | freq_sets[j])

#    print(len(joint_freq))
#    Uncomment section and use Candidate sets size 1 to get Fk-1 X Fk-1 sets
##    hold_list =[]
##    for c in joint_freq:
##        if c not in hold_list:
##            hold_list.append(c)
##    print("length of hold_list" + str(len(hold_list)))
    return  joint_freq

def apriori(dataset, minsupport = 0.05):
    #generates list of freq item sets including the joint sets
#    These are going to be the freq sets in l to 
#    C1 = list(find_candidates(read("/Users/evanchang/Desktop/10000_dataset.txt")))
#    C1 =list(find_candidates(read_data1("/Users/evanchang/Desktop/data1.txt")))
#    C1 = list(find_candidates(read_data1("/Users/evanchang/Desktop/data2.txt")))
#    C1 =list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed.csv")))
#    C1 =list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed_totaldonate.csv")))
    C1 =list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed_totaldonate_zero.csv")))
#    C1 =list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed_obamadonate.csv")))
#    C1 =list(find_candidates(read_data1("/Users/evanchang/Desktop/trimmed_obamadonate_zero.csv")))

    D = map(set,dataset)
    D_list = len(list(map(set,dataset)))
    l1, support_data = find_freq(D,D_list ,C1, minsupport)
    freq_sets = [l1]
    D = map(set,dataset)
    k = 2 
    while (len(freq_sets[k-2]) > 0):
        freq = joint_set(freq_sets[k-2],k)
        D = map(set,dataset)
        lk, supportK = find_freq(D,D_list, freq ,minsupport)
        support_data.update(supportK)
        freq_sets.append(lk)
        k +=1
    return freq_sets, support_data

l , support_data = apriori(dataset, 0.05)

def generateRules(freq_sets, support_data, min_confidence = 0.7):
    #freq_sets and the support_Data will be taken to make rules
    # It will take the freq_sets and add it to a list
    rules = []
    for i in range (1, len(freq_sets)):
        for freqSet in freq_sets[i]:
#            print(freqSet)
            h1 = [frozenset([item]) for item in freqSet]
#            print("freqSet", freqSet, "h1", h1)
            if (i> 1):
                rules_from_conseq(freqSet, h1, support_data, rules, min_confidence)
            else:
                calc_confidence(freqSet, h1, support_data, rules, min_confidence)
    return rules

def calc_confidence(freqSet, h1, support_data, rules, min_confidence):
    pruned_h = []
    for conseq in h1:
        conf = support_data[freqSet]/ support_data[freqSet - conseq]
        if conf >= min_confidence:
#            print( freqSet - conseq, "--->", conseq, "conf:", conf)
            rules.append((freqSet-conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h

rules = generateRules(l, support_data, 0.5)

def generateRules_lift(freq_sets, support_data):
    #freq_sets and the support_Data will be taken to make rules
    # It will take the freq_sets and add it to a list
    rules = []
    for i in range (1, len(freq_sets)):
        for freqSet in freq_sets[i]:
#            print(freqSet)
            h1 = [frozenset([item]) for item in freqSet]
#            print("freqSet", freqSet, "h1", h1)
            if (i> 1):
                rules_from_conseq_lift(freqSet, h1, support_data, rules)
            else:
                calc_lift(freqSet, h1, support_data, rules)
    return rules

rules_lift = generateRules(l,support_data)
def rules_from_conseq(freqSet, h1, support_data, rules, min_confidence):
    m = len(h1[0])
    if (len(freqSet) > (m+1)):
        hmp1 = joint_set(h1, m+1)
        hmp1 = calc_confidence(freqSet, hmp1, support_data, rules, min_confidence)
        if len(hmp1) > 1:
            rules_from_conseq(freqSet, hmp1, support_data, rules, min_confidence)
            

def calc_lift(freqSet, h1, support_data, rules):
    pruned_h = []
    for conseq in h1:
        lift = support_data[freqSet]/ (support_data[freqSet-conseq] * support_data[freqSet])
        if lift > 1:
#            print( freqSet - conseq, "--->", conseq, "conf:", conf)
            rules.append((freqSet-conseq, conseq, lift))
            pruned_h.append(conseq)
    return pruned_h

def rules_from_conseq_lift(freqSet, h1, support_data, rules):
    m = len(h1[0])
    if (len(freqSet) > (m+1)):
        hmp1 = joint_set(h1, m+1)
        hmp1 = calc_lift(freqSet, hmp1, support_data, rules)
        if len(hmp1) > 1:
            rules_from_conseq_lift(freqSet, hmp1, support_data, rules)

