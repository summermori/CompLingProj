import os
from collections import defaultdict

# returns an opened file obj(read only), specificy its name as string input. 
# Assumes directory structure same as seen on branch "data-preprocess"
def open_file(name):
    os.chdir('../')
    os.chdir('datasets')
    file = open(name, "r")
    return file

# takes open file and up_to int that specifies how many lines we want to read. 
# returns list of original lines and then tuples sorted in descending frequency
def format_and_rank(file, up_to):
    dictionary = defaultdict(int)
    lines = file.readlines()
    lines = lines[:up_to]
    for idx, line in enumerate(lines):
        # formatting
        temp = ''
        split = line.split(",")
        for i in range(len(split)):
            if (i != len(split) - 1):
                temp += split[i] + ' ,'
            else:
                temp += split[i]
        temp = temp[:len(temp) - 2] + ' .'
        lines[idx] = temp

        # adding to dict
        splot = temp.split(" ")
        for word in splot:
            dictionary[word] += 1
        ret = []
        for k, v in sorted(dictionary.items(), key = lambda x: x[1], reverse = True):
            ret.append((k, v))
    return (lines, ret)

file = open_file("wikisent2.txt")
lines, frequencies = format_and_rank(file, 100000)



