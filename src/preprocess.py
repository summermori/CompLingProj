import os
from collections import defaultdict
import pickle

# returns an opened file obj(read only), specificy its name as string input. 
# Assumes directory structure same as seen on branch "data-preprocess"
def open_file(name):
    os.chdir('../')
    os.chdir('datasets')
    file = open(name, "r")
    return file

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# takes open file and up_to int that specifies how many lines we want to read. 
# returns list of strings, each string is an original line(formatted so that all are lowercase and punctuation is spaced) and 
# then tuples("word", frequency) sorted in descending order
def format_and_rank(file):
    dictionary = defaultdict(int)
    lines = file.readlines()
    processed = []
    nums = 0
    for line in lines:
        if hasNumbers(line) == True:
            nums = 1
        # formatting
        temp = ''
        split = line.split(",")
        for i in range(len(split)):
            if (i != len(split) - 1):
                temp += split[i] + ' ,'
            else:
                temp += split[i]
        temp = temp[:len(temp) - 2] + ' .'

        # adding to dict
        splot = temp.split(" ")
        if nums == 0:
            for word in splot:
                dictionary[word] += 1
            processed.append(temp)
        elif nums == 1:
            elim_nums = ""
            for  word in splot:
                dictionary[word] += 1
                if hasNumbers(word) == True:
                    elim_nums += "nmbr "
                else:
                    elim_nums += word + " "
            processed.append(elim_nums)
            nums = 0
    # generating frequencies
    ret = []
    for k, v in sorted(dictionary.items(), key = lambda x: x[1], reverse = True):
        ret.append((k, v))
    return (processed, ret)

file = open_file("wikisent2.txt")
lines, frequencies = format_and_rank(file)
out_1 = open("wiki_processed_data.pkl", 'wb')
out_2 = open("wiki_frequencies.pkl", 'wb')
pickle.dump(lines, out_1)
pickle.dump(frequencies, out_2)
out_1.close()
out_2.close()



# test = "12 peaches , 3-6 29 1 in my mouth ."
# split = test.split(" ")
# elim_nums = ""
# for word in split:
#     if hasNumbers(word) == True:
#         elim_nums += "nmbr "
#     else:
#         elim_nums += word + " "
# print(elim_nums)

