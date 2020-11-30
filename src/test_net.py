import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from train_net import NN_LM
import pickle

# converts data to correct input type
def convert2ids(data, word2id):
	ids = []

	for word in data:
		id = word2id.get(word)
		if id == None:
			ids.append(word2id["unk"])
		else:
			ids.append(id)

	return ids

def test_model(test_corpus, word2id, id2word):
    for sentence in test_corpus:
        data = sentence.split()
        data_ids = convert2ids(data, word2id)
        data_ids = torch.LongTensor(data_ids)

        output = model(data_ids)
        output = output.detach().numpy()
        
        next = np.argmax(output)

        print(sentence, ":", id2word[next])

#load test data

test_corpus = [
'reached for the',
'the empty roof',
'force which he',
'the black empathy',
'rest of the',
'thought , and',
'work vacuity specific',
'he off to'
]

model = NN_LM(77)
model.load_state_dict(torch.load("trained_model.pt"))
model.eval()

file = open("word2id.pkl", "rb")
word2id = pickle.load(file)
file.close()

file = open("id2word.pkl", "rb")
id2word = pickle.load(file)
file.close()

test_model(test_corpus, word2id, id2word)


