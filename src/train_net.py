import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

class NN_LM(nn.Module):

	def __init__(self,vocab_size):
		self.hidden_size = 15
		self.emb_size = 5
		super(NN_LM, self).__init__()
		self.emb = nn.Embedding(vocab_size, self.emb_size)
		self.H_lyr = nn.Linear(3*self.emb_size, self.hidden_size)
		self.U_lyr = nn.Linear(self.hidden_size,vocab_size)

	def forward(self,inputs):
		x = self.emb(inputs).view((-1,3*self.emb_size))
		hid = torch.tanh(self.H_lyr(x))
		out = F.log_softmax(self.U_lyr(hid),dim=1)
		return out


def process_training_data(corpus_text):
    """Tokenizes a text file."""
    # Create the model's vocabulary and map to unique indices
    word2id = {}
    id2word = []

    for sentence in corpus_text:
	    for word in sentence.split():
	        if word not in word2id:
	            id2word.append(word)
	            word2id[word] = len(id2word) - 1

    # Convert string of text into string of IDs in a tensor for input to model
    input_as_ids = []
    for sentence in corpus_text:
    	sentence_as_ids = []
	    for word in sentence.split():
	        sentence_as_ids.append(word2id[word])
	    input_as_ids.append(sentence_as_ids)
    # final_ids = torch.LongTensor(input_as_ids)

    return input_as_ids,word2id,id2word

# This function runs the training of the model.
def run_training(train_data,id2word):
	num_training_epochs = 50

	## Initialize NNLM
	nnlm_model = NN_LM(len(id2word))
	## Define the optimizer as Adam
	nnlm_optimizer = optim.Adam(nnlm_model.parameters(), lr=.001)
	## Define the loss function as negative log likelihood loss
	criterion = nn.NLLLoss()

	# Run training for specified number of epochs
	print('\nNNLM initializing...\n')
	for epoch in range(num_training_epochs):
		# Move through data one word (ID) at a time, extracting a window of three
		# context words, and a target fourth word for the model to predict
		for sentence in train_data:
			for i in range(2:len(sentence) - 1):
				input_context = torch.LongTensor(sentence[:i-1])
				target_word = torch.LongTensor([sentence[i]])

				# Run model on input, get loss, update weights
				nnlm_optimizer.zero_grad()
				output = nnlm_model(input_context)
				loss = criterion(output, target_word)
				loss.backward()
				nnlm_optimizer.step()

	return nnlm_model

# need to pull training corpus
train_corpus = """
okay, he thought ; i'm off to work . he reached for the doorknob that opened the way out into the unlit hall ,
then shrank back as he glimpsed the vacuity of the rest of the building . it lay in wait for him, out here ,
the force which he had felt busily penetrating his specific apartment . god , he thought , and reshut the door .
he was not ready for the trip up those clanging stairs to the empty roof where he had no animal . the echo of
himself ascending : the echo of nothing . time to grasp the handles , he said to himself , and crossed the living
room to the black empathy box .
"""

train_data,word2id,id2word = process_training_data(train_corpus)
model = run_training(train_data,id2word)

torch.save(model.state_dict(), "trained_model_wiki.pt")

file = open("word2id.pkl", "wb")
pickle.dump(word2id, file)
file.close()

file = open("id2word.pkl", "wb")
pickle.dump(id2word, file)
file.close()

print("Vocab size: ", len(id2word))
