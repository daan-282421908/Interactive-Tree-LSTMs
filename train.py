import os
import sys
import math
import torch

def embeding(filename,index, embedding):
    my_word2vec = gensim.models.Word2Vec.load(filename)
    not_found = 0
    for key in index.keys():
	pretrain_vector = my_word2vec.wv[key].tolist()
	index = word_index[key]
	embedding[index] = pretrain_vector[:]
    return word_embedding

if __name__ == '__main__':
       for epoch in range(50):
	    running_loss = 0.0
	    loss1, loss2, loss3, loss4 = 0.0, 0.0, 0.0, 0.0
	    addition_count = 0.0
	    tr_validation, rc_validation, ee_validation = [], [], []
	
	for i, j in enumerate(trainloader, 0):
	    loss = 0
	    optimizer.zero_grad()
	    for data in batch:  
		input_word, input_entity, input_char, target, entity_loc, event_loc, relation, event_para, modification, fname = data
		input_word, input_entity, target = Variable(input_word), Variable(input_entity), Variable(target.view(-1))  # L of indices
	    loss.backward()
	    optimizer.step()
	    running_loss += loss.item()

