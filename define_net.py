import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


path_ = os.path.abspath('.')
f = open(path_+'/event_index', 'rb')
event_index = pickle.load(f)


class encode(nn.Module):
    def __init__(self, char_dim):
        super(encode, self).__init__()
        embedding_size = 16
        kernel = 5
        stride = 1
        padding = 2
        self.conv_out_channel = 64
        self.embedding = nn.Embedding(char_dim, embedding_size)
        self.conv = nn.Conv1d(embedding_size, self.conv_out_channel, kernel, stride, padding)
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x) 
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.pooling(x)
        x = x.view(1,-1,self.conv_out_channel) 
        return x

class treeLSTM(nn.Module):
    def __init__(self, word_dim, entity_dim):
        super(treeLSTM, self).__init__()
        self.hidden_size = 128
        self.num_hidden = 1
        embedding_size = 192
        entity_embedding_size = 16
        conv_out_channel = 64
        self.word_embedding = nn.Embedding(word_dim, embedding_size)
        self.entity_embedding = nn.Embedding(entity_dim, entity_embedding_size)
        self.lstm = nn.LSTM(embedding_size+entity_embedding_size+conv_out_channel,
                            self.hidden_size,
                            self.num_hidden,
                            bidirectional=True,
                            dropout=0)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, inputs, hidden):
        words, entitys, char_features = inputs
        length = words.size()[1]
        word_emb = self.word_embedding(words)
        word_emb = self.dropout(word_emb)
        word_emb = word_emb.permute(1,0,2) 
        entity_emb = self.entity_embedding(entitys) 
        entity_emb = self.dropout(entity_emb)
        entity_emb = entity_emb.permute(1,0,2) 
        input_ = torch.cat((word_emb, entity_emb, char_features), 2) 
        output, hidden = self.lstm(input_, hidden) 
        return output, hidden, entity_emb

    def initHidden(self,num_layers=2,batch=1):
        return (Variable(torch.zeros(num_layers*self.num_hidden, batch, self.hidden_size)),
                Variable(torch.zeros(num_layers*self.num_hidden, batch, self.hidden_size)))

class Trigger_Recognition(nn.Module):
    def __init__(self, event_dim):
        super(Trigger_Recognition, self).__init__()
        self.event_dim = event_dim
        self.BiLSTM_hidden_size = 2*128 
        self.embedding_size = 16
        tmp_dim = 128
        self.event_embedding = nn.Embedding(event_dim, self.embedding_size)
        self.linear = nn.Linear(self.BiLSTM_hidden_size+self.embedding_size, tmp_dim)
        self.linear2 = nn.Linear(tmp_dim, self.event_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, output, target=[], infer = True, event_index = event_index['NONE']):
        ner_output = []
        ner_emb = []
        ner_index = []

        event_index = torch.LongTensor([[event_index]]) 
        event_emb = self.event_embedding(Variable(event_index)) 
        event_emb = event_emb.view(-1,self.embedding_size)

        for i in range(0,bilstm_output.size()[0]):
            inputs = torch.cat((bilstm_output[i], event_emb), 1) 
            inputs = self.dropout(inputs)
            hidden = F.leaky_relu(self.linear(inputs)) 
            hidden = self.linear2(hidden)
            ner_softmax = self.softmax(hidden) 

            row = list(ner_softmax[0])
            for k in range(0, len(row)):
                row[k] = float(row[k].detach())

            if infer:
                event_index = row.index(max(row))
            else:
                event_index = target[i].item()

            event_index = torch.LongTensor([[event_index]])
            event_emb = self.event_embedding(Variable(event_index)) 
            event_emb = event_emb.view(-1, self.embedding_size) 

            ner_index.append(event_index)
            ner_output.append(ner_softmax)
            ner_emb.append(event_emb.view(-1, 1, self.embedding_size))

        ner_index = torch.cat(ner_index, 0)
        ner_index = ner_index.view(-1)
        ner_output = torch.cat(ner_output, 0)
        ner_emb = torch.cat(ner_emb, 0)
        return ner_output, ner_emb, ner_index

    def get_event_embedding(self, event_index):

        event_index = torch.LongTensor([[event_index]])
        event_emb = self.event_embedding(Variable(event_index))
        return event_emb

class Relation_Classification(nn.Module):
    def __init__(self, relation_dim):
        super(Relation_Classification, self).__init__()
        tmp_dim = 128
        self.BiLSTM_hidden_size = 2*128 
        self.relation_dim = relation_dim
        self.ner_embedding = 16
        self.position_dim = 8
        self.max_position = 16
        self.conv_out_dim = 128

        self.position_embedding = nn.Embedding(2*self.max_position+1, self.position_dim)
        self.empty_embedding = nn.Embedding(1, self.BiLSTM_hidden_size+2*self.ner_embedding) #+2*self.ner_embedding
        self.max_pooling = nn.AdaptiveMaxPool1d(1)


        self.linear = nn.Linear(6*self.ner_embedding+3*self.BiLSTM_hidden_size+self.position_dim+self.conv_out_dim, tmp_dim)
        self.linear2 = nn.Linear(tmp_dim, self.relation_dim)

        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, src,middle,dst,reverse_flag,event_flag):
        length_middle = torch.LongTensor(1,1).zero_()
        length_middle[0][0] = min(middle.size()[0]+1, self.max_position)
        if reverse_flag:
            length_middle[0][0] = - length_middle[0][0]
        length_middle[0][0] = length_middle[0][0] + self.max_position
        pe_middle = self.position_embedding(Variable(length_middle))
        pe_middle = pe_middle.view(1,-1,1)

        src = src.permute(1,2,0) 
        src = self.max_pooling(src) 
        dst = dst.permute(1,2,0)
        dst = self.max_pooling(dst)

        if middle.size()[0] == 0 :
            middle = self.empty_embedding(torch.LongTensor([[0]]))
        middle = middle.permute(1,2,0)
        middle_p = self.max_pooling(middle)

        if not reverse_flag :
            middle_c = self.conv_3(middle)
        else:
            middle_c = self.conv_3r(middle)

        middle_c = self.max_pooling(middle_c)

        x = torch.cat((src,middle_p,middle_c,dst,pe_middle),1) 
        x = x.view(1,-1) 
        x = self.dropout(x)
        x = F.leaky_relu(self.linear(x)) 
        x = self.linear2(x)
        x = self.softmax(x)

        return x
