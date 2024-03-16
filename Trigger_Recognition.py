import os
import pickle
import torch
import torch.nn as nn

class produce_model(nn.Module):
    def __init__(self, char_dim):
        super(encode, self).__init__()
        self.embedding = nn.Embedding(char_dim, embedding_size)
        self.tree_lstm = nn.Child_sum_treeLSTM()
        
    def forward(self, x):
        x = self.embedding(x) 
        x = x.permute(0,2,1)
        return x

class Child_sum_TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

class Trigger_Recognition(nn.Module):
    def __init__(self, event_dim):
        super(Trigger_Recognition, self).__init__()
        self.event_dim = event_dim
        self.hidden_size = 2*128 
        self.embedding_size = 16
        tmp_dim = 128
        self.event_embedding = nn.Embedding(event_dim, self.embedding_size)
        self.linear = nn.Linear(self.hidden_size+self.embedding_size, tmp_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.18)

    def forward(self, output, target=[], infer = True, event_index = event_index['NONE']):
        event_index = torch.LongTensor([[event_index]]) 
        event_emb = self.event_embedding(Variable(event_index)) 
        event_emb = event_emb.view(-1,self.embedding_size)

        for i in range(0,output.size()[0]):
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
