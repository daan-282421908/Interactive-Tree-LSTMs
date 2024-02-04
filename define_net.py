import os
import pickle
import torch
import torch.nn as nn

class encode_model(nn.Module):
    def __init__(self, char_dim):
        super(encode, self).__init__()
        embedding_size = 16
        kernel = 5
        stride = 1
        padding = 2
        self.conv_out_channel = 64
        self.embedding = nn.Embedding(char_dim, embedding_size)
        self.tree_lstm = nn.Child_sum_treeLSTM()
        
    def forward(self, x):
        x = self.embedding(x) 
        x = x.permute(0,2,1)
        return x

class Child_sum_TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
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
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

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

class argument_Classification(nn.Module):
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
