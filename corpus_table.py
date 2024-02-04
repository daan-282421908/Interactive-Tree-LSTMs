import os
import re
import csv
from nltk import sent_tokenize


def get_pos(file_path):
    with open(file_path,'r') as f:
        title = f.readline()
    with open(file_path,'r') as f:
        abstract = f.readline()
       
    if asf == '\n':
        asf = src.readline() 

    title = title.replace('\n','')
    abstract = abstract.replace('\n','')
    list.append(title)
    list.extend(sent_tokenize(abstract))
    
    for i,s in enumerate(list) :
        ids = re.split('([ -/().,:;])',s)
        for j,w in enumerate(ids) :
            if w != ' ' and w != '':
                position_dic_word[position] = w
                if i != 0 :
                    if j == len(words)-2 :
                        end.add(position)
                else :
                    if j >= len(words)-4 and w == '.' :
                        position_dic_end.add(position)
            position += len(w)
        position += 1
        
    return position_of_word,position_of_end

def get_event_info(file_path,position_of_word):
    with open(file_path,'r') as f:
        event_notation = dict()
        event_index = dict()
        
        line = src.readline()
      
        while line:
            if line[0] != 'T' :
                line = src.readline()
                continue
          
            words = re.split('[ \t]',line)
            index = words[0]
            type_ = words[1]
            start = int(words[2])
            end = int(words[3])
    
            i = 0
            while position2word[i+1][0] <= start:
                i += 1
            start = position2word[i][0]
    
            event_list = []
            for p2w in position2word:
                p = p2w[0]
                if ( p >= start and p < end ) :
                event_list.append(p)

            if len(event_list) == 1 :
                event = event_list[0]
                event_notation[event] = type_ + '-Unit'
            else :
                event = event_list[0]
                event_notation[event] = type_ + '-Begin'
                event = event_list[-1]
                event_notation[event] = type_ + '-Last'
                for i in range(1,len(event_list)-1):
                    event = event_list[i]
                    event_notation[event] = type_ + '-Inside'
            for event in event_list :
                event_index[event] = index
        line = f.readline()
    return event_notation, event_index

txt_path = os.path.abspath('.') + '/txt' 

for fpath,_,files in os.walk(txt_path):
    for fl in files:
        file_path = os.path.join(fpath,fl)
        position_of_word,position_of_end = get_pos(file_path)

        for p in sorted(position2word.keys()):
            position2word = (p,position2word[p])

        file_path = file_path.replace('txt','a1')
        entity_notation,entity_index = get_pos(file_path,position2word)

        file_path = file_path.replace('a1','a2')
        event_notation,event_index = get_event_info(file_path,position2word) #dict(),dict()
        
        file_path = file_path.replace('/a2', '/table')
        file_path = file_path.replace('.a2', '.csv')

        with open(file_path,'w') as f:
            writer = csv.writer(csvfile)
            writer.writerow(['position', 'word', 'entity_index', 'entity_notation','event_index', 'event_notation', 'is_end'])

            csvfile = open(file_path,'a+')
            writer = csv.writer(csvfile)
            
            for p2w in position2word:

                p = p2w[0]
                w = p2w[1]

                if p in entity_index:
                    e_i = entity_index[p]
                    e_n = entity_notation[p]
                else:
                    e_i, e_n = '',''

                if p in event_index:
                    ev_i = event_index[p]
                    ev_n = event_notation[p]
                else:
                    ev_i,ev_n = '',''

                if p in position2end:
                    flag = True
                else:
                    flag = False
                    
                writer.writerow([p, w, e_i, e_n, ev_i, ev_n, flag])
                
            csvfile.close()
