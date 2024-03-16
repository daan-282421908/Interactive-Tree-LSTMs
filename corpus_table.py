import os
import re
import csv
from nltk import sent_tokenize

def pos(file_path):
    with open(file_path,'r') as f:
        title = f.read()
       
    if asf == '\n':
        asf = src.readline() 
    title = title.replace('\n','')
    return title

def get_event_info(file_path,position_of_word):
    with open(file_path,'r') as f:
        event_notation = dict()
        event_index = dict()
        line = src.readline()
    return line

for fpath,_,files in os.walk(path):
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

