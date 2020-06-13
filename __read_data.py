"""
  Reading data module
"""

import os
import numpy as np

def get_w(w, tag=False):
    if tag:
        return w.split('/')[0]
    else:
        return w
    
def read_train_data(ws=True, tag=False):
    datadir = './data/SA2016-training-data-ws'
    filenames = [
        'train_positive_tokenized.txt',
        'train_negative_tokenized.txt',
        'train_neutral_tokenized.txt',
    ]
        
    if not ws:
        datadir = './data/SA2016-training_data'
        filenames = ['SA-training_positive.txt',
                     'SA-training_negative.txt',
                     'SA-training_neutral.txt',
        ]

    label_codes = ['POS', 'NEG', 'NEU']

    sentences = []
    labels = []
    for i, filename in enumerate(filenames):
        path  = os.path.join(datadir, filename)
        label = label_codes[i]
        f = open(path, 'r')
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            words = [ get_w(w, tag) for w in line.split()]
            sentences.append( ' '.join( words ) )
            labels.append(label)
            
    y = np.array(labels)

    return (sentences, y)
    
def read_test_data(ws=True, tag=False):
    """ Read test data from file
    Return array of sentences
    """

    datafile = './data/test_tokenized.txt'

    if not ws:
        datafile = '.data/test_raw.txt'

    sentences = []
    with open(datafile, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            words = [ get_w(w, tag) for w in line.split()]
            sentences.append( ' '.join( words ) )

    return sentences        
