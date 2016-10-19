""" Sample errors made by our system
    For each class (pos, neg, neu), sample N incorrect cases
    in which there are N/2 examples for each of other classes
"""

import os
import sys
import random
import re

def read_file(filename):
    f = open(filename, 'r')
    labels  = []
    reviews = []
    i = 1
    for line in f:
        line = line.strip()
        if line == '':
            continue
        if i % 2 == 0:
            if line == 'NEUTRAL':
                line = 'NEU' 
            labels.append(line)
        else:
            reviews.append(line)
        i += 1
    f.close()

    return reviews, labels

def main(args):
    if len(args) < 3:
        print("usage: predict gold_file N")
        sys.exit(1)

    random.seed(42)

    predict_file = args[0]
    gold_file = args[1]
    nsamples  = int(args[2])

    reviews, y_pred = read_file(predict_file)
    reviews, y_true = read_file(gold_file)

    pos_indexes = []
    neg_indexes = []
    neu_indexes = []

    for i in range(len(y_pred)):
        pred = y_pred[i]
        gold = y_true[i]
        if gold == pred:
            continue
        if gold == 'POS':
            pos_indexes.append(i)
        elif gold == 'NEG':
            neg_indexes.append(i)
        else:
            neu_indexes.append(i)
     
    pos_sample = random.sample(pos_indexes, nsamples)
    neg_sample = random.sample(neg_indexes, nsamples)
    neu_sample = random.sample(neu_indexes, nsamples)

    cols = ['id', 'GOLD_LABEL', 'PREDICT', 'CONTENT']
    print("\t".join(cols))

    for i,arr in enumerate([pos_sample, neg_sample, neu_sample]):
        for j in arr:
            text = reviews[j]
            text = re.sub(r"\s+", " ", text)
            data = [str(j), y_true[j], y_pred[j], text]
            print("\t".join(data))

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)

