""" Evaluate experimental result
"""

import os
import sys
from sklearn.metrics import accuracy_score,classification_report, precision_score, recall_score, f1_score

def read_file(filename):
    f = open(filename, 'r')
    labels = []
    i = 1
    for line in f:
        line = line.strip()
        if line == '':
            continue
        if i % 2 == 0:
            if line == 'NEUTRAL':
                line = 'NEU' 
            labels.append(line)
        i += 1
    f.close()

    return labels

def main(args):
    if len(args) < 2:
        print("usage: predict gold")
        sys.exit()

    predict_file = args[0]
    gold_file = args[1]

    y_pred = read_file(predict_file)
    y_true = read_file(gold_file)

    print("Accuracy: %2.2f" % (100 * accuracy_score(y_true, y_pred)))
    for lb in ['POS', 'NEG', 'NEU']:
        if lb == 'POS':
            print("Positive:")
        elif lb == 'NEG':
            print("Negative:")
        else:
            print("Neutral")
            
        labels = [lb]
        p  = 100 * precision_score(y_true, y_pred,average='macro',
                                   labels=labels)
        r  = 100 * recall_score(y_true, y_pred,average='macro',labels=labels)
        f1 = 100 * f1_score(y_true, y_pred,average='macro',labels=labels)
        print("Precision: %2.2f Recall: %2.2f F1: %2.2f" %(p, r, f1))

    print("\n== Classification report ==")    
    print( classification_report(y_true, y_pred) )

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
