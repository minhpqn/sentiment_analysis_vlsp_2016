""" For each class, count the number of errors made by our system
    and the number of each class our system predicted
"""

import os
import sys
from sklearn.metrics import confusion_matrix
import pandas as pd

def read_file(filename):
    f = open(filename, 'r')
    labels  = []
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
    pass


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: predict gold_file")
        sys.exit(1)

    predict_file = args[0]
    gold_file = args[1]

    y_pred = read_file(predict_file)
    y_true = read_file(gold_file)

    print(confusion_matrix(y_true, y_pred, labels=['POS', 'NEG', 'NEU']))
    y_actu = pd.Series(y_true, name='Gold')
    y_predi = pd.Series(y_pred, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_predi, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion)

    df_confusion = pd.crosstab(y_actu, y_predi, rownames=['Actual'], colnames=['Predicted'], normalize=True, margins=True)
    print(df_confusion)
              

    
