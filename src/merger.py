import os
import sys
import pandas as pd
from optparse import OptionParser
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

tag = {'Null': -1, 'POS': 0, 'INH': 1, 'BCK': 2}


def merge_score(d1, d2):
    if d1 == 'I' and d2 == 'B':
        return 'Null'
    elif d1 == 'I' and d2 == 'R':
        return 'INH'
    elif d1 == 'R' and d2 == 'B':
        return 'BCK'
    elif d1 == 'R' and d2 == 'R':
        return 'POS'


def compute_score(labels, golds):
    y_pred = list(map(lambda x: tag[x], labels))
    y_test = list(map(lambda x: tag[x], golds))
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average=None)
    prec_m = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_m = recall_score(y_test, y_pred, average=None, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    print('Accuray ' + str(score) + '\n')
    print('Precision (macro) ' + str(prec) + '\n')
    print('Recall (macro) ' + str(rec) + '\n')
    print('F1 score (macro) ' + str(f1) + '\n')
    print('\n')
    print('Precision (weighted) ' + str(prec_w) + '\n')
    print('Recall (weighted) ' + str(rec_w) + '\n')
    print('F1 score (weighted) ' + str(f1_w) + '\n')
    print('\n')
    print('Precision ' + str(prec_m) + '\n')
    print('Recall ' + str(rec_m) + '\n')
    print('F1 score ' + str(f1_m) + '\n')
    print('\n')
    print(str(matrix))
    print('\n')


def main():
    inh_df = pd.read_csv(options.inh, sep=';', encoding='utf-8', error_bad_lines=False)
    inh_df['pred_i'] = inh_df['pred'].apply(lambda x: 'R' if x == 0 else 'I')
    inh_df['gold_i'] = inh_df['gold'].apply(lambda x: 'R' if x == 0 else 'I')
    inh_df = inh_df[['id', 'text', 'pred_i', 'gold_i']]

    bck_df = pd.read_csv(options.bck, sep=';', encoding='utf-8', error_bad_lines=False)
    bck_df['pred_b'] = bck_df['pred'].apply(lambda x: 'R' if x == 0 else 'B')
    bck_df['gold_b'] = bck_df['gold'].apply(lambda x: 'R' if x == 0 else 'B')
    bck_df = bck_df[['pred_b', 'gold_b']]

    joined = pd.concat([inh_df, bck_df], axis=1)

    golds = []
    labels = []
    for _, row in joined.iterrows():
        golds.append(merge_score(row['gold_i'], row['gold_b']))
        labels.append(merge_score(row['pred_i'], row['pred_b']))

    new_df = joined[['id', 'text']]
    new_df['pred'] = labels
    new_df['gold'] = golds

    new_df.to_csv(os.path.join(options.output, 'final_pred.csv'), sep=';', encoding='utf-8')

    compute_score(labels, golds)


if __name__ == "__main__":
    print("One-vs-Rest\n")

    argv = sys.argv[1:]
    parser = OptionParser()

    # classifier
    parser.add_option("-d", "--dataset", help='dataset', action="store", type="string", dest="data",
                      default="../../input/subset.csv")
    parser.add_option("-i", "--inh", help='inh', action="store", type="string", dest="inh",
                      default="../../output/test_inh.csv")
    parser.add_option("-b", "--bck", help='bck', action="store", type="string", dest="bck",
                      default="../../output/test_bck.csv")
    parser.add_option("-o", "--output", help='output folder', action="store", type="string", dest="output",
                      default="../../output/")

    (options, args) = parser.parse_args()

    main()
