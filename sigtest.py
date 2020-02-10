import pandas as pd
import os
import pickle
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script contains all utility function to perform significance testing
(McNemar's test) on the test output of the models across feature sets.
This script must bve run after the test run of the classifier.
"""

__author__ = "André Bittar"
__copyright__ = "Copyright 2020, André Bittar"
__credits__ = ["André Bittar"]
__license__ = "GPL"
__email__ = "andre.bittar@kcl.ac.uk"


def do_all_mcnemar():
    d = {}

    fs1, fs2, a, p = do_mcnemar('STRUCT', 'GATE')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('STRUCT', 'TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('STRUCT', 'GATE+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('STRUCT', 'STRUCT+GATE')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('STRUCT', 'STRUCT+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('STRUCT', 'STRUCT+GATE+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('GATE', 'TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('GATE', 'GATE+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('GATE', 'STRUCT+GATE')  # fail p-value=0.206
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('GATE', 'STRUCT+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('GATE', 'STRUCT+GATE+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('TFIDF', 'GATE+TFIDF')  # fail p-value=0.400
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('TFIDF', 'STRUCT+GATE')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('TFIDF', 'STRUCT+TFIDF')  # fail p-value=0.065
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('TFIDF', 'STRUCT+GATE+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('GATE+TFIDF', 'STRUCT+GATE')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('GATE+TFIDF', 'STRUCT+TFIDF')  # fail p-value=0.077
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('GATE+TFIDF', 'STRUCT+GATE+TFIDF')  # pass p-value=0.004
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('STRUCT+GATE', 'STRUCT+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('STRUCT+GATE', 'STRUCT+GATE+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    fs1, fs2, a, p = do_mcnemar('STRUCT+TFIDF', 'STRUCT+GATE+TFIDF')
    tmp = d.get(fs1, {})
    tmp[fs2] = p
    d[fs1] = tmp

    df = pd.DataFrame.from_dict(d)

    return df


def do_mcnemar(fs1, fs2):
    path = './output/'

    pin1 = os.path.join(path + 'SVM_gold_versus_pred_30d_' + fs1 + '.pickle')
    pin2 = os.path.join(path + 'SVM_gold_versus_pred_30d_' + fs2 + '.pickle')

    df1 = pickle.load(open(pin1, 'rb'))
    df2 = pickle.load(open(pin2, 'rb'))

    df1['correct'] = df1['gold_' + fs1] == df1['pred_' + fs1]
    df2['correct'] = df2['gold_' + fs2] == df2['pred_' + fs2]

    table = np.zeros((2,2))

    """
                     | c2_correct | c2_incorrect
        c1_correct   |            |
        c1_incorrect |            |
    """

    for i in range(len(df1)):
        v1 = df1.iloc[i].correct
        v2 = df2.iloc[i].correct

        # c1_correct, c2_incorrect
        if v1 and not v2:
            table[0, 1] += 1
        # c1_correct, c2_correct
        if v1 and v2:
            table[0, 0] += 1
        # c1_incorrect, c2_correct
        if not v1 and v2:
            table[1, 0] += 1
        # c1_incorrect, c2_incorrect
        if not v1 and not v2:
            table[1, 1] += 1

    result = mcnemar(table, exact=True)

    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0) for', fs1, fs2)
    else:
        print('Different proportions of errors (reject H0) for', fs1, fs2)

    return fs1, fs2, alpha, result.pvalue


def t_test():
    path = './output'

    metric = 'f1_score'

    global_scores = {}

    for file in [f for f in os.listdir(path) if f.endswith('.pickle')]:
        if 'SVM' not in file:
            continue

        feature_set = None

        if 'STRUCT+GATE+TFIDF' in file:
            feature_set = 'STRUCT+GATE+TFIDF'
        elif 'STRUCT+GATE' in file:
            feature_set = 'STRUCT+GATE'
        elif 'STRUCT+TFIDF' in file:
            feature_set = 'STRUCT+TFIDF'
        elif 'GATE+TFIDF' in file:
            feature_set = 'GATE+TFIDF'
        elif 'GATE' in file:
            feature_set = 'GATE'
        elif 'STRUCT' in file:
            feature_set = 'STRUCT'
        elif 'TFIDF' in file:
            feature_set = 'TFIDF'

        pin = os.path.join(path, file)
        res = pickle.load(open(pin, 'rb'))
        results = res['results']

        best = results.get('rank_test_' + metric)[0]

        scores = []

        for i in range(10):
            key = 'split' + str(i) + '_test_' + metric
            split_score = results[key][best]
            scores.append(split_score)

        global_scores[feature_set] = scores

    s1 = global_scores['STRUCT+TFIDF']
    s2 = global_scores['GATE+TFIDF']

    stat, p = ttest_ind(s1, s2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0) for', feature_set)
    else:
        print('Different distributions (reject H0) for', feature_set)


def plot_heatmap(df):
    sns.set(style='white')
    sns.set(font_scale=1.75)
    plt.figure(figsize=(15, 10))
    hm = sns.heatmap(df, annot=True, robust=True, cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True))
    plt.xticks(rotation=45, ha='center')
    plt.yticks(rotation=45)
    hm.figure.savefig('output/mcnemar_heatmap.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    df = do_all_mcnemar()
    df = df.loc[['GATE', 'TFIDF', 'GATE+TFIDF', 'STRUCT+GATE', 'STRUCT+TFIDF', 'STRUCT+GATE+TFIDF']].T
    df.to_pickle('output/mcnemar_grid.pickle')
    plot_heatmap(df)