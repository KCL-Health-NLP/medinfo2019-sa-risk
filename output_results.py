import argparse
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np

"""
This script is used to generate plots of the results obtained. Results must have
been previously generated in the 'output' directory.
"""

__author__ = "André Bittar"
__copyright__ = "Copyright 2020, André Bittar"
__credits__ = ["André Bittar"]
__license__ = "GPL"
__email__ = "andre.bittar@kcl.ac.uk"


def plot_test_scores(path):
    d_prf = {}

    for file in [f for f in os.listdir(path) if f.endswith('txt') and 'SVM_' in f]:
        pin = os.path.join(path, file)

        match = re.search('(SVM|NB|RF|MLP)', pin)
        model = match.group(1)

        match = re.search('30d_([^_]+)_', pin)
        fs = match.group(1)

        start = False
        for line in open(pin, 'r'):
            if 'precision' in line:
                start = True
            if start and '          1' in line:
                match = re.search('^\s+1\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)', line)
                if match is not None:
                    p = float(match.group(1))
                    r = float(match.group(2))
                    f = float(match.group(3))
                    d_prf[fs] = {'Precision': p, 'Recall': r, 'F-score': f}

    cols = ['STRUCT', 'GATE', 'TFIDF', 'STRUCT+GATE', 'STRUCT+TFIDF', 'GATE+TFIDF', 'STRUCT+GATE+TFIDF']
    df_prf = pd.DataFrame.from_dict(d_prf)
    df_prf = df_prf[cols]
    print(df_prf)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_xticks(np.arange(len(df_prf)))
    ax.set_xticklabels(df_prf.index, rotation=0, ha='center')
    p = df_prf.loc[['Precision', 'Recall', 'F-score']].plot(kind='bar', use_index=True, ax=ax, ylim=(0, 1), fontsize=12, figsize=(8, 5),
                colormap='Set3', rot=0, linewidth=2)  # bone
    bars = ax.patches
    hatches = ''.join(h * len(df_prf) for h in '/\\o-x.+')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='best', bbox_to_anchor=(1, 1), ncol=2)
    plt.title('Performance for ' + model + ' model')
    plt.savefig(path + '/prf_vs_feature_set.png', bbox_inches='tight')
    plt.show()


def plot_auroc(path):
    d_auc = {}

    for file in [f for f in os.listdir(path) if f.endswith('txt') and 'SVM' in f]:
        pin = os.path.join(path, file)

        match = re.search('(SVM|NB|RF|MLP)', pin)
        model = match.group(1)

        match = re.search('30d_([^_]+)_', pin)
        fs = match.group(1)

        start = False
        for line in open(pin, 'r'):
            if 'precision' in line:
                start = True
            if start and 'AUC' in line:
                match = re.search('AUC ROC:\s+([0-9\.]+)', line)
                if match is not None:
                    score = float(match.group(1))
                    d_auc[fs] = [score]
                else:
                    raise ValueError('No match!')

    df_auc = pd.DataFrame.from_dict(d_auc).T
    df_auc = df_auc.reindex(['STRUCT', 'GATE', 'TFIDF', 'STRUCT+GATE', 'STRUCT+TFIDF', 'GATE+TFIDF', 'STRUCT+GATE+TFIDF'])
    df_auc.columns = ['AUROC']
    #df_auc = df_auc.sort_values('AUROC')
    df_auc.to_excel(path + '/auroc.xlsx')
    print(df_auc)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('AUROC')
    ax.set_xticks(np.arange(len(df_auc)))
    ax.set_xticklabels(df_auc.index, rotation=0, ha='right')
    p = df_auc.plot(kind='bar', use_index=True, ax=ax, ylim=(0, 1), fontsize=12, figsize=(8, 5),
                colormap='Set3', rot=45, linewidth=2)  # bone
    #p.set_axis_bgcolor('#F8F8F8')

    bars = ax.patches
    hatches = '/\\o-x.+'
    colors = ['#8EDBCC', '#BFC1DD', '#81B2D3', '#B6E16D', '#D7D9DC', '#CCEABF', '#FCF66E']

    for bar, hatch, color in zip(bars, hatches, colors):
        bar.set_facecolor(color)
        bar.set_hatch(hatch)
    ax.legend().remove()

    plt.ylim(ymin=0.5)
    ax.set_ylim(bottom=0.5)
    plt.title('AUROC for ' + model + ' model')
    plt.savefig(path + '/auroc_vs_feature_set.png', bbox_inches='tight')
    plt.show()


def plot_auroc_sns(path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="ticks", color_codes=True)

    d_auc = {}

    for file in [f for f in os.listdir(path) if f.endswith('txt') and 'SVM' in f]:
        pin = os.path.join(path, file)

        match = re.search('(SVM|NB|RF|MLP)', pin)
        model = match.group(1)

        match = re.search('30d_([^_]+)_', pin)
        fs = match.group(1)

        start = False
        for line in open(pin, 'r'):
            if 'precision' in line:
                start = True
            if start and 'AUC' in line:
                match = re.search('AUC ROC:\s+([0-9\.]+)', line)
                if match is not None:
                    score = float(match.group(1))
                    d_auc[fs] = [score]
                else:
                    raise ValueError('No match!')

    df_auc = pd.DataFrame.from_dict(d_auc).T
    df_auc = df_auc.reindex(['STRUCT', 'GATE', 'TFIDF', 'STRUCT+GATE', 'STRUCT+TFIDF', 'GATE+TFIDF', 'STRUCT+GATE+TFIDF'])
    df_auc.reset_index(inplace=True)
    df_auc.columns = ['Feature Set', 'AUROC']
    #df_auc = df_auc.sort_values('AUROC')
    df_auc.to_excel(path + '/auroc.xlsx')
    print(df_auc)

    f = sns.catplot(x='Feature Set', y='AUROC', jitter=False, data=df_auc)
    f.set_xticklabels(rotation=45, ha='right')
    plt.title('AUROC for ' + model + ' model')
    plt.savefig(path + '/auroc_vs_feature_set_sns.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eHOST-IT: extract scores from \
                                     saved tuned model files for plotting and \
                                     significance testing.')
    parser.add_argument('-i', '--input_path', metavar='input_path', type=str, 
                        nargs=1, help='specify the path to data for input.', 
                        required=True)
    parser.add_argument('-p', '--plot_type', metavar='plot_type', type=str, 
                        nargs=1, help='specify the type of plot to generate', 
                        choices=['auroc', 'scores'], required=True)
    
    args = parser.parse_args()
    
    if os.path.isdir('./' + args.input_path[0]):
        if args.plot_type[0] == 'auroc':
            plot_auroc('./' + args.input_path[0])
            #plot_auroc_sns(args.input_path[0])
        else:
            plot_test_scores('./' + args.input_path[0])
    else:
        parser.error('-- Invalid path: ' + args.input_path[0] + '. The path ' + \
                     'must be a directory.')