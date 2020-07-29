# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:39:13 2018

@author: ABittar, andre.bittar@kcl.ac.uk

This script contains code needed to run the experiments described in the 
paper "Text Classification to Inform Suicide Risk Assessment in Electronic 
Health Records", Bittar A, Velupillai S, Roberts A, Dutta R., from MedInfo 2019 
(https://www.ncbi.nlm.nih.gov/pubmed/31437881).

To run tuning:
    1. declare a configuration as tuple of the form
       (model_name, model_object, parameters):
       - model_name: a string containing the name of the model, e.g. SVM, KNN
       - model_object: a scikit-learn instantiated model object, e.g LinearSVC()
       - parameters: a dictionary specifying the parameters to tune and their
         respective values.
    2. declare a list of periods: these are just integers that specify the 
       number of days prior to a suicide attempt to be considered, i.e. the 
       temporal window that is represented in the data set. 
       The only currently possible value is 30, although other values can be 
       used (originally 90 and 180 were to be used).
    3. declare a feature set (list of features): choose from the following:
       - STRUCT: structured fields only
       - GATE: GATE fields only
       - TFIDF: full-text TFIDF features only
       - STRUCT+GATE: structured and GATE features
       - STRUCT+TFIDF: strucutred and TFIDF features
       - GATE+TFIDF: GATE and TFIDF features
       - STRUCT+GATE+TFIDF: all features
       - GLOVE: represent text with GloVe embeddings (not in article)
    4. run script with option -r (--run_type) tune
       - results will show all specified feature sets and best tuned parameters

To run test with tuned model paramters:
    1. run_test() contains the best configuration(s) found by tuning.
      - run script with option -r (--run_type) test

TODO:
    - logger for message output
    - enternalise configuration to file
    - set command line arguments to specify outputs (logs, plots, etc.)
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import spacy
import sys
import time

from collections import defaultdict
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.extmath import density
from zeugma import EmbeddingTransformer

__author__ = "André Bittar"
__copyright__ = "Copyright 2020, André Bittar"
__credits__ = ["André Bittar"]
__license__ = "GPL"
__email__ = "andre.bittar@kcl.ac.uk"

nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner']) # remove the costly and unnecessary processing

SEED = 7

# TOOD specify these as command line arguments
BASE_DIR_T = 'T:/Andre Bittar/workspace/ehost-it/text-classification-suicide-risk/'
BASE_DIR_Z = 'Z:/Andre Bittar/Projects/eHOST-IT/'

CATEGORY_HEADER = 'category'
ENCODED_HEADINGS = ['sex', 'all_docs_30_days_to_admidate', 'age_admistart', 'Accommodation_Status', 'Substance_Abuse_Current', 'Substance_Abuse_Past', 'PSA_Employment', 'Marital_Status', 'Lives_With', 'Ethnicity', 'Employment', 'Disabled', 'Interpreter_Needed', 'Welfare_Benefits', 'Abstract_Thinking', 'Aggression', 'Agitation', 'Amphetamine', 'Anergia', 'Anhedonia', 'Apathy', 'Appetite', 'BFlat_Affect', 'Bradykinesia', 'Cannabis', 'Catatonic_Syndrome', 'Circumstantial_Speech', 'Cocaine', 'Coherence', 'Concentration', 'Delusion', 'Derailment_Of_Speech', 'Diabetes', 'Disturbed_Sleep', 'Echolalia', 'Echopraxia', 'Elation', 'Elevated_Mood', 'Emotion_Instability', 'Emotional_Withdrawal', 'Energy', 'FOI', 'FTD', 'Grandiosity', 'Guilt', 'Hallucination', 'Hallucination_OTG', 'Helpless', 'Hopeless', 'Hostility', 'Immobility', 'Insight', 'Insomnia', 'Irritability', 'Low_Mood', 'Mannerism', 'MDMA', 'Mood_Instability', 'Mutism', 'Negative_Symptom', 'Paranoia', 'Persecution', 'Perseverance', 'Poor_Motivation', 'Posturing', 'Poverty_Of_Speech', 'Poverty_Of_Thought', 'Pressured_Speech', 'Psychomotor', 'Rigid_Dementia', 'Rigidity', 'Social_Withdrawal', 'Stereotype', 'Stupor', 'Suicide', 'Tangential_Speech', 'Tearful', 'Thought_Block', 'Treatment_Resistant', 'Tremor', 'Weightloss', 'Worthless']
FORM_HEADINGS = ['sex', 'age_band', 'Accommodation_Status', 'Substance_Abuse_Current', 'Substance_Abuse_Past', 'PSA_Employment', 'Marital_Status', 'Lives_With', 'Ethnicity', 'Employment', 'Disabled', 'Interpreter_Needed', 'Welfare_Benefits']
GATE_HEADINGS = ['Abstract_Thinking', 'Aggression', 'Agitation', 'Amphetamine', 'Anergia', 'Anhedonia', 'Apathy', 'Appetite', 'BFlat_Affect', 'Bradykinesia', 'Cannabis', 'Catatonic_Syndrome', 'Circumstantial_Speech', 'Cocaine', 'Coherence', 'Concentration', 'Delusion', 'Derailment_Of_Speech', 'Diabetes', 'Disturbed_Sleep', 'Echolalia', 'Echopraxia', 'Elation', 'Elevated_Mood', 'Emotion_Instability', 'Emotional_Withdrawal', 'Energy', 'FOI', 'FTD', 'Grandiosity', 'Guilt', 'Hallucination', 'Hallucination_OTG', 'Helpless', 'Hopeless', 'Hostility', 'Immobility', 'Insight', 'Insomnia', 'Irritability', 'Low_Mood', 'Mannerism', 'MDMA', 'Mood_Instability', 'Mutism', 'Negative_Symptom', 'Paranoia', 'Persecution', 'Perseverance', 'Poor_Motivation', 'Posturing', 'Poverty_Of_Speech', 'Poverty_Of_Thought', 'Pressured_Speech', 'Psychomotor', 'Rigid_Dementia', 'Rigidity', 'Social_Withdrawal', 'Stereotype', 'Stupor', 'Suicide', 'Tangential_Speech', 'Tearful', 'Thought_Block', 'Treatment_Resistant', 'Tremor', 'Weightloss', 'Worthless']


def tokenize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ if token.lemma_ != '-PRON-'
                         else token.lower_ for token in doc])


tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english', 
                                  token_pattern='[A-Za-z][\w\-]+', norm='l2')


skb = SelectKBest(k=500)


def encode_data(df, headings=None):
    d = defaultdict(LabelEncoder)

    if headings is not None:
        print('-- Using custom headings:', headings)
        df[headings] = df[headings].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        return df[headings]

    df[ENCODED_HEADINGS] = df[ENCODED_HEADINGS].apply(lambda x: d[x.name].fit_transform(x.astype(str)))

    return df


def three_way_split(df, ready=False):
    """
    df: the data
    ready: True indicates that the data has been prepared for classification.
           That is, all unnecessary fields have been removed and the category
           labels have been added. False, otherwise.
    """
    print('-- Splitting data into train (60%), dev (20%), test (20%)', file=sys.stderr)
    np.random.seed(SEED)
    df_train, df_dev, df_test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    print("train={} dev={} test={}".format(len(df_train), len(df_dev), len(df_test)), file=sys.stderr)
    
    if not ready:
        t = df_train[['pk', CATEGORY_HEADER]].groupby(CATEGORY_HEADER).count()['pk']
        d = df_dev[['pk', CATEGORY_HEADER]].groupby(CATEGORY_HEADER).count()['pk']
        x = df_test[['pk', CATEGORY_HEADER]].groupby(CATEGORY_HEADER).count()['pk']
    else:
        t = len(df_train)
        d = len(df_dev)
        x = len(df_test)

    print("train=", t, file=sys.stderr)
    print("dev=", d, file=sys.stderr)
    print("test=", x, file=sys.stderr)
    
    return df_train, df_dev, df_test


def plot_roc_curve(model_name, period, feature_set, fpr, tpr, label, scale_data=False):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')
    plt.show()


def plot_precision_recall_vs_threshold(model_name, period, feature_set, precisions, recalls, thresholds, scale_data=False):
    """
    From: https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title(model_name + " Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.show()


def precision_recall_threshold(model_name, period, fout, p, r, thresholds, y_scores, y_test, t=0.5):
    """
    From: https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """

    def adjusted_classes(y_scores, t):
        """
        This function adjusts class predictions based on the prediction threshold (t).
        Will only work for binary classification problems.
        """
        return [1 if y >= t else 0 for y in y_scores]
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2, where='post')
    plt.fill_between(r, p, step='post', alpha=0.2, color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k', markersize=15)
    plt.show()


def create_plot():
    """
    This plots the evolution of F-score during tuning for each algorithm 
    and feature set.
    """
    path = BASE_DIR_T + 'output/'

    model_names = ['SVM', 'MLP', 'RF', 'NB']
    feature_sets = ['STRUCT', 'GATE', 'TFIDF', 'STRUCT+GATE', 'STRUCT+TFIDF', 'GATE+TFIDF', 'STRUCT+GATE+TFIDF']

    df = pd.DataFrame(columns=model_names, index=feature_sets)

    for file in [f for f in os.listdir(path) if f.endswith('.pickle')]:
        feature_set = None
        model_name = re.search('30d_([^\_]+)_', file)
        if model_name is not None:
            model_name = model_name.group(1)
        else:
            raise ValueError('-- Invalid model name:', model_name)

        if 'STRUCT+GATE+TFIDF' in file:
            feature_set = 'STRUCT+GATE+TFIDF'
        elif 'STRUCT+TFIDF' in file:
            feature_set = 'STRUCT+TFIDF'
        elif 'STRUCT+GATE' in file:
            feature_set = 'STRUCT+GATE'
        elif 'GATE+TFIDF' in file:
            feature_set = 'GATE+TFIDF'
        elif 'STRUCT' in file:
            feature_set = 'STRUCT'
        elif 'GATE' in file:
            feature_set = 'GATE'
        elif 'TFIDF' in file:
            feature_set = 'TFIDF'

        pin = os.path.join(path, file)
        res = pickle.load(open(pin, 'rb'))
        results = res['results']
        f1 = results['mean_test_f1_score'].max()
        df[model_name][feature_set] = np.float64(f1)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('F-score')
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df.index, rotation=0, ha='right')
    p = df.plot(use_index=True, ax=ax, ylim=(0, 1), fontsize=12, figsize=(8, 5), 
            colormap='viridis', rot=45, linewidth=2) # bone
    p.set_axis_bgcolor('#F8F8F8')
    plt.savefig(path + '/model_performance_vs_feature_set.png')
    plt.show()


def print_tfidf(tfidf_matrix, doc_number, n):
    feature_names = tfidfvectorizer.get_feature_names()
    feature_index = tfidf_matrix[doc_number,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc_number, x] for x in feature_index])
    tmp = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:n]
    for w, s in [(feature_names[i], s) for (i, s) in tmp]:
        print(w, s)


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    From: https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html

    For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        # this returns the column requested
        #print("Selecting", self.key)
        if self.key == 'text':
            return data_dict.loc[:,self.key]
        else:
            return data_dict.loc[:,[self.key]].astype(float)


def grid_search_wrapper(model_name, fout, classifier_instance, parameter_grid, scorer_list, X_train_sample, X_test_sample, y_train_sample, y_test_sample, refit_score='precision_score'):
    """
    Fit a GridSearchCV classifier using refit_score for optimization.
    Print classifier performance metrics.
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(classifier_instance, parameter_grid, scoring=scorer_list, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1, verbose=10)
    
    grid_search.fit(X_train_sample, y_train_sample)

    # make the predictions
    y_pred = grid_search.predict(X_test_sample)

    report = classification_report(y_test_sample, y_pred)
    print(report, file=fout)
    print(report)

    print('Best params for {}'.format(refit_score), file=fout)
    print(grid_search.best_params_, file=fout)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data
    df_confusion_matrix = pd.DataFrame(confusion_matrix(y_test_sample, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])

    print('\nConfusion matrix of ' + model_name + ' optimized for {} on the test data:'.format(refit_score), file=fout)
    print(df_confusion_matrix, file=fout)

    print('\nConfusion matrix of ' + model_name + ' optimized for {} on the test data:'.format(refit_score))
    print(df_confusion_matrix)

    # output k-best features
    # TODO this may only work for feature sets with TFIDF
    if hasattr(classifier_instance, 'coef_'):
        feature_names = tfidfvectorizer.get_feature_names()
        feature_names = [feature_names[i] for i
                             in skb.get_support(indices=True)]
        feature_names = np.asarray(feature_names)
        
        def trim(s):
            """Trim string to fit on terminal (assuming 80-column display)"""
            return s if len(s) <= 80 else s[:77] + "..."
        
        print("dimensionality: %d" % classifier_instance.coef_.shape[1])
        print("density: %f" % density(classifier_instance.coef_))
        
        print("top 10 keywords per class:")
        target_names = [0, 1]
        for i, label in enumerate(target_names):
            top10 = np.argsort(classifier_instance.coef_[i])[-10:]
            print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    return grid_search


def process(config, period, X_train, X_test, y_train, y_test, feature_set='STRUCT+GATE', scale_data=False):
    model_name = config[0]
    clf = config[1]
    param_grid = config[2]

    # Parameters must be set with the format: estimator__paramname
    new_param_grid = {}
    for key in param_grid:
        new_key = 'clf__' + key
        new_param_grid[new_key] = param_grid[key]
    
    param_grid = new_param_grid

    pout = BASE_DIR_T + 'output/' + model_name + '_' + str(period) + 'd_' + feature_set + '_unscaled_output.txt'
    if scale_data:
        pout = BASE_DIR_T + 'output/' + model_name + '_' + str(period) + 'd_' + feature_set + '_scaled_output.txt'
    
    if not os.path.exists(BASE_DIR_T + 'output/'):
        os.makedirs(BASE_DIR_T + 'output/')
    
    fout = open(pout, 'w')

    print('-' * len(model_name), file=fout)
    print('-' * len(model_name))
    print(model_name, file=fout)
    print(model_name)
    print('-' * len(model_name), file=fout)
    print('-' * len(model_name))

    # Not testing yet
    if feature_set in ['TFIDF_OLD', 'STRUCT+GATE+TFIDF_OLD']:
        pipeline = Pipeline([('feats', 
                       FeatureUnion(
                               transformer_list=[('text', Pipeline([
                                               ('selector', ItemSelector(key='text')), 
                                               ('tfidf', tfidfvectorizer)])
                                                )]
                                    )),
                    ('clf', clf)])
    elif feature_set in ['GATE', 'STRUCT', 'STRUCT+GATE', 'STRUCT+GATE+TFIDF', 'TFIDF', 'STRUCT+TFIDF', 'GATE+TFIDF']:
        if model_name == 'RF':
            if 'TFIDF' in feature_set:
                print('-- Using SelectKBest:', skb)
                print('-- Using SelectKBest:', skb, file=fout)
                pipeline = Pipeline([('kbest', skb), ('clf', clf)])
            else:
                print('-- Using SelectKBest: k="all"')
                print('-- Using SelectKBest: k="all"', file=fout)
                pipeline = Pipeline([('kbest', SelectKBest(k='all')), ('clf', clf)])
        else:
            pipeline = Pipeline([('clf', clf)])
    elif feature_set in ['GLOVE', 'WORD2VEC']:
        vec = EmbeddingTransformer(feature_set.lower())
        pipeline = Pipeline([('feats', 
                       FeatureUnion(
                               transformer_list=[('text', Pipeline([
                                               ('selector', ItemSelector(key='text')), 
                                               ('vectorizer', vec)])
                                                )]
                                    )),
                    ('clf', clf)])
    else:
        raise ValueError('-- Error: invalid feature set', feature_set)
    
    if scale_data:
        print('-- Using scaled data')
        print('-- Using scaled data', file=fout)
    
    print('Period is', period, 'days prior to admission')
    print('Using ' + feature_set + ' feature set')
    print('Grid search for', model_name)
    print('y_train class distribution')
    # deal with the numpy array format (i.e. not a DataFrame)
    if isinstance(y_train, np.ndarray):
        print(y_train)
    else:
        print(y_train.value_counts(normalize=True))
    print('y_test class distribution')
    if isinstance(y_test, np.ndarray):
        print(y_test)
    else:
        print(y_test.value_counts(normalize=True))
    
    print('Period is', period, 'days prior to admission', file=fout)
    print('Using ' + feature_set +  'feature set', file=fout)
    print('Grid search for', model_name, file=fout)
    if isinstance(y_train, np.ndarray):
        print(y_train, file=fout)
    else:
        print(y_train.value_counts(normalize=True), file=fout)
    print('y_test class distribution', file=fout)
    if isinstance(y_test, np.ndarray):
        print(y_test, file=fout)
    else:
        print(y_test.value_counts(normalize=True), file=fout)
    
    scorers = {
            'f1_score': make_scorer(f1_score),
            'precision_score': make_scorer(precision_score),
            'recall_score': make_scorer(recall_score),
            'accuracy_score': make_scorer(accuracy_score)
    }
    
    # Choose metric to optimise
    metric = 'f1_score'
    metric = 'recall_score'
    #metric = 'precision_score'

    print('Doing grid search for', model_name + ' ' + metric + '...', file=fout)
    print('Parameters:', param_grid, file=fout)
    
    print('Doing grid search for', model_name + ' ' + metric + '...')
    print('Parameters:', param_grid)
    
    t0 = time.time()
    
    try:
        grid_search_clf = grid_search_wrapper(model_name, fout, pipeline, param_grid, scorers, X_train, X_test, y_train, y_test, refit_score=metric)
    except Exception as e:
        print('-- Error: could not run grid search for', model_name, period, feature_set)
        print('-- Error: could not run grid search for', model_name, period, feature_set, file=fout)
        print(e)
        print(e, file=fout)
        return {}

    results = pd.DataFrame(grid_search_clf.cv_results_)
    results = results.sort_values(by='mean_test_recall_score', ascending=False)
    headings = ['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score']
    headings.extend(['param_' + param for param in param_grid.keys()])
    results[headings].round(3).head()

    if hasattr(grid_search_clf, 'decision_function'):
        # for classifiers with decision_function, this achieves similar results
        y_scores = grid_search_clf.decision_function(X_test)
    else:
        y_scores = grid_search_clf.predict_proba(X_test)[:, 1]

    p, r, thresholds = precision_recall_curve(y_test, y_scores)

    precision_recall_threshold(model_name, period, fout, p, r, thresholds, y_scores, y_test, t=0.30)
    
    # Use the same p, r, thresholds that were previously calculated
    plot_precision_recall_vs_threshold(model_name, period, feature_set, p, r, thresholds, scale_data=scale_data)

    fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)

    print('AUC ROC:', auc(fpr, tpr), file=fout) # AUC of ROC
    print('AUC ROC:', auc(fpr, tpr)) # AUC of ROC
    
    plot_roc_curve(model_name, period, feature_set, fpr, tpr, 'recall_optimized', scale_data=scale_data)

    t1 = time.time() - t0
    
    print('-- Total time: done in', t1, 'seconds.', file=fout)
    print('-- Total time: Done in', t1, 'seconds.')

    fout.close()

    return {'model': model_name,
            'results': results,
            'p': p,
            'r': r,
            'thresholds': thresholds,
            'y_test': y_test,
            'y_scores': y_scores,
            't1': t1,
            'clf': grid_search_clf
            }


def run_gridsearchcv(config, periods, feature_set):
    # Run for both structured and full feature sets
    for feature_set in feature_set:
        for period in periods:
            X_train, X_test, y_train, y_test = prepare_train_data(feature_set, scale_data=True)
            for c in config:
                model_name = c[0]
                if model_name in ['NB']:
                    X_train, X_test, y_train, y_test = prepare_train_data(feature_set, scale_data=False)
                output_path = BASE_DIR_T + 'output/grid_search_pipeline_results_' + str(period) + 'd_' + model_name + '_' + feature_set + '_scaled.pickle'
                r = process(c, period, X_train, X_test, y_train, y_test, feature_set=feature_set, scale_data=True)
                pickle.dump(r, open(output_path, 'wb'))
                plt.close('all')


def prepare_train_data(feature_set, scale_data=False):
    #df = pd.read_pickle(BASE_DIR_Z + 'data/cc_30d_struct_GATE+TFIDF_unencoded_train.pickle')
    df = pd.read_pickle(BASE_DIR_Z + 'data/final_run/cc_30d_struct_gate_no_empty_text_unencoded_train.pickle')
    #df = pd.read_pickle(BASE_DIR_Z + '/data/cc_30d_struct_gate_no_empty_text_unencoded_tokenized_train.pickle')
    
    # Remove redundant 0 count features
    for c in ['Affect_Instability',  'Catalepsy', 'Waxy_Flexibility']:
        if c in df.columns:
            df = df.drop(c, axis=1)
    
    df = encode_data(df)
    df['text'] = df.text.fillna(value='')
    
    print('-- Using ' + feature_set + ' feature set')
    
    train, dev, test = three_way_split(df, ready=True)
    df = train.append(dev)
    
    targets = df[CATEGORY_HEADER]
    df = df.drop(CATEGORY_HEADER, axis=1)
    
    if feature_set == 'STRUCT':
        df = df.drop(GATE_HEADINGS, axis=1)
        df = df.drop('text', axis=1)
        structured_headings = list(set(ENCODED_HEADINGS) - set(GATE_HEADINGS))
        df = encode_data(df, headings=structured_headings)
        
        X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler()
            X_train[structured_headings] = scaler.fit_transform(X_train[structured_headings])
            X_test[structured_headings] = scaler.fit_transform(X_test[structured_headings])

    elif feature_set == 'STRUCT+TFIDF':
        df = df.drop(GATE_HEADINGS, axis=1)
        df_2 = pd.DataFrame(df['text'])
        structured_headings = list(set(ENCODED_HEADINGS) - set(GATE_HEADINGS))
        df = encode_data(df, headings=structured_headings)
        
        f = df[structured_headings].values
        df_t = tfidfvectorizer.fit_transform(df_2['text'])
        
        targets = targets.values.T
        x_all = hstack([f, df_t])
        
        X_train, X_test, y_train, y_test = train_test_split(x_all, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler(with_mean=False) # with_mean=False to scale text features
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

    elif feature_set == 'GATE':
        df = df.drop('text', axis=1)
        df = df.drop('age_admistart', axis=1)
        df = df.drop('all_docs_30_days_to_admidate', axis=1)
        for item in FORM_HEADINGS:
            if item in df.columns:
                df = df.drop(item, axis=1)
        df = encode_data(df, headings=df.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler()
            X_train[GATE_HEADINGS] = scaler.fit_transform(X_train[GATE_HEADINGS])
            X_test[GATE_HEADINGS] = scaler.fit_transform(X_test[GATE_HEADINGS])

    elif feature_set == 'GATE+TFIDF':
        df_2 = pd.DataFrame(df['text'])
        for item in FORM_HEADINGS:
            if item in df.columns:
                df = df.drop(item, axis=1)
        df = encode_data(df, headings=df.columns)
        
        f = df[GATE_HEADINGS].values
        df_t = tfidfvectorizer.fit_transform(df_2['text'])
        targets = targets.values.T
        x_all = hstack([f, df_t])
        
        X_train, X_test, y_train, y_test = train_test_split(x_all, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler(with_mean=False) # with_mean=False to scale text features
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

    elif feature_set == 'TFIDF_OLD':
        df = pd.DataFrame(df['text'])
                
        X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets, shuffle=True, random_state=SEED)
                
    elif feature_set == 'STRUCT+GATE':
        df = df.drop('text', axis=1)
        df = encode_data(df)

        X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler()
            X_train[ENCODED_HEADINGS] = scaler.fit_transform(X_train[ENCODED_HEADINGS])
            X_test[ENCODED_HEADINGS] = scaler.fit_transform(X_test[ENCODED_HEADINGS])
        
    elif feature_set == 'STRUCT+GATE+TFIDF_OLD':
        df = encode_data(df)
                
        X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler()
            X_train[ENCODED_HEADINGS] = scaler.fit_transform(X_train[ENCODED_HEADINGS])
            X_test[ENCODED_HEADINGS] = scaler.fit_transform(X_test[ENCODED_HEADINGS])

    elif feature_set == 'TFIDF':
        df = pd.DataFrame(df['text'])
        df_t = tfidfvectorizer.fit_transform(df['text'])
        
        targets = targets.values.T        
        
        X_train, X_test, y_train, y_test = train_test_split(df_t, targets, stratify=targets, shuffle=True, random_state=SEED)
        
    elif feature_set == 'STRUCT+GATE+TFIDF':
        f = df[ENCODED_HEADINGS].values
        df_t = tfidfvectorizer.fit_transform(df['text'])

        x_all = hstack([f, df_t])

        targets = targets.values.T
        
        X_train, X_test, y_train, y_test = train_test_split(x_all, targets, stratify=targets, shuffle=True, random_state=SEED)
    
    elif feature_set == 'GLOVE':
        df = pd.DataFrame(df['text'])
        vec = EmbeddingTransformer(feature_set.lower())
        df_t = vec.fit_transform(df['text'])
        
        targets = targets.values.T
        
        X_train, X_test, y_train, y_test = train_test_split(df_t, targets, stratify=targets, shuffle=True, random_state=SEED)
        
    else:
        raise ValueError('-- Error: invalid feature set', feature_set)
    
    return X_train, X_test, y_train, y_test


def prepare_test_data(period, feature_set, scale_data=False):
    print('-- Preparing test data')
    print('-- Using ' + feature_set + ' feature set')
    df = pd.read_pickle(BASE_DIR_Z + 'data/final_run/cc_30d_struct_gate_no_empty_text_unencoded_train.pickle')
    
    # Remove redundant 0 count features
    for c in ['Affect_Instability',  'Catalepsy', 'Waxy_Flexibility']:
        if c in df.columns:
            df = df.drop(c, axis=1)
    
    df = encode_data(df)
    df['text'] = df.text.fillna(value='')
    
    targets = df[CATEGORY_HEADER]
    df = df.drop(CATEGORY_HEADER, axis=1)
    
    # Set correct columns to encode
    if 'all_docs_30_days_to_admidate' not in ENCODED_HEADINGS:
        ENCODED_HEADINGS.append('all_docs_30_days_to_admidate')
    if 'all_docs_90_days_to_admidate' in ENCODED_HEADINGS:
        ENCODED_HEADINGS.remove('all_docs_90_days_to_admidate')
    if 'all_docs_180_days_to_admidate' in ENCODED_HEADINGS:
        ENCODED_HEADINGS.remove('all_docs_180_days_to_admidate')

    if feature_set == 'STRUCT':
        structured_headings = list(set(ENCODED_HEADINGS) - set(GATE_HEADINGS))
        df = df.drop(GATE_HEADINGS, axis=1) # test just structured fields
        df = df.drop('text', axis=1)
        df = encode_data(df, headings=structured_headings)
        
        X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler()
            X_train[structured_headings] = scaler.fit_transform(X_train[structured_headings])
            X_test[structured_headings] = scaler.fit_transform(X_test[structured_headings])

    elif feature_set == 'STRUCT+TFIDF':
        df = df.drop(GATE_HEADINGS, axis=1)
        df_2 = pd.DataFrame(df['text'])
        structured_headings = list(set(ENCODED_HEADINGS) - set(GATE_HEADINGS))
        df = encode_data(df, headings=structured_headings)
        
        f = df[structured_headings].values 
        df_t = tfidfvectorizer.fit_transform(df_2['text'])
        
        print('-- Vocab size:', len(tfidfvectorizer.vocabulary_))
        
        targets = targets.values.T
        x_all = hstack([f, df_t])
        
        X_train, X_test, y_train, y_test = train_test_split(x_all, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler(with_mean=False) # with_mean=False to scale text features
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

    elif feature_set == 'GATE':
        df = df.drop('text', axis=1)
        df = df.drop('age_admistart', axis=1)
        df = df.drop('all_docs_30_days_to_admidate', axis=1)
        for item in FORM_HEADINGS:
            if item in df.columns:
                df = df.drop(item, axis=1)
        df = encode_data(df, headings=df.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler()
            X_train[GATE_HEADINGS] = scaler.fit_transform(X_train[GATE_HEADINGS])
            X_test[GATE_HEADINGS] = scaler.fit_transform(X_test[GATE_HEADINGS])

    elif feature_set == 'GATE+TFIDF':
        df_2 = pd.DataFrame(df['text'])
        for item in FORM_HEADINGS:
            if item in df.columns:
                df = df.drop(item, axis=1)
        df = encode_data(df, headings=df.columns)
        
        f = df[GATE_HEADINGS].values 
        df_t = tfidfvectorizer.fit_transform(df_2['text'])
        
        print('-- Vocab size:', len(tfidfvectorizer.vocabulary_))
        
        targets = targets.values.T
        x_all = hstack([f, df_t])
        
        X_train, X_test, y_train, y_test = train_test_split(x_all, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler(with_mean=False) # with_mean=False to scale text features
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

    elif feature_set == 'STRUCT+GATE':
        df = df.drop('text', axis=1)
        df = encode_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets, shuffle=True, random_state=SEED)
        
        if scale_data:
            scaler = StandardScaler()
            X_train[ENCODED_HEADINGS] = scaler.fit_transform(X_train[ENCODED_HEADINGS])
            X_test[ENCODED_HEADINGS] = scaler.fit_transform(X_test[ENCODED_HEADINGS])

    elif feature_set == 'TFIDF':
        df = pd.DataFrame(df['text'])
        df_t = tfidfvectorizer.fit_transform(df['text'])
        
        print('-- Vocab size:', len(tfidfvectorizer.vocabulary_))
        
        targets = targets.values.T
                
        X_train, X_test, y_train, y_test = train_test_split(df_t, targets, stratify=targets, shuffle=True, random_state=SEED)


    elif feature_set == 'STRUCT+GATE+TFIDF':
        f = df[ENCODED_HEADINGS].values
        df_t = tfidfvectorizer.fit_transform(df['text'])
        
        print('-- Vocab size:', len(tfidfvectorizer.vocabulary_))
        
        x_all = hstack([f, df_t])
        targets = targets.values.T
        
        X_train, X_test, y_train, y_test = train_test_split(x_all, targets, stratify=targets, shuffle=True, random_state=SEED)

    elif feature_set == 'GLOVE':
        df = pd.DataFrame(df['text'])
        vec = EmbeddingTransformer(feature_set.lower())
        df_t = vec.fit_transform(df['text'])
        
        print('-- Vocab size:', len(vec.model.vocab))
        
        targets = targets.values.T
                
        X_train, X_test, y_train, y_test = train_test_split(df_t, targets, stratify=targets, shuffle=True, random_state=SEED)        

    else:
        raise ValueError('-- Error: invalid feature set', feature_set)

    return X_train, y_train, X_test, y_test


def print_text_feature_importances(X, targets, fout):
    """
    From: https://buhrmann.github.io/tfidf-analysis.html
    """
    def top_tfidf_feats(row, features, top_n=25):
        ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
        topn_ids = np.argsort(row)[::-1][:top_n].tolist()[0]
        
        top_feats = []
        for i in topn_ids:
            if i < len(features):
                f = features[i]
                r = row.tolist()[0][i]
                t = (f, r)
                top_feats.append(t)
        
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        return df

    def top_feats_in_doc(Xtr, features, row_id, top_n=25):
        ''' Top tfidf features in specific document (matrix row) '''
        row = np.squeeze(Xtr[row_id].toarray())
        return top_tfidf_feats(row, features, top_n)

    def top_mean_feats(Xtr, targets, cat, features, min_tfidf=0.1, top_n=25):
        ''' Return the top n features that on average are most important amongst documents in rows
            indentified by indices in grp_ids. '''
        print('-- Text feature importances for category:', cat)
        print('-- Text feature importances for category:', cat, file=fout)
        
        group_ids = np.where(targets == cat)
        D = Xtr[group_ids]
        D[D < min_tfidf] = 0
        tfidf_means = np.mean(D, axis=0)
        
        return top_tfidf_feats(tfidf_means, features, top_n)

    features = tfidfvectorizer.get_feature_names() 
    df_1 = top_mean_feats(X, targets, 1, features)
    df_0 = top_mean_feats(X, targets, 0, features)
    
    return df_1, df_0


def plot_coefficients(classifier, model_name, feature_set, feature_names, top_features=20):
    """
    From: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
    """
    coef = classifier.coef_.ravel()
        
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        
    # create plot
    plt.figure(figsize=(15, 7))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.savefig(BASE_DIR_T + 'output/test_pipeline_results_30d_' + model_name + '_' + feature_set + '_scaled_FI.png', bbox_inches='tight')
    plt.show()


def print_feature_importances(rfc, X, period, feature_set, fout):
    """
    Code adapted from: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html (and elsewhere)
    """
    print('------------------')
    print('Feature importance')
    print('------------------')
    
    print('------------------', file=fout)
    print('Feature importance', file=fout)
    print('------------------', file=fout)
    
    if isinstance(X, pd.DataFrame):
        features = sorted(zip(list(range(0, len(X.columns))), X.columns, rfc.feature_importances_), key=lambda x: x[2], reverse=True)
        for n, f, v in features:
            print('{:03} {}\t{}'.format(n, f, v))
            print('{:03} {}\t{}'.format(n, f, v), file=fout)
    else:
        n_feats = 20
        fnames = tfidfvectorizer.get_feature_names()
        features = sorted(zip(list(range(0, n_feats)), X, rfc.feature_importances_), key=lambda x: x[2], reverse=True)
        for n, f, v in features:
            print('n={:03} f={}\tv={}'.format(n, f, v))
            print('{:03} {}\t{}'.format(n, f, v), file=fout)
    
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
    indices = np.argsort(rfc.feature_importances_)[::-1]
    feature_names = X.columns[indices]
    
    # Plot the feature importances of the forest
    plt.figure(figsize=(20, 10))
    plt.title("Feature importances")
    
    if isinstance(X, pd.DataFrame):
        plt.bar(range(X.shape[1]), rfc.feature_importances_[indices],
       color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), feature_names, rotation=45, ha='right')
        plt.xlim([-1, X.shape[1]])
    else:
        n = 0
        plt.bar(range(n), rfc.feature_importances_[indices],
       color="r", yerr=std[indices], align="center")
        plt.xticks(range(n), feature_names)
        plt.xlim([-1, n])
        
    plt.savefig(BASE_DIR_T + 'output/test_pipeline_results_' + str(period) + 'd_RF_' + feature_set + '_scaled_FI.png')
    plt.show()


def create_term_pos_listing(global_top_n_df, ng_size, threshold=0.2):
    """
    Create a DataFrame containing the top ngrams above a given treshold.
    TODO implement ngram size
    """
    tmp = global_top_n_df.loc[global_top_n_df.tfidf > threshold]
    
    if ng_size == 2:
        tmp = pd.DataFrame(tmp.feature.values.tolist(), columns=['1', '2'])
    if ng_size == 3:
        tmp = pd.DataFrame(global_top_n_df.feature.values.tolist(), columns=['1', '2', '3'])
    
    tmp['1'] = tmp['1'].apply(lambda x: x.split('_'))
    tmp['2'] = tmp['2'].apply(lambda x: x.split('_'))
    
    if ng_size == 3:
        tmp['3'] = tmp['3'].apply(lambda x: x.split('_'))
    
    tmp['1'] = tmp[tmp['1'].apply(lambda x: len(x) == 2)]['1']
    tmp['2'] = tmp[tmp['2'].apply(lambda x: len(x) == 2)]['2']

    if ng_size == 3:
        tmp['3'] = tmp[tmp['3'].apply(lambda x: len(x) == 2)]['3']

    tmp.dropna(inplace=True)
    
    tmp1 = pd.DataFrame(tmp['1'].values.tolist(), columns=['w1', 'p1'])
    tmp2 = pd.DataFrame(tmp['2'].values.tolist(), columns=['w2', 'p2'])

    if ng_size == 2:
        tmp = pd.concat([tmp1, tmp2], axis=1)      
        tmp = tmp[['w1', 'w2', 'p1', 'p2']]

    if ng_size == 3:
        tmp3 = pd.DataFrame(tmp['3'].values.tolist(), columns=['w3', 'p3'])
        tmp = pd.concat([pd.concat([tmp1, tmp2], axis=1), tmp3], axis=1)
        tmp = tmp[['w1', 'w2', 'w3', 'p1', 'p2', 'p3']]   
        
    return tmp


def get_top_tfidf_features(df, tfidfvectorizer=None, token_pattern='[A-Za-z][\w\-]+', ngram_range=(1, 1), top_n=20, lowercase=False):
    """
    Load training data and list top n features for a given document (row_id).
    Default token_pattern is for word-only.
    For word_POS use token_pattern='[A-Za-z][\w\-]+\_[A-Z\$]+'
    """
    
    if df is None:
        df = pd.read_pickle(BASE_DIR_Z + '/data/final_run/cc_30d_struct_gate_no_empty_text_unencoded_train.pickle')
    else:
        # Set up pre-processed DataFrame
        print('-- Setting lowercase=False to extract word_POS pairs')
        lowercase = False
        df.rename(columns={'tokens': 'text'}, inplace=True)
        df['text'] = df.text.apply(lambda x: ' '.join(x))

    # Remove redundant 0 count features - why do I bother with structured features?
    for c in ['Affect_Instability',  'Catalepsy', 'Waxy_Flexibility']:
        if c in df.columns:
            df = df.drop(c, axis=1)

    #df = encode_data(df)
    df['text'] = df.text.fillna(value='')

    if tfidfvectorizer is None:
        tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words=None, 
                                          token_pattern=token_pattern, 
                                          norm='l2', ngram_range=ngram_range,
                                          lowercase=lowercase)
    
    print('-- Fitting vectorizer...')
    vec = tfidfvectorizer.fit_transform(df['text'])
    features = tfidfvectorizer.get_feature_names()
    
    print('-- Extracted', len(tfidfvectorizer.vocabulary_), 'features')
    
    # Code taken from https://buhrmann.github.io/tfidf-analysis.html
    def top_tfidf_feats(row, features, top_n=25, threshold=0):
        ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
        if top_n == 'all':
            topn_ids = np.argsort(row)[::-1]
        else:
            topn_ids = np.argsort(row)[::-1][:top_n]
        top_feats = [(features[i], row[i]) for i in topn_ids if row[i] > threshold] # ignore values below a certain threshold
        if top_feats == []:
            return pd.DataFrame(columns=['feature', 'tfidf'])
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        return df
    
    def top_feats_in_doc(Xtr, features, row_id, top_n=25):
        ''' Top tfidf features in specific document (matrix row) '''
        row = np.squeeze(Xtr[row_id].toarray())
        return top_tfidf_feats(row, features, top_n)

    global_top_n_df = pd.DataFrame(columns = ['feature', 'tfidf'])
    
    print('-- Collecting', top_n, 'top features for each of', vec.shape[0], 'documents...', end='', file=sys.stderr)
    
    for i in range(vec.shape[0]):
        top_n_df = top_feats_in_doc(vec, features, i, top_n=top_n)
        global_top_n_df = global_top_n_df.append(top_n_df)
    
    print('Done.', file=sys.stderr)
    
    global_top_n_df = global_top_n_df.reset_index()
    global_top_n_df = global_top_n_df.drop('index', axis=1)
    global_top_n_df.drop_duplicates(inplace=True)
    global_top_n_df = global_top_n_df.sort_values('tfidf', ascending=False).drop_duplicates('feature').sort_index()
    global_top_n_df = global_top_n_df.reset_index()
    global_top_n_df = global_top_n_df.drop('index', axis=1)

    print('-- Collected', len(global_top_n_df), 'total features')
    
    return global_top_n_df


def run_test():
    # list of all tuned configrations for testing
    configs = [('SVM', LinearSVC(random_state=SEED, C=0.01, class_weight='balanced', dual=False, loss='squared_hinge'), 'STRUCT', [30]),
               ('SVM', LinearSVC(random_state=SEED, C=0.01, class_weight='balanced', dual=False, loss='squared_hinge'), 'GATE', [30]),
               ('SVM', LinearSVC(random_state=SEED, C=1, class_weight='balanced', dual=False, loss='squared_hinge'), 'TFIDF', [30]),
               ('SVM', LinearSVC(random_state=SEED, C=0.01, class_weight=None, dual=False, loss='squared_hinge'), 'GATE+TFIDF', [30]),
               ('SVM', LinearSVC(random_state=SEED, C=1, class_weight='balanced', dual=False, loss='squared_hinge'), 'STRUCT+GATE', [30]),
               ('SVM', LinearSVC(random_state=SEED, C=1, class_weight='balanced', dual=False, loss='squared_hinge'), 'STRUCT+GATE+TFIDF', [30]),
               ('SVM', LinearSVC(random_state=SEED, C=0.01, class_weight=None, dual=False, loss='squared_hinge'), 'STRUCT+TFIDF', [30])
               ]

    for config in configs:
        feature_set = config[2]
        periods = config[3]
        
        for period in periods:
            t0 = time.time()
            X_train, y_train, X_test, y_test = prepare_test_data(period, feature_set, scale_data=True)
            
            model_name = config[0]
            
            pout = BASE_DIR_T + '/output/' + model_name + '_test_results_' + str(period) + 'd_' + feature_set + '_scaled.txt'
            
            if model_name in ['NB']:
                X_train, y_train, X_test, y_test = prepare_test_data(period, feature_set, scale_data=False)
                pout = BASE_DIR_T + '/output/' + model_name + '_test_results_' + str(period) + 'd_' + feature_set + '_unscaled.txt'
            
            fout = open(pout, 'w')
            
            print('-' * len(model_name))
            print(model_name)
            print('-' * len(model_name))
            
            print('-' * len(model_name), file=fout)
            print(model_name, file=fout)
            print('-' * len(model_name), file=fout)

            print('-- Config:', config)
            print('-- Config:', config, file=fout)

            if 'TFIDF' in feature_set:
                print('-- Used vectorizer:', tfidfvectorizer)
                print('-- Used vectorizer:', tfidfvectorizer, file=fout)

                if model_name in ['RF']:
                    print('-- Using SelectKBest:', skb)
                    print('-- Using SelectKBest:', skb, file=fout)

                X_train = skb.fit_transform(X_train, y_train)
                X_test = skb.transform(X_test)
            
            clf = config[1]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            print(classification_report(y_test, y_pred))
            print(classification_report(y_test, y_pred), file=fout)

            # Save these for McNemar's test
            # If TFIDF then y_test is an ndarray, else a Series
            y_test_s = None
            if isinstance(y_test, np.ndarray):
                y_test_s = pd.Series(y_test)
            else:
                y_test_s = y_test.reset_index(drop=True)
            y_pred_s = pd.Series(data=y_pred)
            df_y_comp = pd.DataFrame()
            df_y_comp['gold_' + feature_set] = y_test_s
            df_y_comp['pred_' + feature_set] = y_pred_s

            df_y_comp.to_pickle(BASE_DIR_T + '/output/' + model_name + '_gold_versus_pred_' + str(period) + 'd_' + feature_set + '.pickle')
            
            if model_name == 'RF':
                try:
                    if 'TFIDF' not in feature_set:
                        # This isn't yet working with TFIDF features
                        print_feature_importances(clf, X_train, period, feature_set, fout)
                except Exception as e:
                    print('-- Error: cannot calculate feature importances.')
                    print('-- Error: cannot calculate feature importances.', file=fout)
                    print(e)
                    print(e, file=fout)
                try:
                    df_1, df_0 = print_text_feature_importances(X_train, y_train, fout)
                    print(df_1)
                    print('--')
                    print(df_0)
                    print(df_1, file=fout)
                    print('--', file=fout)
                    print(df_0, file=fout)
                except Exception as e:
                    print('-- Error: cannot output text feature importances.')
                    print('-- Error: cannot output text feature importances.', file=fout)
                    print(e)
                    print(e, file=fout)
            # print the most important features, as stored in coef_
            if hasattr(clf, 'coef_'):
                top_n = 20
                if isinstance(X_train, pd.DataFrame):
                    feature_names = X_train.columns.tolist()
                else:
                    feature_names = []
                if feature_set == 'TFIDF':
                    feature_names = tfidfvectorizer.get_feature_names()
                elif feature_set == 'STRUCT+TFIDF':
                    feature_names = sorted(set(ENCODED_HEADINGS) - set(GATE_HEADINGS))
                    feature_names.extend(tfidfvectorizer.get_feature_names())
                elif feature_set == 'GATE+TFIDF':
                    feature_names = sorted(GATE_HEADINGS)
                    feature_names.extend(tfidfvectorizer.get_feature_names())
                elif feature_set == 'STRUCT+GATE+TFIDF':
                    feature_names = sorted(ENCODED_HEADINGS)
                    feature_names.extend(GATE_HEADINGS)
                    feature_names.extend(tfidfvectorizer.get_feature_names())
                else:
                    top_n = len(feature_names)
                args = {'feature_names': feature_names, 'clf': clf, 'feature_set': feature_set, 'model_name': model_name}
                pickle.dump(args, open(BASE_DIR_T + '/output/SVM_clf_30d_' + feature_set + '_args_for_coeff_mapping.pickle', 'wb'))
                plot_coefficients(clf, model_name, feature_set, feature_names, top_features=top_n)
            
            t1 = time.time() - t0

            print('-- Total time: done in', t1, 'seconds.', file=fout)
            print('-- Total time: Done in', t1, 'seconds.')
    
            print('-- Wrote file:', pout)
            print('-- Wrote file:', pout, file=fout)
            fout.close()


if __name__ == '__main__':
    """
    eHOST-IT hospital admission classifier - tuning and training script.
    
    Run grid search on a specified configuration.
    A configuration is a tuple of the form:
        ('model_name', 'model_object', 'parameters')
    The parameters are those to be tuned, along with the corresponding values.
    """
    parser = argparse.ArgumentParser(description='Run eHOST-IT text classification for suicide risk assessment (MedInfo2019)')
    parser.add_argument('-r', '--run_type', metavar='run_type', type=str, nargs=1, help='specify the type of run, either train or test.', choices=['tune', 'test'], required=True)
    
    args = parser.parse_args()
    
    if args.run_type[0] == 'tune':
        print('-- Running parameter tuning (grid search)...')
        # Define configuration for parameter tuning
        config1 = [('SVM', LinearSVC(random_state=SEED), {'C': [0.01, 1, 10, 1000],
                    'dual': [False], 'class_weight': ['balanced', None],
                    'loss': ['squared_hinge']})
                ]
        # The period is the number of days prior to a suicide attempt (30 in this case, but were we envisaging 90 and 180)
        periods = [30]
        # Declare the feature sets to be tuned on
        #feature_set = ['GLOVE', 'WORD2VEC'] # embeddings not used in paper
        feature_set = [ 'STRUCT', 'GATE', 'STRUCT+GATE', 'STRUCT+GATE+TFIDF', 'TFIDF', 'STRUCT+TFIDF', 'GATE+TFIDF']
        run_gridsearchcv(config1, periods, feature_set)
    elif args.run_type[0] == 'test':
        print('-- Running test...')
        config = ('SVM', LinearSVC(random_state=SEED, C=1, class_weight='balanced', dual=False, loss='squared_hinge'), 'TFIDF', [30])
        run_test()
    else:
        parser.usage()