# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:43:03 2018

@author: ABittar

Utility functions to query CRIS and prepare data for experiments described in 
the paper "Text Classification to Inform Suicide Risk Assessment in Electronic 
Health Records", Bittar A, Velupillai S, Roberts A, Dutta R., from MedInfo 2019 
(https://www.ncbi.nlm.nih.gov/pubmed/31437881).
"""

import os
import pandas as pd
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from copy import copy
from scipy.sparse import csr_matrix


def concatenate_text_by_day(ctype, emo_source=None):
    """
    Concatenate all texts on a single day, per admission (admission-days)
    ctype: 'case' or 'control'
    emo_source: the sentiment lexicon that was used to extract sentiment words
                (afinn, emolex, liwc, opinion, pattern, swn) for use as 
                classification features, or None if no sentiment words have 
                been extracted and stored in the data.
    """
    
    print('-- Concatenating text by day for', ctype, 'and', emo_source)
    
    if emo_source is None:
        pin = 'Z:/Andre Bittar/Projects/eHOST-IT/data/' + ctype + '_30_text_ordinal_dates_p2.pickle'
        pout = 'Z:/Andre Bittar/Projects/eHOST-IT/data/' + ctype + '_30_text_per_day_' + emo_source + '_p2.pickle'
        cols = ['pk', 'brcid_' + ctype, 'text_' + ctype, 'day']
    else:
        pin = 'Z:/Andre Bittar/Projects/eHOST-IT/data/' + ctype + '_30_text_ordinal_dates_' + emo_source + '_p2.pickle'
        pout = 'Z:/Andre Bittar/Projects/eHOST-IT/data/clpsych/text_per_day/' + ctype + '_30_text_per_day_' + emo_source + '_p2.pickle'
        cols = ['pk', 'brcid_' + ctype, 'text_' + ctype, 'day', 'pos_' + emo_source, 'neg_' + emo_source, 'pos_words_' + emo_source, 'neg_words_' + emo_source]
    
    print('-- Loading data...', end='')
    df = pd.read_pickle(pin)
    print('Done.')

    df_day = pd.DataFrame(columns=cols)
    n = 0
    grouped = df.groupby(['pk', 'Date_ord_norm', 'brcid_' + ctype])
    t = len(grouped)
    for g in grouped:
        pk = g[0][0]
        brcid = g[0][2]
        day = g[0][1]
        text = '\n'.join(g[1]['text_' + ctype])

        d = {}
        if emo_source is None:
            d = [{'pk': pk, 'brcid_' + ctype: brcid, 'text_' + ctype: text, 'day': day}]
        else:
            posw = [item for sublist in g[1]['pos_words'] for item in sublist]
            negw = [item for sublist in g[1]['neg_words'] for item in sublist]
            d['pk'] = pk
            d['brcid_' + ctype] = brcid
            d['text_' + ctype] = text
            d['day'] = day
            d['pos_' + emo_source] = sum(g[1]['pos'])
            d['neg_' + emo_source] = sum(g[1]['neg'])
            d['pos_words_' + emo_source] = posw
            d['neg_words_' + emo_source] = negw

            if emo_source == 'emolex':
                d['ang'] = sum(g[1]['anger'])
                d['ant'] = sum(g[1]['anticipation'])
                d['dis'] = sum(g[1]['disgust'])
                d['fea'] = sum(g[1]['fear'])
                d['joy'] = sum(g[1]['joy'])
                d['sad'] = sum(g[1]['sadness'])
                d['sur'] = sum(g[1]['surprise'])
                d['tru'] = sum(g[1]['trust'])
        
        df_tmp = pd.DataFrame([d], index=[n])
        df_day = df_day.append(df_tmp)
        
        if n % 1000 == 0:
            print(n, '/', t)
        
        n += 1
    
    df_day['day'] = pd.to_numeric(df_day.day)
    
    df_day.to_pickle(pout)
    
    print('-- Wrote file:', pout)
    
    return df_day


def concatenate_case_text(df, period, save_to_disk=False):
    """
    Concatenate all text for cases into one text per admission.
    df: the dataframe with each text in a separate row.
    period: 30, 90 or 180 days
    """
    new_df = pd.DataFrame(columns=['pk', 'text_case'])
    n = 0
    m = 0
    
    print('-- Concatenating and preprocessing text', file=sys.stderr)
    
    for p in df.pk.unique():
        texts = df.loc[df.pk == p]['text_case']
        m += len(texts)
        new_text = ''
        for t in texts:
            new_text += '\n' + t
        df_temp = pd.DataFrame(list({p: new_text}.items()),
                               columns=['pk', 'text_case'], index=[n])
        new_df = new_df.append(df_temp)
        print(n, file=sys.stderr)
        n += 1

    print('-- Total concatenated documents:', m)
    
    if save_to_disk:
        pout = 'Z:/Andre Bittar/Projects/eHOST-IT/data/case_' + str(period) + '_text.pickle'
        new_df.to_pickle(pout)
        print('-- Saved output to ', pout, file=sys.stderr)
      
    return new_df


def concatenate_control_text(df, period, save_to_disk=False):
    """
    Concatenate all text for controls into one text per admission.
    Requires a different method to that used for cases.
    df: the dataframe with each text in a separate row.
    """
    df_control_agg_text = pd.DataFrame(columns=['pk', 'brcid_control', 'text_control'])
    n = 0
    m = 0

    print('-- Concatenating and preprocessing text', file=sys.stderr)

    for g in df.groupby(by=['pk', 'brcid_control']):
        control_texts = g[1].text_control
        text = '\n'.join(control_texts)
        m += len(control_texts)
        pk = g[0][0]
        brcid_control = g[0][1]
        d = [{'pk': pk, 'brcid_control': brcid_control, 'text_control': text}]
        df_tmp = pd.DataFrame(d, index=[n])
        df_control_agg_text = df_control_agg_text.append(df_tmp)
        print(n)
        n += 1

    print('-- Total concatenated documents:', m)

    if save_to_disk:
        pout = 'Z:/Andre Bittar/Projects/eHOST-IT/data/control_' + str(period) + '_text.pickle'
        df_control_agg_text.to_pickle(pout)
        print('-- Saved output to ', pout, file=sys.stderr)

    return df_control_agg_text


def load_dataframe_from_query(path):
    """
    TODO get import to work without changing directory.
    """
    from db_connection import fetch_dataframe
    query = open(path, 'r').read()
    return fetch_dataframe('BRHNSQL094', 'CDLS_Workspace', query)


def load_data_with_text():
    """
    Load all structured and text data for cases and controls and merge into a 
    single DataFrame.
    """
    from db_connection import fetch_dataframe
    server_name = 'BRHNSQL094'
    db_name = 'CDLS_Workspace'
    
    query = open('Z:/Andre Bittar/Projects/eHOST-IT/Andre_HES/case_control_final_with_additions_20181010_full_clean.sql').read()
    df = fetch_dataframe(server_name, db_name, query)
    
    df = df.loc[df.admission_no == 1]
    
    # Values are NO and YES instead of No and Yes
    df['Substance_Abuse_Past_case'] = df.Substance_Abuse_Past_case.apply(lambda x: 'No' if x == 'NO' else 'Yes')
    df['Substance_Abuse_Past_control'] = df.Substance_Abuse_Past_control.apply(lambda x: 'No' if x == 'NO' else 'Yes')
    
    case_columns = [c for c in df.columns if 'control' not in c and c not in 
                    ['admission_no', 'dob_case', 'admidate', 'age_band_case', 
                     'Accommodation_Status_Date_case', 'SGA_Date_case']]
    
    control_columns = [c for c in df.columns if 'control' in c and c not in 
                       ['control_number', 'dob_control', 'age_band_control', 
                        'Accommodation_Status_Date_control', 'SGA_Date_control']
                       or c == 'pk']
    
    df_case = df[case_columns].drop_duplicates()
    query_case = open('Z:/Andre Bittar/Projects/eHOST-IT/Andre_HES/cases_30_days_text.sql').read()
    case_text = fetch_dataframe(server_name, db_name, query_case)
    case_text_concat = concatenate_case_text(case_text, 30)
    df_case_all = df_case.merge(case_text_concat, on='pk', how='left')
    df_case_all['text_case'] = df_case_all.text_case.fillna(value='')
    df_case_all['category'] = 1
    
    df_control = df[control_columns]
    query_control = open('Z:/Andre Bittar/Projects/eHOST-IT/Andre_HES/controls_30_days_text.sql').read()
    control_text = fetch_dataframe(server_name, db_name, query_control)
    control_text_concat = concatenate_control_text(control_text, 30)
    df_control_all = df_control.merge(control_text_concat, on=['pk', 'brcid_control'], how='left')
    df_control_all['text_control'] = df_control_all.text_control.fillna(value='')
    df_control_all['category'] = 0
    
    for col in df_case_all.columns:
        df_case_all.rename(columns={col: col.replace('_case', '')}, inplace=True)
    
    for col in df_control_all.columns:
        df_control_all.rename(columns={col: col.replace('_control', '')}, inplace=True)
    
    df_case_all.drop(['pk', 'brcid'], axis=1, inplace=True)
    df_control_all.drop(['pk', 'brcid'], axis=1, inplace=True)
    
    df_all = df_case_all.append(df_control_all)
    df_all['age_admistart'] = df_all['age_admistart'].astype(int)

    return df_all


def load_raw_data():
    """
    Load all structured data (no text) from query into a DataFrame
    """
    from db_connection import fetch_dataframe # run from utils for this to work (until package install)
    path = 'Z:/Andre Bittar/Projects/eHOST-IT/Andre_HES/case_control_final_with_additions_20181010_full_clean.sql'
    server_name = 'BRHNSQL094'
    db_name = 'CDLS_Workspace'
    query = open(path).read()
    df = fetch_dataframe(server_name, db_name, query)
    df_1 = df.loc[df.admission_no == 1] # get first admission only
    
    case_columns = [c for c in df_1.columns if 'control' not in c and c not in 
                    ['pk', 'admission_no', 'brcid_case', 'dob_case', 
                     'admidate', 'age_band_case', 
                     'Accommodation_Status_Date_case', 'SGA_Date_case']]
    
    control_columns = [c for c in df_1.columns if 'control' in c and c not in 
                       ['brcid_control', 'control_number', 'dob_control', 'age_band_control', 
                     'Accommodation_Status_Date_control', 'SGA_Date_control']]
    
    df_case = df_1[case_columns]
    df_case['category'] = 1
    df_case.drop_duplicates(inplace=True)
    
    df_control = df_1[control_columns]
    df_control['category'] = 0
    
    for col in df_case.columns:
        df_case.rename(columns={col: col.replace('_case', '')}, inplace=True)

    for col in df_control.columns:
        df_control.rename(columns={col: col.replace('_control', '')}, inplace=True)
    
    df = df_case.append(df_control)
    
    return df


def save_as_ehost_text():
    """
    Save texts for annotation in eHOST annotation tool.
    Careful: this will generate a reandom sample at each execution.
    """
    df = pd.read_pickle('Z:/Andre Bittar/Projects/eHOST-IT/data/case_30_text_ordinal_dates.pickle')
    df = df.groupby('brcid_case').filter(lambda x: set(range(1, 31)).difference(set(x.Date_ord_norm.unique())) == set())

    brcid_sample_30 = df.brcid_case.unique().tolist()
    df = df.loc[df.brcid_case.isin(random.sample(brcid_sample_30, 10))]
    
    for i, row in df.iterrows():
        brcid = str(row.brcid_case)
        cndocid = str(row.CN_Doc_ID)
        text = row.text_case
        date = str(row.admidate.strftime('%Y-%m-%d'))
        dout = os.path.join('Z:/Andre Bittar/Projects/eHOST-IT/data/annotations', brcid)
        corpus_dir = os.path.join(dout, 'corpus')
        
        if not os.path.isdir(dout):
            os.mkdir(dout)
            os.mkdir(os.path.join(dout, 'config'))
            os.mkdir(corpus_dir)
            os.mkdir(os.path.join(dout, 'saved'))
        
        fout = date + '_' + cndocid + '_00001.txt'
        pout = os.path.join(corpus_dir, fout)
        
        while os.path.isfile(pout):
            match = re.search('_([0-9]+).txt', pout)
            if match is not None:
                fout = date + '_' + cndocid + '_' + str(int(match.group(1)) + 1).zfill(5) + '.txt'
                pout = os.path.join(corpus_dir, fout)
        
        with open(pout, 'w') as output:
            output.write(text)
        output.close()
        
        print('-- Wrote file:', pout)