# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:31:53 2018

@author: ABittar

Utility methods to plot words frequencies over time.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from db_connection import server_name, db_name, fetch_dataframe

def process_cases():
    query_case = open('Z:/Andre Bittar/Projects/eHOST-IT/Andre_HES/cases_30_days_text.sql', 'r').read()

    df = fetch_dataframe(server_name, db_name, query_case)

    df['Date'] = pd.to_datetime(df.Date)

    grouping = df.groupby('pk')

    new_df = pd.DataFrame()
    for group in grouping:
        group[1]['Date_ord'] = group[1].Date.apply(lambda x: x.toordinal())
        nmax = group[1]['Date_ord'].max()
        nmin = group[1]['Date_ord'].min()
        group[1]['Date_ord_norm'] = group[1].Date_ord.apply(lambda x: (1 + (nmax - nmin) * (x - nmin) / (nmax - nmin)))
        group[1]['Date_ord_norm'].fillna(value=1.0, inplace=True)
        new_df = new_df.append(group[1])

    #new_df.to_pickle('Z:/Andre Bittar/Projects/eHOST-IT/data/case_30_text_ordinal_dates.pickle')
    
    return new_df


def process_controls():
    query_case = open('Z:/Andre Bittar/Projects/eHOST-IT/Andre_HES/controls_30_days_text.sql', 'r').read()

    df = fetch_dataframe(server_name, db_name, query_case)

    df['Date'] = pd.to_datetime(df.Date)

    grouping = df.groupby('pk')

    new_df = pd.DataFrame()
    for group in grouping:
        group[1]['Date_ord'] = group[1].Date.apply(lambda x: x.toordinal())
        nmax = group[1]['Date_ord'].max()
        nmin = group[1]['Date_ord'].min()
        group[1]['Date_ord_norm'] = group[1].Date_ord.apply(lambda x: (1 + (nmax - nmin) * (x - nmin) / (nmax - nmin)))
        group[1]['Date_ord_norm'].fillna(value=1.0, inplace=True)
        new_df = new_df.append(group[1])

    #new_df.to_pickle('Z:/Andre Bittar/Projects/eHOST-IT/data/control_30_text_ordinal_dates.pickle')
    
    return new_df


def plot_ordinal_date_mentions_separately(pop_type='case', terms=['overdose'], plot_type='area'):
    """
    Plot word frequency over time.
    """
    text_label = 'text_case'
    if pop_type == 'case':
        df = pd.read_pickle('Z:/Andre Bittar/Projects/eHOST-IT/data/case_30_text_ordinal_dates.pickle')
    elif pop_type == 'control':
        df = pd.read_pickle('Z:/Andre Bittar/Projects/eHOST-IT/data/control_30_text_ordinal_dates.pickle')
        text_label = 'text_control'

    for term in terms:
        df[term] = df[text_label].apply(lambda x: len(re.findall(term + '[^a-z]', x, flags=re.I)))

        if pop_type == 'case':
            df_final = df[['Date_ord_norm', term]].groupby('Date_ord_norm').sum()
        if pop_type == 'control':
            # divide by 4 as there are 4 times more controls than cases
            df_final = df[['Date_ord_norm', term]].groupby('Date_ord_norm').sum() / 4.0
        
        plot_path = 'T:/Andre Bittar/workspace/ehost-it/plots/30d_' + pop_type + '_' + term + '_mentions_' + plot_type +'.png'
        if plot_type == 'pie':
            ax = df_final.plot(kind=plot_type, subplots=True)
            fig = ax.get_figure()
            #fig.savefig(plot_path)
        else:
            ax = df_final.plot(kind=plot_type, xlim=(1, 30), ylim=(0, 5000))
            fig = ax.get_figure()
            #fig.savefig(plot_path)
        plt.show()


def plot_ordinal_date_mentions(df_case, df_control, terms=['overdose'], plot_type='area', weighting='tokens', split_y_axis=False):
    """
    Plot word frequency over time for both cases and controls.
    """
    df_case = pd.read_pickle('Z:/Andre Bittar/Projects/eHOST-IT/data/case_30_text_ordinal_dates.pickle')
    df_control = pd.read_pickle('Z:/Andre Bittar/Projects/eHOST-IT/data/control_30_text_ordinal_dates.pickle')
    
    for term in terms:
        key_case = term + ' (cases)'
        key_control = term + ' (controls)'

        # identify all tokens as per TfIdfVectorizer
        df_case['text_case_tok'] = df_case['text_case'].apply(lambda x: re.findall('[A-Za-z][\w\-]+', x, flags=re.I))
        df_control['text_control_tok'] = df_control['text_control'].apply(lambda x: re.findall('[A-Za-z][\w\-]+', x, flags=re.I))

        df_case[key_case] = df_case['text_case_tok'].apply(lambda x: x.count(term))
        df_control[key_control] = df_control['text_control_tok'].apply(lambda x: x.count(term))
        
        df_case['tokens'] = df_case.text_case_tok.apply(lambda x: len(x))
        df_control['tokens'] = df_control.text_control_tok.apply(lambda x: len(x))
        
        df_case_final = df_case[['Date_ord_norm', 'tokens', key_case]].groupby('Date_ord_norm').sum().astype(float)
        df_control_final = df_control[['Date_ord_norm', 'tokens', key_control]].groupby('Date_ord_norm').sum() / 4 # divide by 4 to account for case-tronol ratio
        df_control_final[key_control] = df_control_final[key_control].round()

        y_upper = 5000

        if weighting == 'tokens':
            # calculate as words-per-million
            df_case_final[key_case] = df_case_final[key_case] * 1000000 / df_case_final.tokens
            df_control_final[key_control] = df_control_final[key_control] * 1000000 / df_control_final.tokens
            
            df_case_final = df_case_final.drop('tokens', axis=1)
            df_control_final = df_control_final.drop('tokens', axis=1)

            y_upper = max(df_case_final[key_case].max(), df_control_final[key_control].max())
            y_next = max(np.sort(df_case_final[key_case].values)[-2], np.sort(df_control_final[key_control].values)[-2])

        elif weighting == 'docs':
            # NOTE - this is no good - use token weighting, as that is the 
            # true measure of how much text there is per group
            # weighting accounts for number of total documents for each day
            weights_case = []
            for x in df_case.Date_ord_norm.unique():
                f = len(df_case.loc[df_case.Date_ord_norm == x])# / len(df_case)
                weights_case.append(f)
                print('cases   : {:3}{:10}'.format(x, f))

            weights_control = []
            for x in df_control.Date_ord_norm.unique():
                f = len(df_control.loc[df_control.Date_ord_norm == x])# / len(df_control)
                weights_control.append(f)
                print('controls: {:3}{:10}'.format(x, f))
        
            n = 1
            for i in range(len(weights_case)):
                print('case:', float(df_case_final[key_case][n]), '/', weights_case[i], '=', df_case_final[key_case][n] / weights_case[i])
                df_case_final[key_case][n] = df_case_final[key_case][n] / weights_case[i] #* (1 - weights_case[i])
                n += 1

            n = 1
            for i in range(len(weights_control)):
                print('ctrl:', df_control_final[key_control][n], '/', weights_control[i], '=', df_control_final[key_control][n] / weights_control[i])
                df_control_final[key_control][n] = df_control_final[key_control][n] / weights_control[i] # * (1 - weights_control[i])
                n += 1
            
            # divide by 4 to normalise for case-control ratio
            y_upper = max(df_case_final[key_case].max(), df_control_final[key_control].max())
            y_next = max(np.sort(df_case_final[key_case].values)[-2], np.sort(df_control_final[key_control].values)[-2])
        
        df_case_final.reset_index(inplace=True)
        df_control_final.reset_index(inplace=True)
        df_final = df_case_final.merge(df_control_final)
        
        #df_final.to_pickle('T:/Andre Bittar/workspace/ehost-it/plots_slides/cc_30d_'  + '_' + term + '_mentions.pickle')
        
        #print('wcse', weights_case)
        #print('wcon', weights_control)
        print(df_final)
        
        plot_path = 'T:/Andre Bittar/workspace/ehost-it/plots_slides/cc_30d_'  + '_' + term + '_mentions_' + plot_type +'.png'
        if plot_type == 'pie':
            ax = df_final.plot(kind=plot_type, subplots=True)
            fig = ax.get_figure()
            #fig.savefig(plot_path, bbox_inches='tight')
        elif plot_type == 'area':
            if split_y_axis:
                f, axis = plt.subplots(2, 1, sharex=True)
                df_final.plot(kind='area', x='Date_ord_norm', xlim=(30, 1), stacked=False, ax=axis[0])
                df_final.plot(kind='area', x='Date_ord_norm', xlim=(30, 1), stacked=False, ax=axis[1])

                axis[0].set_ylim(y_upper - (y_upper / 3), y_upper)
                axis[1].set_ylim(0, y_next)
                axis[1].legend().set_visible(False)
            
                axis[1].set_xlabel('Days before admission')
                axis[1].set_ylabel('Word count')
            
                axis[0].spines['bottom'].set_visible(False)
                axis[1].spines['top'].set_visible(False)
                axis[0].xaxis.tick_top()
                axis[0].tick_params(labeltop='off')
                axis[1].xaxis.tick_bottom()
                d = .015
                kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
                axis[0].plot((-d,+d),(-d,+d), **kwargs)
                axis[0].plot((1-d,1+d),(-d,+d), **kwargs)
                kwargs.update(transform=axis[1].transAxes)
                axis[1].plot((-d,+d),(1-d,1+d), **kwargs)
                axis[1].plot((1-d,1+d),(1-d,1+d), **kwargs)
            
                h1 = '///'
                h2 = '\\\\\\'
            
                axis[0].collections[0].set_hatch(h1)
                axis[0].collections[1].set_hatch(h2)
                axis[0].collections[0].set_color('#D9D75E')
                axis[0].collections[1].set_color('#76A0BD')
                axis[0].get_lines()[0].set_color('#D9D75E')
                axis[0].get_lines()[1].set_color('#76A0BD')

                axis[1].collections[0].set_hatch(h1)
                axis[1].collections[1].set_hatch(h2)
                axis[1].collections[0].set_color('#D9D75E')
                axis[1].collections[1].set_color('#76A0BD')
                axis[1].get_lines()[0].set_color('#D9D75E')
                axis[1].get_lines()[1].set_color('#76A0BD')
            
                axis[0].collections[0].set_linewidth(2)
                axis[0].collections[1].set_linewidth(2)
                axis[1].collections[0].set_linewidth(2)
                axis[1].collections[1].set_linewidth(2)
                
                axis[0].legend()

                fig = axis[0].get_figure()
                #fig.savefig(plot_path)
            else:
                ax = df_final.plot(kind='area', x='Date_ord_norm', xlim=(30, 1), ylim=(0, y_upper), stacked=False)

                h1 = '///'
                h2 = '\\\\\\'

                ax.set_xlabel('Days before admission')
                ax.set_ylabel('Term frequency (words per million)')

                ax.collections[0].set_hatch(h1)
                ax.collections[1].set_hatch(h2)
                ax.collections[0].set_color('#D9D75E')
                ax.collections[1].set_color('#76A0BD')
                ax.get_lines()[0].set_color('#D9D75E')
                ax.get_lines()[1].set_color('#76A0BD')
                ax.collections[0].set_linewidth(2)
                ax.collections[1].set_linewidth(2)
                ax.legend()

                fig = ax.get_figure()
                #fig.savefig(plot_path, bbox_inches='tight')
        else:
            ax = df_final.plot(kind=plot_type, xlim=(30, 1), ylim=(0, y_upper))
            fig = ax.get_figure()
            #fig.savefig(plot_path, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    plot_ordinal_date_mentions_separately(pop_type='case', terms=['overdose', 'suicide', 'happy'], plot_type='area')