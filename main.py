#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:43:01 2021

@author: aqiph

"""

import os, warnings
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_size(12)

from utils.metrics import calculate_PR, aupr
from utils.compound_selection_utils import isPositive, combine_multiple_expts, jaccard_similarity



### Calculate, print and plot AUPR, precision and recall ###
def calculate_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label'):
    """
    Helper function for print_PR_for_multiple_expts().
    Calculate AUPR, precision and recall for multiple experiments.
    :param srcDir: str, directory path of predicted results.
    :param threshold: float, threshold to label predicted results.
    :param prediction_column_name: str, column name for predicted scores.
    :param target_column_name: str, column name for true labels.
    :return: pd.DataFrame, calculated precision and recall.
    """
    expts = []   # experiment names
    df_sum = pd.DataFrame()

    # read df and compute precision and recall
    files = os.listdir(srcDir)

    for file in files:
        if os.path.splitext(file)[1] != '.csv':
            continue
        try:
            # read file
            expt = os.path.splitext(file)[0]
            expts.append(expt)
            df = pd.read_csv(os.path.join(srcDir, file))
            # drop the row if Label is Nan
            df.dropna(subset=target_column_name, how='any', inplace = True)
            print('{} is read, number of rows: {}'.format(expt, df.shape[0]))
            # compute AUPR
            aupr_0, aupr_1, aupr_hmean = aupr(df[prediction_column_name].tolist(), df[target_column_name].tolist())
            metrics_list = [aupr_1, aupr_0, aupr_hmean]
            # compute precision and recall
            df['Predicted_Label'] = df[prediction_column_name].apply(lambda p: int(p >= threshold))
            precision_1, recall_1, precision_0, recall_0 = calculate_PR(df['Predicted_Label'].tolist(), df[target_column_name].tolist())
            metrics_list += [precision_1, recall_1, precision_0, recall_0]
            df_sum[expt] = metrics_list
        except:
            pass

    # calculate mean and std, change to standard format
    df_sum.rename(index={0: 'AUPR_1', 1: 'AUPR_0', 2: 'AUPR_hmean', 3: 'Precision_1', 4: 'Recall_1', 5: 'Precision_0', 6: 'Recall_0'}, inplace=True)
    df_sum['Mean'] = df_sum.apply(lambda row: row.mean(), axis=1)
    df_sum['Std'] = df_sum.apply(lambda row: row.std(), axis=1)
    df_sum = df_sum.map(lambda x: '{:2.4f}'.format(x))
    df_sum['Ave ± Std'] = df_sum.apply(lambda row: row['Mean'] + '\u00B1' + row['Std'], axis=1)
    df_sum = pd.DataFrame(df_sum, columns=sorted(expts) + ['Ave ± Std'])
    df_sum = df_sum.transpose()

    return df_sum


def print_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label', output_file = None):
    """
    Calculate and print AUPR, precision and recall for multiple experiments.
    :param srcDir: str, directory path of predicted results.
    :param threshold: float, threshold to label predicted results.
    :param prediction_column_name: str, column name for predicted scores.
    :param target_column_name: str, column name for true labels.
    :param output_file: str, name of the output file.
    """
    # output file
    folder, basename = os.path.split(os.path.abspath(srcDir))
    if output_file is None:
        output_file = basename
    output_file = os.path.join(folder, os.path.splitext(output_file)[0] + '_PR.csv')

    # calculate precision and recall for multiple experiments
    df_sum = calculate_PR_for_multiple_expts(srcDir, threshold, prediction_column_name, target_column_name)

    # write to file
    print('Number of rows:', df_sum.shape[0])
    df_sum.to_csv(output_file)


def plot_PR(input_file, num_points, prediction_column_name='score', target_column_name='Label'):
    """
    Plot precision recall curves for single experiment.
    :param input_file: str, path of the file containing predicted scores and true labels.
    :param num_points: int, number of points on the plot.
    :param prediction_column_name: str, column name for predicted scores.
    :param target_column_name: str, column name for true labels.
    """
    # read files
    df = pd.read_csv(input_file, index_col = 0)
    predictions = df[prediction_column_name].values.tolist()
    true_labels = df[target_column_name].values.tolist()

    # output file
    folder, basename = os.path.split(os.path.abspath(input_file))
    output_file, fmt = os.path.splitext(basename)
    output_file = os.path.join(folder, '{}_{}_Precision_Recall'.format(output_file, prediction_column_name))

    # calculate thresholds
    step = 1.0 / num_points
    thresholds = [step * i for i in range(num_points + 1)]

    # calculate precisions and recalls
    PR_dict = {'Precision_1': [], 'Recall_1': [], 'Precision_0': [], 'Recall_0': []}

    for threshold in thresholds:
        predicted_labels = [int(p >= threshold) for p in predictions]
        precision_1, recall_1, precision_0, recall_0 = calculate_PR(predicted_labels, true_labels)

        PR_dict['Precision_1'].append(precision_1)
        PR_dict['Recall_1'].append(recall_1)
        PR_dict['Precision_0'].append(precision_0)
        PR_dict['Recall_0'].append(recall_0)

    # plot
    plt.figure(1)
    plt.plot(thresholds, PR_dict['Precision_1'], label='Precision[1]', color='blue')
    plt.plot(thresholds, PR_dict['Recall_1'], label='Recall[1]', color='red')
    plt.plot(thresholds, PR_dict['Precision_0'], label='Precision[0]', color='blue', linestyle='dashed')
    plt.plot(thresholds, PR_dict['Recall_0'], label='Recall[0]', color='red', linestyle='dashed')

    plt.grid(True)
    plt.xlabel('Thresholds', fontproperties=font)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)

    plt.legend(frameon=False, fontsize=12)
    plt.title('Precision/Recall vs. Threshold', fontproperties=font)
    plt.savefig(output_file + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # write to file
    df_PR = pd.DataFrame(PR_dict)
    df_PR = df_PR.map(lambda x: '{:2.4f}'.format(x))
    df_PR['Threshold'] = pd.Series(thresholds).apply(lambda x: '{:2.2f}'.format(x))
    df_PR = pd.DataFrame(df_PR, columns=['Threshold','Precision_1','Recall_1','Precision_0','Recall_0'])
    df_PR.to_csv(output_file + '.csv')


### Select data based on rules ###
def select_compounds(input_file, threshold_list, how = 'any', output_file = None, output_option='selected'):
    """
    Use the given strategy to return a subset of selected compounds,
    satisfy that the score of the nth experiment >= the nth threshold.
    :param input_file: str, path of the file containing predicted scores and true labels.
    :param threshold_list: list of floats, thresholds for different experiments.
    :param how: str, how to label, 'all', 'any', 'vote' and 'average'.
    :param output_file: str, name of the output file.
    :param output_option: str, options to output data, allowed values include 'selected', 'not_selected' and 'all'.
    :return: list of floats, precision_1, recall_1, precision_0, recall_0 for current strategy.
    """
    # read files
    df_prediction = pd.read_csv(input_file, index_col = 0)
    COLUMNS = df_prediction.columns.tolist()
    if 'True_Label' not in COLUMNS:
        df_prediction['True_Label'] = np.nan
        COLUMNS.append('True_Label')
    expts = df_prediction.columns.tolist()[2:-1]
    assert len(expts) == len(threshold_list), 'Error: Number of experiments should be same as number of thresholds'
    
    # output file
    folder, basename = os.path.split(os.path.abspath(input_file))    
    if output_file is None:
        output_file = basename
    output_file = os.path.join(folder, os.path.splitext(output_file)[0] + '_' + how)
    
    # get selected compounds
    df_prediction['Selected'] = df_prediction.apply(lambda row: isPositive([row[expt] for expt in expts], threshold_list, how), axis=1)
    if output_option == 'selected':
        df = pd.DataFrame(df_prediction[df_prediction['Selected']], columns=COLUMNS)
    elif output_option == 'not_selected':
        df = pd.DataFrame(df_prediction[~df_prediction['Selected']], columns=COLUMNS)
    elif output_option == 'all':
        df = pd.DataFrame(df_prediction, columns = COLUMNS+['Selected'])
        df['Selected'] = df['Selected'].apply(lambda x: int(x))
    else:
        raise Exception('Error: Invalid output option.')

    # calculate precision and recall
    precision_1, recall_1, precision_0, recall_0 = calculate_PR(df_prediction['Selected'].tolist(),df_prediction['True_Label'].tolist())

    # write to file
    print('Number of rows:', df.shape[0])
    df.to_csv(f'{output_file}_{df.shape[0]}.csv')

    return precision_1, recall_1, precision_0, recall_0


def get_jaccard_similarity(input_file, threshold):
    """
    Calculate Jaccard similarity between two libraries.
    :param input_file: str, path of the file containing predicted scores and true labels.
    :param threshold: float, threshold to label predicted results.
    """
    # read files
    df = pd.read_csv(input_file, index_col=0)
    expts = df.columns.tolist()[2:7]

    # output file
    folder, basename = os.path.split(os.path.abspath(input_file))
    output = open(os.path.join(folder, 'jaccard_similarity.txt'), 'w')

    # calculate jaccard similarity
    threshold_list = [threshold, threshold]

    for i in range(len(expts)):
        for j in range(i + 1, len(expts)):
            score = jaccard_similarity(df, threshold_list, expts[i], expts[j])
            output.write('{}_{}    {:4.4f}\n'.format(expts[i], expts[j], score))

    output.close()


def select_compounds_by_multi_strategies(srcDir, input_file_target = None, threshold = 0.5, output_file = None, output_option='selected'):
    """
    Use multiple strategies to return a subset of selected compounds,
    satisfy that the score of the nth experiment >= the nth threshold.
    :param srcDir: str, directory path of predicted results. None, if no True label is provided.
    :param input_file_target: str, path of the file containing true labels.
    :param threshold: float, threshold to label predicted results.
    :param output_file: str, name of the output file.
    :param output_option: str, options to output data, allowed values include 'selected', 'not_selected' and 'all'.
    """
    # Parameters: strategies, flag_cal_PR
    strategies = ['any', 'vote', 'all', 'average']
    flag_cal_PR = True
    if input_file_target is None:
        flag_cal_PR = False

    # output file
    folder, basename = os.path.split(os.path.abspath(srcDir))
    if output_file is None:
        output_file = basename
    output_file = os.path.splitext(output_file)[0]

    # calculate AUPR, precision and recall
    if flag_cal_PR:
        df_sum = calculate_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label')
    else:
        df_sum = pd.DataFrame()

    # combine multiple predicted results and true labels for selecting compounds
    num_cmps, num_expts = combine_multiple_expts(srcDir, input_file_target, output_file)
    print('Combining results done!')

    # use different strategies to select compounds
    input_file = os.path.join(folder, output_file + f'_{num_cmps}.csv')
    threshold_list = [threshold for _ in range(num_expts)]
    for i, strategy in enumerate(strategies):
        precision_1, recall_1, precision_0, recall_0 = select_compounds(input_file, threshold_list, how=strategy, output_file=output_file, output_option=output_option)
        if flag_cal_PR:
            strategy_PR = pd.DataFrame({'AUPR_1': [np.nan], 'AUPR_0': [np.nan], 'AUPR_hmean': [np.nan],
                'Precision_1':[precision_1], 'Recall_1':[recall_1], 'Precision_0':[precision_0], 'Recall_0':[recall_0]})
            strategy_PR.rename(index={0:strategy}, inplace=True)
            strategy_PR = strategy_PR.map(lambda x: '{:2.4f}'.format(x))
            df_sum = pd.concat([df_sum, strategy_PR], ignore_index=False)

    # write performance to file
    if flag_cal_PR:
        print('Number of rows:', df_sum.shape[0])
        df_sum.to_csv(os.path.join(folder, f'{output_file}_PR.csv'))

    # compute jaccard similarity between every two experiments
    get_jaccard_similarity(input_file, threshold)



if __name__ == '__main__':

    ### Calculate and print precision and recall for multiple experiments ###
    # srcDir = 'tests/prediction'
    # threshold = 0.5
    # output_file = 'predicted_results'
    # print_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label',
    #                             output_file=output_file)

    ### Plot precision and recall ###
    # input_file = 'tests/expt.csv'
    # num_points = 100
    # plot_PR(input_file, num_points, prediction_column_name='score', target_column_name='Label')

    # ### Select data based on rules ###
    # input_file = 'tests/combine_39.csv'
    # threshold_list = [0.5, 0.5, 0.5, 0.5, 0.5]
    # select_compounds(input_file, threshold_list, how='any', output_file=None, output_option='selected')

    ### Select data based on multiple rules ###
    srcDir = 'tests/prediction'
    input_file_target = 'tests/target.csv'
    threshold = 0.5
    output_file = 'predicted_results'
    select_compounds_by_multi_strategies(srcDir, input_file_target, threshold, output_file, output_option='selected')

    ### Calculate jaccard similarity between different experiments ###
    # input_file = 'tests/combine_39.csv'
    # threshold = 0.5
    # get_jaccard_similarity(input_file, threshold=threshold)

    
    

