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



### Combine results ###

def combine_multiple_expts(srcDir, input_file_target = None, output_file = None):
    """
    combine predicted results and true labels
    :param srcDir: str, directory path of predicted results
    :param input_file_target: str, path of the file containing true labels
    :param output_file: str, name of the output file
    :return: int, number of compounds
    """
    num = 0
    expts = []  # experiment names
    dfs = []
    
    # read df
    files = os.listdir(srcDir)
    for file in files:
        if os.path.splitext(file)[1] != '.csv':
            continue
        
        try:
            expt = os.path.splitext(file)[0]
            expts.append(expt)
            df = pd.read_csv(os.path.join(srcDir, file))
            df = pd.DataFrame(df, columns = ['ID', 'Cleaned_SMILES', 'score'])
            df.rename(columns = {'score': expt}, inplace = True)
            dfs.append(df)
            print('{} is read, number of rows: {}'.format(expt, df.shape[0]))
            num += 1

        except:
            pass
    
    # output file
    folder, basename = os.path.split(os.path.abspath(srcDir))
    if output_file is None:
        output_file = basename
    output_file = os.path.join(folder, os.path.splitext(output_file)[0])
    
    # merge multiple experiments
    df_merge = reduce(lambda left, right: pd.merge(left, right, how = 'left', on = ['ID', 'Cleaned_SMILES']), dfs)
    columns = ['ID', 'Cleaned_SMILES'] + sorted(expts)
    df_merge = df_merge.reindex(columns, axis = 1)
    num_expts = df_merge.shape[1] - 2

    # merge target labels
    if input_file_target is not None:
        df_target = pd.read_csv(input_file_target)
        df_target = pd.DataFrame(df_target, columns=['ID', 'Cleaned_SMILES', 'Label'])
        df_target.rename(columns={'ID': 'Training_set_ID', 'Label': 'True_Label'}, inplace=True)
        df_merge = pd.merge(df_merge, df_target, how='left', on=['ID', 'Cleaned_SMILES'])
    
    # write to file
    # df_merge.sort_values(by=df_merge.columns.tolist()[2:], ascending=False, inplace=True)
    print('Number of rows in the output file:', df_merge.shape[0])
    df_merge.to_csv(output_file + f'_{df_merge.shape[0]}.csv')

    return df_merge.shape[0], num_expts


### Calculate and plot precision and recall ###

def calculate_PR(predicted_labels, true_labels):
    """
    calculate precision and recall
    :param predicted_labels: list of ints, list of predicted labels, i.e., 1 or 0
    :param true_labels: list of ints, list of true labels, i.e., 1, 0, Nan
    :return: list of floats, precision_1, recall_1, precision_0, recall_0
    """
    assert len(predicted_labels) == len(true_labels), 'Error: Number of predicted labels should be the same as that of true labels'
    TP, FP, TN, FN = 0, 0, 0, 0

    for i, pred_label in enumerate(predicted_labels):
        true_label = true_labels[i]
        try:
            pred_label, true_label = int(pred_label), int(true_label)
            if pred_label == 1 and true_label == 1:
                TP += 1
            elif pred_label == 1 and true_label == 0:
                FP += 1
            elif pred_label == 0 and true_label == 0:
                TN += 1
            elif pred_label == 0 and true_label == 1:
                FN += 1
        except:
            continue

    # precision for label 1
    precision_1 = np.nan if TP + FP == 0 else TP * 1.0 / (TP + FP)
    # recall for label 1
    recall_1 = np.nan if TP + FN == 0 else TP * 1.0 / (TP + FN)
    # precision for label 0
    precision_0 = np.nan if TN + FN == 0 else TN * 1.0 / (TN + FN)
    # recall for label 0
    recall_0 = np.nan if TN + FP == 0 else TN * 1.0 / (TN + FP)

    return precision_1, recall_1, precision_0, recall_0


def calculate_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label'):
    """
    calculate precision and recall for multiple experiments
    :param srcDir: str, directory path of predicted results
    :param threshold: float, threshold to label predicted results
    :param prediction_column_name: str, column name for predicted scores
    :param target_column_name: str, column name for true labels
    :return: pd.DataFrame, calculated precision and recall
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
            print('{} is read, number of rows: {}'.format(expt, df.shape[0]))

            # compute precision and recall
            df['Predicted_Label'] = df[prediction_column_name].apply(lambda p: int(p >= threshold))
            precision_1, recall_1, precision_0, recall_0 = calculate_PR(df['Predicted_Label'].tolist(), df[target_column_name].tolist())
            df_sum[expt] = [precision_1, recall_1, precision_0, recall_0]

        except:
            pass

    # calculate mean and std, change to standard format
    df_sum.rename(index={0: 'Precision_1', 1: 'Recall_1', 2: 'Precision_0', 3: 'Recall_0'}, inplace=True)
    df_sum['Mean'] = df_sum.apply(lambda row: row.mean(), axis=1)
    df_sum['Std'] = df_sum.apply(lambda row: row.std(), axis=1)
    df_sum = df_sum.applymap(lambda x: '{:2.4f}'.format(x))
    df_sum['Ave ± Std'] = df_sum.apply(lambda row: row['Mean'] + '\u00B1' + row['Std'], axis=1)
    df_sum = pd.DataFrame(df_sum, columns=sorted(expts) + ['Ave ± Std'])
    df_sum = df_sum.transpose()

    return df_sum


def print_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label', output_file = None):
    """
    calculate precision and recall for multiple experiments
    :param srcDir: str, directory path of predicted results
    :param threshold: float, threshold to label predicted results
    :param prediction_column_name: str, column name for predicted scores
    :param target_column_name: str, column name for true labels
    :param output_file: str, name of the output file
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
    plot precision recall curves
    :param input_file: str, path of the file containing predicted scores and true labels
    :param num_points: int, number of points on the plot
    :param prediction_column_name: str, column name for predicted scores
    :param target_column_name: str, column name for true labels
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
    df_PR = df_PR.applymap(lambda x: '{:2.4f}'.format(x))
    df_PR['Threshold'] = pd.Series(thresholds).apply(lambda x: '{:2.2f}'.format(x))
    df_PR = pd.DataFrame(df_PR, columns=['Threshold','Precision_1','Recall_1','Precision_0','Recall_0'])
    df_PR.to_csv(output_file + '.csv')


### Select data based on rules ###

def isPositive(scores, threshold_list, how='any'):
    """
    helper function for select_cmps()
    decide whether this compound is positive based on scores
    :param scores: list of floats, predicted scores for one compound
    :param threshold_list: list of floats, thresholds for different experiments
    :param how: str, how to label, 'all', 'any', 'vote' and 'average'
    :return: bool, whether this compound is positive
    """
    assert len(scores) == len(threshold_list), 'Error: Number of experiments should be same as number of thresholds'

    scores = [float(s) for s in scores]
    scores = np.array(scores)
    threshold_list = np.array(threshold_list)

    labels = scores >= threshold_list

    if how == 'any':
        return np.any(labels)

    elif how == 'all':
        return np.all(labels)

    elif how == 'vote':
        vote_score = np.sum(labels)
        return vote_score >= len(scores) / 2.0

    elif how == 'average':
        ave = np.mean(scores)
        return ave >= threshold_list[0]

    elif how == 'two':
        vote_score = np.sum(labels)
        return vote_score >= 2.0


def select_cmps(input_file, threshold_list, how = 'any', output_file = None, output_option='selected'):
    """
    return a subset of selected compounds, 
    satisfy that the score of the nth experiment >= the nth threshold
    :param input_file: str, path of the file containing predicted scores and true labels
    :param threshold_list: list of floats, thresholds for different experiments
    :param how: str, how to label, 'all', 'any', 'vote' and 'average'
    :param output_file: str, name of the output file
    :param output_option: str, options to output data, allowed values include 'selected', 'not_selected' and 'all'
    :return: list of floats, precision_1, recall_1, precision_0, recall_0 for current strategy
    """
    # read files
    df_prediction = pd.read_csv(input_file, index_col = 0)
    COLUMNS = df_prediction.columns.tolist()
    if 'Training_set_ID' not in COLUMNS:
        df_prediction['Training_set_ID'] = ''
        COLUMNS.append('Training_set_ID')
    if 'True_Label' not in COLUMNS:
        df_prediction['True_Label'] = np.nan
        COLUMNS.append('True_Label')
    expts = df_prediction.columns.tolist()[2:-2]
    assert len(expts) == len(threshold_list), 'Error: Number of experiments should be same as number of thresholds'
    
    # output file
    folder, basename = os.path.split(os.path.abspath(input_file))    
    if output_file is None:
        output_file = basename
    output_file = os.path.join(folder, os.path.splitext(output_file)[0] + '_' + how + '_')
    
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
    df.to_csv(output_file + f'{df.shape[0]}.csv')

    return precision_1, recall_1, precision_0, recall_0


def select_cmps_by_multi_strategies(srcDir, input_file_target = None, threshold = 0.5, output_file = None, output_option='selected'):
    """
    use multiple strategies to return a subset of selected compounds,
    satisfy that the score of the nth experiment >= the nth threshold
    :param srcDir: str, directory path of predicted results. None, if no True label is provided
    :param input_file_target: str, path of the file containing true labels
    :param threshold: float, threshold to label predicted results
    :param output_file: str, name of the output file
    :param output_option: str, options to output data, allowed values include 'selected', 'not_selected' and 'all'
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

    # combine predicted results and true labels
    num_cmps, num_expts = combine_multiple_expts(srcDir, input_file_target, output_file)
    print('Combining results done!')

    # calculate mean and std performance
    df_sum = pd.DataFrame()
    if flag_cal_PR:
        df_sum = calculate_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label')

    # select compounds
    input_file = os.path.join(folder, output_file + f'_{num_cmps}.csv')
    threshold_list = [threshold for _ in range(num_expts)]

    for i, strategy in enumerate(strategies):
        precision_1, recall_1, precision_0, recall_0 = select_cmps(input_file, threshold_list, how=strategy, output_file=output_file, output_option=output_option)
        if flag_cal_PR:
            strategy_PR = pd.DataFrame({'Precision_1':[precision_1], 'Recall_1':[recall_1], 'Precision_0':[precision_0], 'Recall_0':[recall_0]})
            strategy_PR.rename(index={0:strategy}, inplace=True)
            strategy_PR = strategy_PR.applymap(lambda x: '{:2.4f}'.format(x))
            df_sum = pd.concat([df_sum, strategy_PR], ignore_index=False)

    # write to file
    if flag_cal_PR:
        print('Number of rows:', df_sum.shape[0])
        df_sum.to_csv(os.path.join(folder, output_file + '_PR.csv'))

    # compute jaccard similarity between every two experiments
    get_jaccard_similarity(input_file, threshold)


### Calculate jaccard similarity between different experiments

def get_jaccard_similarity(input_file, threshold):
    """
    calculate Jaccard similarity between two libraries
    :param input_file: str, path of the file containing predicted scores and true labels
    :param threshold: float, threshold to label predicted results
    """
    # read files
    df = pd.read_csv(input_file, index_col = 0)    
    expts = df.columns.tolist()[2:-2]
    
    # output file
    folder, basename = os.path.split(os.path.abspath(input_file))  
    output = open(os.path.join(folder, 'jaccard_similarity.txt'), 'w')  
    
    # calculate jaccard similarity 
    threshold_list = [threshold, threshold]
    
    for i in range(len(expts)):
        for j in range(i+1, len(expts)):
            score = jaccard_similarity(df, threshold_list, expts[i], expts[j])
            output.write('{}_{}    {:4.4f}\n'.format(expts[i], expts[j], score))
    
    output.close()


def jaccard_similarity(df, threshold_list, expt1, expt2):
    """
    helper function for get_jaccard_similarity()
    calculate jaccard similarity (intersection/union) for expt1 and expt2 of df
    :param df: DataFrame object, pd.DataFrame containing expt1 and expt2
    :param threshold_list: list of floats, thresholds for different experiments
    :param expt1: str, column name for expt1
    :param expt2: str, column name for expt2
    :return: float, the calculated jaccard similarity
    """
    # calculate intersection
    df_intersect = df.loc[df.apply(lambda row: isPositive([row[expt1], row[expt2]], threshold_list, how = 'all'), axis = 1)]
    intersection = df_intersect.shape[0]
    
    # calculate union
    df_union = df.loc[df.apply(lambda row: isPositive([row[expt1], row[expt2]], threshold_list, how = 'any'), axis = 1)]
    union = df_union.shape[0]
    
    # calculate jaccard similarity
    if union <= 0.0:
        return 0
    similarity = float(intersection)/float(union)
    return similarity



if __name__ == '__main__':

    ### Combine results ###
    srcDir = 'tests/prediction'
    input_file_target = 'tests/target.csv'
    output_file = 'combine'
    combine_multiple_expts(srcDir, input_file_target=input_file_target, output_file=output_file)

    ### Calculate and print precision and recall for multiple experiments ###
    srcDir = 'tests/prediction'
    threshold = 0.5
    output_file = 'predicted_results'
    print_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label',
                                output_file=output_file)

    ### Plot precision and recall ###
    input_file = 'tests/expt.csv'
    num_points = 100
    plot_PR(input_file, num_points, prediction_column_name='score', target_column_name='Label')

    ### Select data based on rules ###
    input_file = 'tests/combine_39.csv'
    threshold_list = [0.5, 0.5, 0.5, 0.5, 0.5]
    select_cmps(input_file, threshold_list, how='any', output_file=None, output_option='selected')

    ### Select data based on multiple rules ###
    srcDir = 'tests/prediction'
    input_file_target = 'tests/target.csv'
    threshold = 0.5
    output_file = 'predicted_results'
    select_cmps_by_multi_strategies(srcDir, input_file_target, threshold, output_file, output_option='selected')

    ### Calculate jaccard similarity between different experiments ###
    input_file = 'tests/combine_39.csv'
    threshold = 0.5
    get_jaccard_similarity(input_file, threshold=threshold)

    
    

