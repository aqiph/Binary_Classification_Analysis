#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:43:01 2021

@author: guohan

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

def combine_multiple_expt(srcDir, output_file = None):
    """
    combine predicted results 
    :para srcDir: str, the directory to store score files
    :para output_file: str, output file name
    """
    num = 0
    dfs = []
    expts = []
    
    # read df
    files = os.listdir(srcDir)
    for file in files:
        if os.path.splitext(file)[1] != '.csv':
            continue
        
        try:
            expt = os.path.splitext(file)[0]
            file = os.path.join(srcDir, file)
            df = pd.read_csv(file)
            df = pd.DataFrame(df, columns = ['ID', 'Cleaned_SMILES', 'score'])
            df.rename(columns = {'score': expt}, inplace = True)

            print('{} is read'.format(expt))
            print('number of rows ', df.shape[0])
            dfs.append(df)
            expts.append(expt)
            num += 1
        except:
            pass
    
    # output file
    folder, basename = os.path.split(os.path.abspath(srcDir))
    if output_file is None:
        output_file = os.path.splitext(basename)[0] + '_combine.csv'
    output_file = os.path.join(folder, output_file)
    
    # merge multiple dataframe
    df_merge = reduce(lambda left, right: pd.merge(left, right, how = 'outer', on = ['ID', 'Cleaned_SMILES']), dfs)
    columns = ['ID', 'Cleaned_SMILES'] + sorted(expts)
    df_merge = df_merge.reindex(columns, axis = 1)
    
    # write to file
    print('Number of rows in the output file:', df_merge.shape[0])
    df_merge.to_csv(output_file)


def combine_prediction_target(input_file, target_file, output_file = None):
    """
    combine predicted results and target value
    :para input_file: str, the name of the predicted file
    :para target_file: str, the name of the target file
    :para output_file: str, output file name
    """
    # read files
    df_prediction = pd.read_csv(input_file, index_col = 0)
    df_target = pd.read_csv(target_file)
    
    # output file
    folder, basename = os.path.split(os.path.abspath(input_file))
    if output_file is None:
        output_file = os.path.splitext(basename)[0] + '_target.csv'
    output_file = os.path.join(folder, output_file)
    
    # change df_target column names
    df_target = pd.DataFrame(df_target, columns = ['ID', 'Cleaned_SMILES', 'Label'])
    df_target.rename(columns = {'ID': 'Training_set_ID', 'Label': 'True_Label'}, inplace = True)
    
    # merge two dataframe
    df = pd.merge(df_prediction, df_target, how = 'left', on = ['Cleaned_SMILES'])
    
    # sort on scores
    df.sort_values(by = df.columns.tolist()[2:], ascending = False, inplace = True)
    
    # write to file
    print('Number of rows in the output file:', df.shape[0])
    df.to_csv(output_file)


### Calculate and plot precision and recall ###

def cal_precision_recall(predicted_labels, true_labels):
    """
    calculate precision and recall
    :para predicted_labels: list of predicted labels, i.e., 1 or 0
    :para true_labels: list of true labels, i.e., 1, 0, Nan
    :return: precision_1, recall_1, precision_0, recall_0
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


def plot_precision_recall(input_file, num_points, prediction_column_name='score', target_column_name='Label'):
    """
    plot precision recall curves
    :para input_file: str, the name of the file with predicted score and target label
    :para threshold: float, threshold to label predictions
    :para prediction_column_name: str, the name of the column with predicted scores
    :para target_column_name: str, the name of the column with target labels
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
        precision_1, recall_1, precision_0, recall_0 = cal_precision_recall(predicted_labels, true_labels)

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

def get_selected(input_file, threshold_list, how = 'any', output_file = None):
    """
    return a subset of selected compounds, 
    satisfy that the score of the nth experiment >= the nth threshold
    :para input_file: str, input file name of the predicted score
    :para threshold_list: list of floats, the threshold for experiments
    :para how: str, how to label, 'all', 'any' and 'vote'
    :para output_file: str, output file name
    """
    # read files
    df_prediction = pd.read_csv(input_file, index_col = 0)
    COLUMNS = df_prediction.columns.tolist()
    expts = df_prediction.columns.tolist()[2:-2]
    assert len(expts) == len(threshold_list), 'Error: Number of experiments should be same as number of thresholds'
    
    # output file
    folder, basename = os.path.split(os.path.abspath(input_file))    
    if output_file is None:
        output_file = basename
    output_file = os.path.join(folder, os.path.splitext(output_file)[0] + '_' + how + '_')
    
    # get selected compounds
    df_prediction['Selected'] = df_prediction.apply(lambda row: isPositive([row[expt] for expt in expts], threshold_list, how), axis=1)
    df = pd.DataFrame(df_prediction[df_prediction['Selected']], columns = COLUMNS)

    # calculate precision and recall
    PR_dict = {'Precision_1': [], 'Recall_1': [], 'Precision_0': [], 'Recall_0': []}

    for i, expt in enumerate(expts):
        df_prediction[expt + '_Selected'] = df_prediction[expt].apply(lambda p: int(p >= threshold_list[i]))
        precision_1, recall_1, precision_0, recall_0 = cal_precision_recall(df_prediction[expt + '_Selected'].tolist(),df_prediction['True_Label'].tolist())
        PR_dict['Precision_1'].append(precision_1)
        PR_dict['Recall_1'].append(recall_1)
        PR_dict['Precision_0'].append(precision_0)
        PR_dict['Recall_0'].append(recall_0)

    precision_1, recall_1, precision_0, recall_0 = cal_precision_recall(df_prediction['Selected'].tolist(),df_prediction['True_Label'].tolist())
    print(precision_1, recall_1, precision_0, recall_0)
    PR_dict['Precision_1'].append(precision_1)
    PR_dict['Recall_1'].append(recall_1)
    PR_dict['Precision_0'].append(precision_0)
    PR_dict['Recall_0'].append(recall_0)
    df_PR = pd.DataFrame(PR_dict, index=expts + ['Combination'])
    df_PR = df_PR.applymap(lambda x: '{:2.4f}'.format(x))

    # write to file
    print('Number of rows:', df.shape[0])
    df.to_csv(output_file + str(df.shape[0]) + '.csv')
    df_PR.to_csv(output_file + 'PR.csv')


def isPositive(scores, threshold_list, how = 'any'):
    """
    helper function for get_selected()
    decide whether this compound is positive based on scores
    :para scores: list of float, the predicted score for experiment
    :para threshold_list: list of float, the threshold for experiment
    :para how: str, how to label, 'all', 'any', 'vote' and 'average'
    :return: bool, wheter this compound is positive
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


### Calculate jaccard similarity between different experiments

def get_jaccard_similarity(input_file, threshold):
    """
    calculate Jaccard similarity between two libraries
    :para input_file: str, input file name of the predicted score
    :para threshold: float, the threshold for data
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


def jaccard_similarity(df, threshold_list, row1, row2):
    """
    helper function for get_jaccard_similarity()
    calculate jaccard similarity (intersection/union) for row1 and row2 of df
    :para df: DataFrame object, the table containing row1 and row2
    :para threshold_list: list of float, the threshold for experiment
    :para row1: str, the row name of row1
    :para row2: str, the row name of row2
    :return: float, the calculated jaccard similarity
    """
    # calculate intersection
    df_intersect = df.loc[df.apply(lambda row: isPositive([row[row1], row[row2]], threshold_list, how = 'all'), axis = 1)]
    intersection = df_intersect.shape[0]
    
    # calculate union
    df_union = df.loc[df.apply(lambda row: isPositive([row[row1], row[row2]], threshold_list, how = 'any'), axis = 1)]
    union = df_union.shape[0]
    
    # calculate jaccard similarity
    if union <= 0.0:
        return 0
    similarity = float(intersection)/float(union)
    return similarity



if __name__ == '__main__':

    ### Combine results ###
    srcDir = 'test/prediction'
    output_file = 'combination.csv'
    combine_multiple_expt(srcDir, output_file)

    input_file = 'test/combination.csv'
    target_file = 'test/target.csv'
    output_file = 'combination_target.csv'
    combine_prediction_target(input_file, target_file, output_file)

    ### Select data based on rules ###
    threshold = 0.5
    threshold_list = [threshold for _ in range(3)]
    input_file = 'test/combination_target.csv'
    output_file = 'predicted'

    get_selected(input_file, threshold_list = threshold_list, how = 'any', output_file = output_file)
    get_selected(input_file, threshold_list = threshold_list, how = 'vote', output_file = output_file)
    get_selected(input_file, threshold_list = threshold_list, how = 'all', output_file = output_file)
    get_selected(input_file, threshold_list = threshold_list, how = 'average', output_file = output_file)

    get_jaccard_similarity(input_file, threshold = threshold)

    ### Calculate and plot precision and recall ###
    input_file = 'test/Fold_0.csv'
    num_points = 100
    plot_precision_recall(input_file, num_points, prediction_column_name='score', target_column_name='Label')

    
    

