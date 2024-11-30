#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:08:00 2024

@author: aqiph

"""

import os
import numpy as np
import pandas as pd
from functools import reduce



def combine_multiple_expts(srcDir, input_file_target=None, output_file=None):
    """
    Combine predicted results and true labels.
    :param srcDir: str, directory path of predicted results.
    :param input_file_target: str, path of the file containing true labels.
    :param output_file: str, name of the output file.
    :return: int, number of compounds.
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
            df = pd.DataFrame(df, columns=['ID', 'Cleaned_SMILES', 'score'])
            df.rename(columns={'score': expt}, inplace=True)
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
    df_merge = reduce(lambda left, right: pd.merge(left, right, how='left', on=['ID', 'Cleaned_SMILES']), dfs)
    columns = ['ID', 'Cleaned_SMILES'] + sorted(expts)
    df_merge = df_merge.reindex(columns, axis=1)
    num_expts = df_merge.shape[1] - 2

    # merge target labels
    if input_file_target is not None:
        df_target = pd.read_csv(input_file_target)
        df_target = pd.DataFrame(df_target, columns=['ID', 'Cleaned_SMILES', 'Label'])
        df_target.rename(columns={'Label': 'True_Label'}, inplace=True)
        df_merge = pd.merge(df_merge, df_target, how='left', on=['ID', 'Cleaned_SMILES'])

    # write to file
    # df_merge.sort_values(by=df_merge.columns.tolist()[2:], ascending=False, inplace=True)
    print('Number of rows in the output file:', df_merge.shape[0])
    df_merge.to_csv(f'{output_file}_{df_merge.shape[0]}.csv')

    return df_merge.shape[0], num_expts


def isPositive(scores, threshold_list, how='any'):
    """
    Helper function for select_compounds().
    Decide whether this compound is positive based on scores.
    :param scores: list of floats, predicted scores for one compound.
    :param threshold_list: list of floats, thresholds for different experiments.
    :param how: str, how to label, 'all', 'any', 'vote' and 'average'.
    :return: bool, whether this compound is positive.
    """
    assert len(scores) == len(threshold_list), 'Error: Number of experiments should be same as number of thresholds'

    scores = [float(s) for s in scores]
    scores = np.array(scores)
    threshold_list = np.array(threshold_list)
    labels = scores >= threshold_list

    # apply different strategies
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


def jaccard_similarity(df, threshold_list, expt1, expt2):
    """
    Helper function for get_jaccard_similarity().
    Calculate jaccard similarity (intersection/union) for expt1 and expt2 of df.
    :param df: DataFrame object, pd.DataFrame containing expt1 and expt2.
    :param threshold_list: list of floats, thresholds for different experiments.
    :param expt1: str, column name for expt1.
    :param expt2: str, column name for expt2.
    :return: float, the calculated jaccard similarity.
    """
    # calculate intersection
    df_intersect = df.loc[df.apply(lambda row: isPositive([row[expt1], row[expt2]], threshold_list, how='all'), axis=1)]
    intersection = df_intersect.shape[0]

    # calculate union
    df_union = df.loc[df.apply(lambda row: isPositive([row[expt1], row[expt2]], threshold_list, how='any'), axis=1)]
    union = df_union.shape[0]

    # calculate jaccard similarity
    if union <= 0.0:
        return 0
    similarity = float(intersection) / float(union)
    return similarity