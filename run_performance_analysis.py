#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:40:55 2022

@author: guohan

"""

import sys

path_list = sys.path
module_path = '/Users/guohan/Documents/Codes/Binary_Classification_Analysis'
if module_path not in sys.path:
    sys.path.append(module_path)
    print('Add module path')

from main import combine_multiple_expts, print_PR_for_multiple_expts, plot_PR, select_cmps_by_multi_strategies, get_jaccard_similarity



if __name__ == '__main__':

    ### Calculate precision and recall for multiple experiments ###
    srcDir = 'prediction'
    threshold = 0.5
    output_file = None
    print_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label', output_file=output_file)

    ### Plot precision and recall ###
    for i in range(5):
        input_file = f'prediction/Fold_{i}.csv'
        num_points = 100
        plot_PR(input_file, num_points, prediction_column_name='score', target_column_name='Label')
