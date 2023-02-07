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

from analysis_prediction import *



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


