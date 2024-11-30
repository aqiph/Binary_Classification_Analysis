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

from main import print_PR_for_multiple_expts, plot_PR, select_compounds, select_compounds_by_multi_strategies, get_jaccard_similarity



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


