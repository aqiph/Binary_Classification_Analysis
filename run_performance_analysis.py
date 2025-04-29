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

from main import print_PR_for_multiple_expts, plot_PR



if __name__ == '__main__':

    ### Calculate precision and recall for multiple experiments ###
    srcDir = 'tests/test_performance/prediction'
    threshold = 0.5
    output_file = None
    print_PR_for_multiple_expts(srcDir, threshold, prediction_column_name='score', target_column_name='Label', output_file=output_file)

    ### Plot precision and recall ###
    for i in range(5):
        input_file = f'tests/test_performance/prediction/prediction_model{i}.csv'
        num_points = 100
        plot_PR(input_file, num_points, prediction_column_name='score', target_column_name='Label')
