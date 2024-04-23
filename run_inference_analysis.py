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

from main import combine_multiple_expts, select_cmps, select_cmps_by_multi_strategies



if __name__ == '__main__':
    ### Select data based on rules ###
    srcDir = 'tests/prediction'
    input_file_target = 'tests/target.csv'
    threshold = 0.5
    output_file = 'predicted_results'
    select_cmps_by_multi_strategies(srcDir, input_file_target, threshold, output_file, output_option='selected')

