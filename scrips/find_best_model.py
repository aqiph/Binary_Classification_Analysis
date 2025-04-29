#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:40:55 2022

@author: guohan

"""

import os
import numpy as np
from tornado.web import MissingArgumentError


def best_model(log_file, how='train_hmean', **kwargs):
    """
    Helper function for find_best_models()
    Find the best model for a given log file.
    :param log_file: str, path of the log file.
    :param how: str, how to find the best model.
    :param train_AuPR_cutoff: float, train AUPR cutoff.
    :return: int, epoch number.
    """
    # get epoch number: select epoch based on AuPR of training
    if how in {'train_AuPR[0]', 'train_AuPR[1]', 'train_hmean'}:
        try:
            train_AuPR_cutoff = kwargs.pop('train_AuPR_cutoff')
        except KeyError:
            raise MissingArgumentError("Error: The 'train_AuPR_cutoff' is missing.")

        with open(log_file, 'r') as f:
            line = f.readline()
            while 'INFO: Epoch' not in line:
                line = f.readline()

            while 'done' not in line:
                line = line.strip().split()
                epoch_num = int(line[4])
                if how == 'train_AuPR[0]':
                    if float(line[11]) >= train_AuPR_cutoff:
                        break
                elif how == 'train_AuPR[1]':
                    if float(line[14]) >= train_AuPR_cutoff:
                        break
                else:
                    if float(line[17]) >= train_AuPR_cutoff:
                        break
                f.readline()
                line = f.readline()
            else:
                epoch_num = None

    # get epoch number: Unknown how
    else:
        raise ValueError('Error: Unknown how')

    return epoch_num


def find_best_models(savedModelsDir='saved_models', how='train_hmean', **kwargs):
    """
    Find the best models.
    :param savedModelsDir: str, path of the saved models.
    :param how: str, how to find the best model.
    :param train_AuPR_cutoff: float, train AUPR cutoff.
    """
    epoch_num_list = []

    # get the number of experiments
    dir_count = 0
    with os.scandir(savedModelsDir) as it:
        for entry in it:
            if entry.is_dir():
                dir_count += 1

    # get epoch number for best model
    for i in range(dir_count):
        files = os.listdir(os.path.join(savedModelsDir, f'Fold_{i}'))
        for file in files:
            full_path = os.path.join(savedModelsDir, f'Fold_{i}', file)
            if os.path.isfile(full_path) and file.endswith('.log'):
                epoch_num_list.append(best_model(full_path, how=how, **kwargs))

    print(f"The epoch numbers are {epoch_num_list}.")



if __name__ == '__main__':
    find_best_models(savedModelsDir='saved_models', how = 'train_hmean', train_AuPR_cutoff = 0.9)



