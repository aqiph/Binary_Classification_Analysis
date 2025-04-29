#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:40:55 2022

@author: guohan

"""

import os
import pandas as pd



def get_model(epoch_num, modelDir):
    """
    Helper function for write_inference_input().
    Get model file name.
    :param epoch_num: int, epoch number.
    :param modelDir: str, path of the model directory.
    :return: str, model full path.
    """
    files = os.listdir(modelDir)
    for file in files:
        full_path = os.path.join(modelDir, file)
        if os.path.isfile(full_path) and file.startswith(f'epoch_{epoch_num}_loss'):
            return full_path
    else:
        raise FileNotFoundError(f'Error: The file does not exist in {modelDir}.')


def write_inference_input(inference_input_file, epoch_num_list, num_expts=5):
    """
    Write inference input file.
    :param inference_input_file: str, path of the inference input file.
    :param epoch_num_list: list of ints, list of epoch numbers.
    :param num_expts: int, number of experiments.
    :return: None.
    """
    content = ("#!/bin/sh\n#$ -S /bin/bash\n#$ -cwd\n\n"
               "#$ -N inference\n\n### gpus\n\n"
               "#$ -l ngpus=4\n#$ -pe p100 1\n#$ -q p100\n\n"
               "######\nsource /home/cudasoft/bin/startcuda.sh\n\n"
               "conda activate zoo_py36\n\n")

    # loop over experiments
    if inference_input_file is None:
        flag_validation = True
    else:
        flag_validation = False
    assert len(epoch_num_list) == num_expts, 'Error: The number of epochs is not correct.'
    for i in range(num_expts):
        if flag_validation:
            inference_input_file = f"split_test_{i}.csv"
        modelDir = f"saved_models/Fold_{i}/models"
        model_path = get_model(epoch_num_list[i], modelDir)
        content_expt = (f"python /data01/hanguo/Code/Model_Zoo/inference.py \\\n"
                        f"--saved_model '{model_path}' \\\n"
                        f"--atom_dict_file '/data01/hanguo/Code/Model_Zoo/data/atom_dict.pkl' \\\n"
                        f"--dataset_path 'data' \\\n"
                        f"--dataset '{inference_input_file}' \\\n"
                        f"--sort_result \\\n"
                        f"--output_file 'prediction_model{i}.csv'\n\n")
        content += content_expt

    content += ("conda deactivate\n\n"
                "source /home/cudasoft/bin/end_cuda.sh\n"
                "######\n\necho \"All done!\"")

    # write to run_inference.py
    if flag_validation:
        jobid = 'validation'
    else:
        jobid = os.path.splitext(inference_input_file)[0].split('_')[2]
    with open(f'run_inference_{jobid}.sh', 'w') as f:
        f.write(content)



if __name__ == '__main__':
    # inference_input_file = 'ClpP_MLM_public20231204-inhouse20241125-trainset_cutoff30_7165.csv'
    inference_input_file = None
    epoch_num_list = [1, 2, 3, 4, 5]
    write_inference_input(inference_input_file, epoch_num_list)