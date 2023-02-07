# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')        
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()
    if 'val' in config.DATASET.TEST_SET:
        logging.info("####### START VALIDATION #######")
        mean_IoU, IoU_array, pixel_acc, mean_acc, confusion_matrix = testval(config, 
                                                           test_dataset, 
                                                           testloader, 
                                                           model)
        ## Print & Save Confusion Matrix ##
        import csv
        logging.info("Confusion Matrix")
        logging.info(confusion_matrix)
        logging.info("Save Confusion Matrix csv File at /home/avg/AVG-NIA/HRNet/log/csv/confusion_matrix.csv")
        cfFilePath = "/home/avg/AVG-NIA/HRNet/log/csv/confusion_matrix.csv"

        confusionMatrix = []
        for i in range(0, len(confusion_matrix), 1):
            contents = []
            for j in range(0, len(confusion_matrix), 1):
                 contents.append(confusion_matrix[i][j])   
            confusionMatrix.append(contents)

        with open(cfFilePath, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerows(confusionMatrix)

        truePositive = np.diag(confusion_matrix)
        logging.info("True Positive")
        logging.info(truePositive)

        true_pos_false_pos = np.sum(confusion_matrix, axis=0).astype(float)
        true_pos_false_neg = np.sum(confusion_matrix, axis=1).astype(float)
        falsePositive = true_pos_false_pos - truePositive
        falseNegative = true_pos_false_neg - truePositive

        logging.info("False Positive")
        logging.info(falsePositive)

        logging.info("False Negative")
        logging.info(falseNegative)

        logging.info("Save tp_fp_fn csv File at /home/avg/AVG-NIA/HRNet/log/csv/tp_fp_fn.csv")
        tp_fp_fnFilePath = "/home/avg/AVG-NIA/HRNet/log/csv/tp_fp_fn.csv"
        tp_fp_fn = []
        tp_fp_fn.append(truePositive)
        tp_fp_fn.append(falsePositive)
        tp_fp_fn.append(falseNegative)

        precision = truePositive / (true_pos_false_pos)
        recall = truePositive / (true_pos_false_neg)
        print("precision")
        print(precision)
        print("recall")
        print(recall)

        with open(tp_fp_fnFilePath, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerows(tp_fp_fn)

        '''
        print("true_pos")
        print(true_pos)
        print("true_pos_false_pos")
        print(true_pos_false_pos)
        print("true_pos-false_pos")
        print(true_pos - true_pos_false_pos)
        print("true_pos_false_neg")
        print(true_pos_false_neg)
        print("precision")
        print(precision)
        print("recall")
        print(recall)
        '''

        msg = 'MeanIOU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
            pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)
    elif 'test' in config.DATASET.TEST_SET:
        logging.info("####### START INFERENCE #######")
        test(config, 
             test_dataset, 
             testloader, 
             model,
             sv_dir=final_output_dir)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int64((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
