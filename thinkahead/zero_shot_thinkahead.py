# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import os
import time
import argparse
import logging
import warnings
import json
import polars as pl
import numpy as np
from utils import setup_logging, train_test_map, get_map_length, union_preload_process, find_count_two_time_series
from AE_config import GB, meta_file
warnings.filterwarnings('ignore')

res_path = '../result/zero_shot/'

def zero_shot_exp(args):
    logger = logging.getLogger('file_logger')
    logger.info(f'Zero-shot algorithm param: {vars(args)}')
    logger.info(f'OSS bandwidth: {args.bandwidth}MB/s')

    train_files_map, test_files_map = train_test_map(args)
    get_map_length(train_files_map, test_files_map, logger)
    zero_shot_preload(args, test_files_map, logger)

def zero_shot_preload(args, test_files_map, logger):

    df = pl.read_csv(f'../{meta_file}')
    total_test_version = len(test_files_map)
    avg_hit_rate_res, total_hit_rate_res = {}, {}
    total_wait_latency_res = {} 
    total_accuracy_res = {}
    zero_shot_preload_series = {}
    preload_series_save_path = os.path.join(res_path, f'thinkahead_{args.alpha_num}_{args.beta_num}_preload_series.json')
    logger.info('Calculate the preload series for zero-shot test traces.')
    if os.path.exists(preload_series_save_path):
        with open(preload_series_save_path, 'r') as f:
            zero_shot_preload_series = json.load(f)
        logger.info(f'Load preload series from {preload_series_save_path}')
    else:
        logger.warning(f'No preload series found for {preload_series_save_path}, will calculate on-the-fly.')
    for i, (test_version, test_files) in enumerate(test_files_map.items()):
        logger.info(f'----------Test version: {test_version} [{i+1}/{total_test_version}]----------')
        avg_hit_rate = 0
        file_num = len(test_files)
        valid_num = 0
        for j, test_file in enumerate(test_files):
            logger.info(f'----------Test file: {test_file} [{j+1}/{file_num}]----------')
            if not os.path.exists(test_file):
                logger.info(f'Test file {test_file} not exists, skip this file.')
                continue
            if test_version not in zero_shot_preload_series:
                preload_series = find_zero_shot_series(args, df, test_file, logger)
                zero_shot_preload_series[test_version] = preload_series
            else:
                preload_series = zero_shot_preload_series[test_version]
            res = union_preload_process(args.bandwidth, df, preload_series, test_file, logger, False)
            if res is not None:
                hit_rate, wait_latency, accuracy = res
                avg_hit_rate += hit_rate
                key_name = f'{test_file}'
                total_hit_rate_res[key_name] = hit_rate
                total_wait_latency_res[key_name] = wait_latency
                total_accuracy_res[key_name] = accuracy
                valid_num += 1
                logger.info(f'Preload hit rate: {hit_rate:.4f}, largest wait latency: {max(wait_latency)}, accuracy: {accuracy:.4f}')
        if valid_num > 0:
            avg_hit_rate /= valid_num
            avg_hit_rate_res[test_version] = avg_hit_rate
            logger.info(f'Average hit rate for version {test_version}: {avg_hit_rate:.4f}')
        else:
            logger.warning(f'No valid hit rate for {test_version}, skip this version.')

    if not os.path.exists(preload_series_save_path):
        with open(preload_series_save_path, 'w') as f:
            json.dump(zero_shot_preload_series, f)
    avg_hit_rate_save_path = os.path.join(res_path, f'thinkahead_avg_zero_{args.alpha_num}_{args.beta_num}_hit_rate_{args.bandwidth}MB.json')
    all_hit_rate_save_path = os.path.join(res_path, f'thinkahead_all_zero_{args.alpha_num}_{args.beta_num}_hit_rate_{args.bandwidth}MB.json')
    all_wait_latency_save_path = os.path.join(res_path, f'thinkahead_all_zero_{args.alpha_num}_{args.beta_num}_wait_latency_{args.bandwidth}MB.npy')
    all_accuracy_save_path = os.path.join(res_path, f'thinkahead_all_zero_{args.alpha_num}_{args.beta_num}_accuracy_{args.bandwidth}MB.json')
    with open(avg_hit_rate_save_path, 'w') as f:
        json.dump(avg_hit_rate_res, f)
    with open(all_hit_rate_save_path, 'w') as f:
        json.dump(total_hit_rate_res, f)
    with open(all_accuracy_save_path, 'w') as f:
        json.dump(total_accuracy_res, f)
    np.save(all_wait_latency_save_path, np.array(total_wait_latency_res))
    avg_hit_rate = np.mean(list(avg_hit_rate_res.values()))
    p50_latency = np.percentile([lat for lat_list in total_wait_latency_res.values() for lat in lat_list], 50)
    print(f'Zero-shot ThinkAhead Average Hit Rate: {avg_hit_rate:.4f}, P50 Wait Latency: {p50_latency} us.')
    print(f'Save all and average hit rate results to {os.path.abspath(avg_hit_rate_save_path)}:1 and {os.path.abspath(all_hit_rate_save_path)}:1, save accuracy results to {os.path.abspath(all_accuracy_save_path)}:1, save latency results to {os.path.abspath(all_wait_latency_save_path)}:1 finished!')

def find_zero_shot_series(args, df, test_file, logger):
    
    preload_series = []
    test_volume = test_file.split('/')[-1].split('.')[0]
    test_meta = df.filter(pl.col('volume_id') == int(test_volume))
    same_meta_df = df.filter((pl.col('image_type') == test_meta['image_type']) & (pl.col('device_size') == test_meta['device_size']) & (pl.col('extra_info') == test_meta['extra_info']) & (pl.col('meta_info') == test_meta['meta_info']))
    same_user_meta_df = same_meta_df.filter(pl.col('user_id') == test_meta['user_id'])
    train_files = []
    if len(same_user_meta_df) > 0:
        for row in same_user_meta_df.iter_rows(named=True):
            device_size = row['device_size'] // GB
            file_path = f'../trace/it{row["image_type"]}/{row["image_id"]}/{device_size}GB/{row["volume_id"]}.npy'
            if os.path.exists(file_path):
                train_files.append(file_path)
        logger.info(f'Find {len(train_files)} train files for {test_file}.')
    else:
        for row in same_meta_df.iter_rows(named=True):
            device_size = row['device_size'] // GB
            file_path = f'../trace/it{row["image_type"]}/{row["image_id"]}/{device_size}GB/{row["volume_id"]}.npy'
            if os.path.exists(file_path):
                train_files.append(file_path)
        logger.warning(f'No same user meta for {test_file}.')
    preload_series = find_count_two_time_series(args.alpha_num, args.beta_num, train_files)
    return preload_series

def main():
    parser = argparse.ArgumentParser(description='Zero Shot Algorithm')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode')
    parser.add_argument('--bandwidth', '-bw', type=int, required=True, choices=[2, 4, 6, 8, 10, 30, 50, 70, 90, 150], help='Bandwidth in MB')
    parser.add_argument('--alpha_num', '-an', type=float, default=0.1, help='weight of the number of accesses')
    parser.add_argument('--beta_num', '-bn', type=float, default=0.2, help='weight of the access time')
    args = parser.parse_args()
    args.data_type = 'zero-shot'
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    global res_path
    if args.debug:
        logfile = 'zero_shot_debug.log'
    else:
        logfile = f'thinahead_{args.bandwidth}MB_{cur_time}.log'
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    setup_logging(log_name=logfile, log_path='../log/zero_shot')
    zero_shot_exp(args)

    # Example command to run: python zero_shot_algo.py -bw 2

if __name__ == "__main__":
    main()
