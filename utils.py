# -*- encoding: utf-8 -*-

from AE_config import image_types, GB, BLOCK_SIZE, START_TIME
import os
import logging
import numpy as np
import re
import json
import polars as pl
import copy
import time
import sys
from collections import defaultdict
from tqdm import tqdm

pattern = re.compile(r'/it(\d+)')

def setup_logging(log_name, log_path='../log'):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = f'{log_path}/{log_name}'
    logging.basicConfig(
        filename=logfile,
        filemode='w',
        level=logging.INFO,
        format='[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        # datefmt='%Y-%m-%d %H:%M:%S'
    )
    print(f'Log file path: {os.path.abspath(logfile)}')

def train_test_map(args):

    if args.data_type == 'few-shot':
        train_map_path = '../data/eval_files/few_shot/train_files_map.json'
        test_map_path = '../data/eval_files/few_shot/test_files_map.json'
    elif args.data_type == 'zero-shot':
        train_map_path = '../data/eval_files/zero_shot/train_files_map.json'
        test_map_path = '../data/eval_files/zero_shot/test_files_map.json'
    else:
        assert False, f'Invalid data type: {args.data_type}'
    if os.path.exists(train_map_path) and os.path.exists(test_map_path):
        with open(train_map_path, 'r') as f:
            train_files_map = json.load(f)
        with open(test_map_path, 'r') as f:
            test_files_map = json.load(f)
        print(f'Load train files map and test files map from {train_map_path} and {test_map_path}')
    else:
        train_files_map, test_files_map = get_data_files(args)
        train_files_map = dict(sorted(train_files_map.items(), key=lambda x: x[0]))
        test_files_map = dict(sorted(test_files_map.items(), key=lambda x: x[0]))
        with open(train_map_path, 'w') as f:
            json.dump(train_files_map, f, indent=4)
        with open(test_map_path, 'w') as f:
            json.dump(test_files_map, f, indent=4)
    return train_files_map, test_files_map

def get_data_files(args):
    base_folder = '../data/eval_files'
    train_files_map, test_files_map = {}, {}
    train_num, test_num = 0, 0
    if args.data_type == 'few-shot':
        log_folder = os.path.join(base_folder, 'few_shot')
    elif args.data_type == 'zero-shot':
        log_folder = os.path.join(base_folder, 'zero_shot')
    else:
        assert False, f'Invalid data type: {args.data_type}'
    train_log_files = [file for file in os.listdir(log_folder) if file.endswith('_trainfile.txt')]
    test_log_files = [file for file in os.listdir(log_folder) if file.endswith('_testfile.txt')]
    for file in train_log_files:
        image_type = file.split('_')[0][2:]
        with open(os.path.join(log_folder, file), 'r') as f:
            for line in f:
                line = line.strip()
                image_id = line.split('/')[-3]
                device_size = line.split('/')[-2]
                key_id = f'{image_type}_{image_id}_{device_size}'
                if key_id not in train_files_map:
                    train_files_map[key_id] = []
                train_files_map[key_id].append(line)
                train_num += 1
    for file in test_log_files:
        image_type = file.split('_')[0][2:]
        with open(os.path.join(log_folder, file), 'r') as f:
            for line in f:
                line = line.strip()
                image_id = line.split('/')[-3]
                device_size = line.split('/')[-2]
                key_id = f'{image_type}_{image_id}_{device_size}'
                if key_id not in test_files_map:
                    test_files_map[key_id] = []
                test_files_map[key_id].append(line)
                test_num += 1
    print(f'Get {train_num} train files, {test_num} test files, test rate: {train_num / (train_num+test_num) * 100:.2f}%')
    return train_files_map, test_files_map

def get_map_length(train_file_map, test_file_map, logger=None):
    train_all_num, test_all_num = {}, {}
    total_num = 0
    for key, files in train_file_map.items():
        image_type = key.split('_')[0]
        if image_type not in train_all_num:
            train_all_num[image_type] = 0
        train_all_num[image_type] += len(files)
        total_num += len(files)
    for key, files in test_file_map.items():
        image_type = key.split('_')[0]
        if image_type not in test_all_num:
            test_all_num[image_type] = 0
        test_all_num[image_type] += len(files)
        total_num += len(files)
    for image_type, num in train_all_num.items():
        if logger:
            logger.info(f'Image type: {image_type}, train file num: {num}, test file num: {test_all_num[image_type]}')
        else:
            print(f'Image type: {image_type}, train file num: {num}, test file num: {test_all_num[image_type]}')
    return total_num


def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def get_public_user_files():
    all_records = []
    public_records, user_records = [], []
    for image_type in image_types:
        train_file = f'../data/eval_files/few_shot/it{image_type}_rt1200_trainfile.txt'
        test_file = f'../data/eval_files/few_shot/it{image_type}_rt1200_testfile.txt'
        with open(train_file, 'r') as f:
            train_files = f.readlines()
            for train_file in train_files:
                train_file = train_file.strip()
                all_records.append(train_file)
                if '/m-' in train_file:
                    user_records.append(train_file)
                else:
                    public_records.append(train_file)
        with open(test_file, 'r') as f:
            test_files = f.readlines()
            for test_file in test_files:
                test_file = test_file.strip()
                all_records.append(test_file)
                if '/m-' in test_file:
                    user_records.append(test_file)
                else:
                    public_records.append(test_file)
    print(len(all_records), "files in total")
    return all_records, public_records, user_records

def collect_train_info(df, same_version_train_files):
    block_access_num, block_access_all_time, block_access_min_time = {}, {}, {}
    start_interval = []
    max_access_num, max_access_time = 0, 0
    for i in tqdm(range(len(same_version_train_files))):
        train_file = same_version_train_files[i]
        train_volume_id = train_file.split('/')[-1].split('.')[0]
        match = pattern.search(train_file)
        assert match, f'No match for {train_file}'
        train_image_type = match.group(1)
        train_image_id = train_file.split('/')[-3]
        train_device_size = int(train_file.split('/')[-2][:-2]) * GB
        volume_df = df.filter((df['volume_id'] == train_volume_id) & (df['image_type'] == train_image_type) & (df['image_id'] == train_image_id) & (df['device_size'] == train_device_size))
        record = np.load(train_file)
        record = record[record[:, 0]==1]
        if len(record) == 0:
            continue
        if not np.all(record[:, 3][:-1] <= record[:, 3][1:]):
            record = record[record[:, 3].argsort()] 
        start_time = record[0, 3]
        if len(volume_df) > 0:
            gmt_create = volume_df['gmt_create'].to_numpy()[0]
            create_unix_ts = int(time.mktime(time.strptime(gmt_create, '%Y-%m-%d %H:%M:%S'))) * 1_000_000  
            assert start_time >= create_unix_ts, f'Start time {start_time} is earlier than create time {create_unix_ts} for {train_file}'
            start_interval.append(start_time - create_unix_ts)
        valid_masks = record[:, 3] - start_time <= START_TIME * 1_000_000
        valid_record = record[valid_masks]
        start_blocks = valid_record[:, 1] // BLOCK_SIZE
        end_blocks = (valid_record[:, 1] + valid_record[:, 2]) // BLOCK_SIZE
        for start_block, end_block, ts in zip(start_blocks, end_blocks, valid_record[:, 3]):
            start_block, end_block = int(start_block), int(end_block)
            relative_time = np.int32(ts - start_time)
            if start_block == end_block:
                block_access_num[start_block] = block_access_num.get(start_block, 0) + 1
                if start_block not in block_access_all_time:
                    block_access_all_time[start_block] = []
                block_access_all_time[start_block].append(relative_time)
                if start_block not in block_access_min_time:
                    block_access_min_time[start_block] = relative_time
                else:
                    block_access_min_time[start_block] = min(block_access_min_time[start_block], relative_time)
                max_access_num = max(max_access_num, block_access_num[start_block])
                max_access_time = max(max_access_time, relative_time)
            else:
                block_access_num[start_block] = block_access_num.get(start_block, 0) + 1
                block_access_num[end_block] = block_access_num.get(end_block, 0) + 1
                if start_block not in block_access_all_time:
                    block_access_all_time[start_block] = []
                if end_block not in block_access_all_time:
                    block_access_all_time[end_block] = []
                block_access_all_time[start_block].append(relative_time)
                block_access_all_time[end_block].append(relative_time)
                if start_block not in block_access_min_time:
                    block_access_min_time[start_block] = relative_time
                else:
                    block_access_min_time[start_block] = min(block_access_min_time[start_block], relative_time)
                if end_block not in block_access_min_time:
                    block_access_min_time[end_block] = relative_time
                else:
                    block_access_min_time[end_block] = min(block_access_min_time[end_block], relative_time)
                max_access_num = max(max_access_num, block_access_num[start_block], block_access_num[end_block])
                max_access_time = max(max_access_time, relative_time)
    all_info = {}
    all_info['access_num'] = block_access_num
    all_info['access_avg_time'] = {block: np.mean(access_time) for block, access_time in block_access_all_time.items()}
    all_info['access_min_time'] = block_access_min_time
    all_info['max_access_num'] = max_access_num
    all_info['max_access_time'] = max_access_time
    all_info['avg_start_interval'] = np.mean(start_interval) if start_interval else -1
    return all_info

def find_count_two_time_series_history(alpha_num, beta_num, train_info):
    block_access_num = train_info['access_num']
    block_access_avg_time = train_info['access_avg_time']
    block_access_min_time = train_info['access_min_time']
    max_access_num = train_info['max_access_num']
    max_access_time = train_info['max_access_time']
    block_score = {}
    for k, v in block_access_num.items():
        normal_access_num = v / max_access_num
        normal_mean_access_time = (max_access_time - block_access_avg_time[k]) / max_access_time
        normal_min_access_time = (max_access_time - block_access_min_time[k]) / max_access_time
        score = alpha_num * normal_access_num + beta_num * normal_mean_access_time + (1-alpha_num-beta_num) * normal_min_access_time
        block_score[k] = score
    block_score = dict(sorted(block_score.items(), key=lambda x: x[1], reverse=True))
    return list(block_score.keys())


def union_preload_process(bandwidth: float, df: pl.DataFrame, init_preload_series: list, test_file: str, logger=None, print_flag=False):

    test_record = np.load(test_file)
    test_record = test_record[test_record[:, 0] == 1]
    if not np.all(test_record[:, 3][:-1] <= test_record[:, 3][1:]):
        test_record = test_record[test_record[:, 3].argsort()]
    if len(test_record) == 0:
        if logger:
            logger.info(f'No read I/O in {test_file}')
        else:
            if print_flag:
                print(f'No read I/O in {test_file}')
        return
    if test_record[0][1] != 0:
        if logger:
            logger.info(f'The first read I/O of {test_file} is not 0: {test_record[0]}!')
        else:
            if print_flag:
                print(f'The first read I/O of {test_file} is not 0: {test_record[0]}!')
        return
    if logger:
        logger.info(f'Initial preload series length: {len(init_preload_series)}')
    else:
        if print_flag:
            print(f'Initial preload series length: {len(init_preload_series)}')
    
    preload_series = copy.deepcopy(init_preload_series)
    test_volume_id = test_file.split('/')[-1].split('.')[0]
    match = pattern.search(test_file)
    assert match, f'No match for {test_file}'
    test_image_type = match.group(1)
    test_image_id = test_file.split('/')[-3]
    test_device_size = int(test_file.split('/')[-2][:-2]) * GB
    volume_df = df.filter((df['volume_id'] == test_volume_id) & (df['image_type'] == test_image_type) & (df['image_id'] == test_image_id) & (df['device_size'] == test_device_size))
    if len(volume_df) == 0:
        if logger:
            logger.info(f'volume_id: {test_volume_id} not found in {df}')
        else:
            if print_flag:
                print(f'volume_id: {test_volume_id} not found in {df}')
        return
    gmt_create = volume_df['gmt_create'].to_numpy()[0]
    create_unix_ts = int(time.mktime(time.strptime(gmt_create, '%Y-%m-%d %H:%M:%S'))) * 1_000_000
        
    hit_rate, total_wait_latency, accuracy = preload_func(bandwidth, test_file, test_record, create_unix_ts, preload_series)
    return hit_rate, total_wait_latency, accuracy

def preload_func(bandwidth, test_file, test_record, create_unix_ts, preload_series):

    block_time = BLOCK_SIZE / (bandwidth * 1024 * 1024)
    block_time_us = int(block_time * 1_000_000)
    accuracy = 0
    final_preload_blocks = []
    total_wait_latency = []
    CHECK_NUM = min(3000, len(test_record))
    hit_num = 0
    hit_blocks = set() 
    miss_block_num = 0
    miss_pulled_blocks = []
    preload_blocks = []
    not_hit_block = {}
    for i in range(CHECK_NUM):
        access_blocks = [test_record[i][1] // BLOCK_SIZE]
        lba_end_block = (test_record[i][1] + test_record[i][2]) // BLOCK_SIZE
        if lba_end_block not in access_blocks: 
            access_blocks.append(lba_end_block)
        interval = test_record[i][3] - create_unix_ts
        assert interval > 0, f'Preload time {interval} is not positive for {test_file}'
        not_hit_block = dict(sorted(not_hit_block.items(), key=lambda x: x[1]))
        delete = []
        for j, (k, v) in enumerate(not_hit_block.items()):
            if v + block_time_us < test_record[i][3]:
                miss_pulled_blocks.append(k)
                delete.append(k)
            else:
                break
        for k in delete:
            del not_hit_block[k]
        if len(not_hit_block) == 0:
            pulling_time = 0 
        else:
            pulling_time = max(0, test_record[i][3] - list(not_hit_block.values())[0])
        pull_block_num = int((interval - len(miss_pulled_blocks) * block_time_us - pulling_time) // block_time_us)
        assert pull_block_num >= 0, f'Pull block num {pull_block_num} is negative for {test_file} at index {i}'
        preload_blocks = preload_series[:pull_block_num]
        preload_blocks.extend(miss_pulled_blocks)
        if i == CHECK_NUM - 1:
            final_preload_blocks = copy.deepcopy(preload_blocks)
        if len(access_blocks) == 1:
            access_block = access_blocks[0]
            if access_block in preload_blocks:  
                hit_num += 1
                total_wait_latency.append(0)
                hit_blocks.add(access_block)
            else:
                if access_block in not_hit_block:  
                    total_wait_latency.append(int(not_hit_block[access_block] + block_time_us - test_record[i][3]))
                    continue
                miss_block_num += 1
                if len(not_hit_block) == 0:
                    if int(interval // block_time_us) >= len(preload_series): 
                        not_hit_block[access_block] = test_record[i][3]
                        total_wait_latency.append(block_time_us)
                    else:
                        cur_preload_block = preload_series[int(interval // block_time_us)]
                        cur_remain_t = block_time_us - interval % block_time_us
                        if cur_preload_block == access_block:  
                            total_wait_latency.append(int(cur_remain_t))
                            continue
                        not_hit_block[access_block] = test_record[i][3] + cur_remain_t
                        total_wait_latency.append(int(cur_remain_t + block_time_us))  
                else:
                    last_miss_block_time = list(not_hit_block.values())[-1]
                    not_hit_block[access_block] = last_miss_block_time + block_time_us
                    total_wait_latency.append(int(last_miss_block_time + 2 * block_time_us - test_record[i][3]))
                if access_block in preload_series:  
                    preload_series.remove(access_block)  
        else: 
            all_hit_flag = True  
            for access_block in access_blocks:
                if access_block not in preload_blocks:
                    miss_block_num += 1
                    all_hit_flag = False
                else:
                    hit_blocks.add(access_block)
            if all_hit_flag:
                hit_num += 1
                total_wait_latency.append(0)
            else:
                if len(not_hit_block) == 0:
                    if int(interval // block_time_us) >= len(preload_series):
                        add_num = 0
                        min_wait_latency = sys.maxsize
                        for access_block in access_blocks:
                            if access_block in preload_blocks:
                                min_wait_latency = 0
                            else:
                                not_hit_block[access_block] = test_record[i][3] + add_num * block_time_us
                                min_wait_latency = min(min_wait_latency, block_time_us+add_num * block_time_us)
                                add_num += 1
                        total_wait_latency.append(min_wait_latency)
                    else:
                        cur_preload_block = preload_series[int(interval // block_time_us)]
                        cur_remain_t = block_time_us - interval % block_time_us
                        min_wait_latency = sys.maxsize
                        add_num = 0
                        for access_block in access_blocks:
                            if access_block in preload_blocks:
                                min_wait_latency = 0
                            elif access_block == cur_preload_block:
                                min_wait_latency = min(0, cur_remain_t)
                            else:
                                not_hit_block[access_block] = test_record[i][3] + cur_remain_t + add_num * block_time_us
                                min_wait_latency = min(min_wait_latency, cur_remain_t + block_time_us + add_num * block_time_us)
                                add_num += 1
                        total_wait_latency.append(min_wait_latency)
                else:
                    last_miss_block_time = list(not_hit_block.values())[-1]
                    min_wait_latency = sys.maxsize
                    add_num = 0
                    for access_block in access_blocks:
                        if access_block in preload_blocks:
                            min_wait_latency = 0
                        elif access_block in not_hit_block:
                            min_wait_latency = min(min_wait_latency, not_hit_block[access_block] + block_time_us - test_record[i][3])
                        else:
                            not_hit_block[access_block] = last_miss_block_time + block_time_us + add_num * block_time_us
                            min_wait_latency = min(min_wait_latency, last_miss_block_time + 2 * block_time_us + add_num * block_time_us - test_record[i][3])
                            add_num += 1
                    total_wait_latency.append(min_wait_latency)
                for access_block in access_blocks:
                    if access_block not in preload_blocks and access_block in preload_series:
                        preload_series.remove(access_block)
            
    for k in miss_pulled_blocks:
        assert k not in preload_series and k not in not_hit_block, f'Miss pulled block {k} should not be in preload_blocks or not_hit_block!'
    assert len(total_wait_latency) == CHECK_NUM, f'Total wait latency length {len(total_wait_latency)} does not match CHECK_NUM {CHECK_NUM} for {test_file}'
    hit_rate = hit_num / CHECK_NUM
    accuracy = len(hit_blocks) / len(final_preload_blocks) if len(final_preload_blocks) > 0 else -1
    total_wait_latency = np.array(total_wait_latency, dtype=np.int32)
    return hit_rate, total_wait_latency, accuracy

def find_count_two_time_series(alpha_num, beta_num, train_files):

    block_access_num = defaultdict(int)
    block_access_time = defaultdict(list)
    block_min_time = defaultdict(np.int32)
    block_score = {}
    max_access_num, max_access_time = 0, 0
    for train_file in train_files:
        record = np.load(train_file)
        record = record[record[:, 0]==1]
        if len(record) == 0:
            continue
        start_time = record[0, 3]
        valid_masks = record[:, 3] - start_time <= START_TIME * 1_000_000
        valid_record = record[valid_masks]
        start_blocks = valid_record[:, 1] // BLOCK_SIZE
        end_blocks = (valid_record[:, 1] + valid_record[:, 2]) // BLOCK_SIZE
        for start_block, end_block, ts in zip(start_blocks, end_blocks, valid_record[:, 3]):
            start_block, end_block = int(start_block), int(end_block)
            relative_time = np.int32(ts - start_time)
            if start_block == end_block:
                block_access_num[start_block] += 1
                block_access_time[start_block].append(relative_time)
                if start_block not in block_min_time:
                    block_min_time[start_block] = relative_time
                else:
                    block_min_time[start_block] = min(block_min_time[start_block], relative_time)
                max_access_num = max(max_access_num, block_access_num[start_block])
                max_access_time = max(max_access_time, relative_time)
            else:
                block_access_num[start_block] += 1
                block_access_num[end_block] += 1
                block_access_time[start_block].append(relative_time)
                block_access_time[end_block].append(relative_time)
                if start_block not in block_min_time:
                    block_min_time[start_block] = relative_time
                else:
                    block_min_time[start_block] = min(block_min_time[start_block], relative_time)
                if end_block not in block_min_time:
                    block_min_time[end_block] = relative_time
                else:
                    block_min_time[end_block] = min(block_min_time[end_block], relative_time)
                max_access_num = max(max_access_num, block_access_num[start_block], block_access_num[end_block])
                max_access_time = max(max_access_time, relative_time)
    for k, v in block_access_num.items():
        normal_access_num = v / max_access_num
        normal_mean_access_time = (max_access_time - np.mean(block_access_time[k])) / max_access_time
        normal_min_access_time = (max_access_time - block_min_time[k]) / max_access_time
        score = alpha_num * normal_access_num + beta_num * normal_mean_access_time + (1-alpha_num-beta_num) * normal_min_access_time
        block_score[k] = score
    block_score = dict(sorted(block_score.items(), key=lambda x: x[1], reverse=True))
    return list(block_score.keys())
