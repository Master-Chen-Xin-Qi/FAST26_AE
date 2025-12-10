# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import os
import time
import random
import argparse
import logging
import warnings
import json
import polars as pl
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from AE_config import meta_file
from utils import setup_logging, train_test_map, get_map_length, union_preload_process
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

res_path = '../result/sufficient'
param_path = '../data/eval_files/params'
train_info_file = '../data/eval_files/trainset_info.npy'
CONSEC_GEN = 30
MAX_WORKERS = 15

class GeneticAlgorithm:
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.param_ranges = {
            'alpha_weight': (0.1, 0.2),      
            'beta_weight': (0.0, 0.9),       
            'gamma_weight': (0.0, 0.9),    
        }
        
        self.population = []
        self.fitness_scores = []
        self.best_individual = {}
        self.best_fitness = -float('inf')
        
    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual()
            self.population.append(individual)
    
    def _create_random_individual(self) -> Dict[str, float]:
        individual = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            individual[param_name] = random.uniform(min_val, max_val)
        
        total_weight = (individual['alpha_weight'] + 
                       individual['beta_weight'] + 
                       individual['gamma_weight'])
        
        alpha_ratio = individual['alpha_weight'] / total_weight
        if alpha_ratio < 0.1:
            alpha_ratio = 0.1
        elif alpha_ratio > 0.2:
            alpha_ratio = 0.2
        
        individual['alpha_weight'] = alpha_ratio
        
        remaining_weight = 1.0 - alpha_ratio
        
        if abs(individual['beta_weight'] + individual['gamma_weight']) < 1e-10:
            individual['beta_weight'] = remaining_weight * 0.5
            individual['gamma_weight'] = remaining_weight * 0.5
        else:
            beta_ratio = individual['beta_weight'] / (individual['beta_weight'] + individual['gamma_weight'])
            individual['beta_weight'] = remaining_weight * beta_ratio
            individual['gamma_weight'] = remaining_weight * (1 - beta_ratio)

        return individual
    
    def selection(self) -> List[Dict[str, float]]:

        fitness_sum = sum(max(0, score) for score in self.fitness_scores) 
        if fitness_sum == 0:
            probabilities = [1.0 / len(self.fitness_scores)] * len(self.fitness_scores)
        else:
            probabilities = [max(0, score) / fitness_sum for score in self.fitness_scores]
        
        selected = []
        for _ in range(self.population_size):
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(self.population[i].copy())
                    break
        
        return selected
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()

        for param_name in self.param_ranges.keys():
            if param_name not in child1:
                child1[param_name] = parent1.get(param_name, 0.0)
            if param_name not in child2:
                child2[param_name] = parent2.get(param_name, 0.0)
        
        crossover_point = random.randint(1, len(self.param_ranges) - 1)
        param_names = list(self.param_ranges.keys())
        
        for i in range(crossover_point, len(param_names)):
            param_name = param_names[i]
            child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        for child in [child1, child2]:
            total_weight = (child['alpha_weight'] + 
                          child['beta_weight'] + 
                          child['gamma_weight'])
            
            alpha_ratio = child['alpha_weight'] / total_weight
            if alpha_ratio < 0.1:
                alpha_ratio = 0.1
            elif alpha_ratio > 0.2:
                alpha_ratio = 0.2
            
            child['alpha_weight'] = alpha_ratio
            
            remaining_weight = 1.0 - alpha_ratio
            
            if abs(child['beta_weight'] + child['gamma_weight']) < 1e-10:
                child['beta_weight'] = remaining_weight * 0.5
                child['gamma_weight'] = remaining_weight * 0.5
            else:
                beta_ratio = child['beta_weight'] / (child['beta_weight'] + child['gamma_weight'])
                child['beta_weight'] = remaining_weight * beta_ratio
                child['gamma_weight'] = remaining_weight * (1 - beta_ratio)
        
        return child1, child2
    
    def mutation(self, individual: Dict[str, float]):
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                current_val = individual[param_name]
                mutation_range = (max_val - min_val) * 0.2
                new_val = current_val + random.uniform(-mutation_range, mutation_range)
                new_val = max(min_val, min(max_val, new_val))
                individual[param_name] = new_val
        
        total_weight = (individual['alpha_weight'] + 
                       individual['beta_weight'] + 
                       individual['gamma_weight'])
        
        alpha_ratio = individual['alpha_weight'] / total_weight
        if alpha_ratio < 0.1:
            alpha_ratio = 0.1
        elif alpha_ratio > 0.2:
            alpha_ratio = 0.2
        
        individual['alpha_weight'] = alpha_ratio
        
        remaining_weight = 1.0 - alpha_ratio
        if abs(individual['beta_weight'] + individual['gamma_weight']) < 1e-10:
            individual['beta_weight'] = remaining_weight * 0.5
            individual['gamma_weight'] = remaining_weight * 0.5
        else:
            beta_ratio = individual['beta_weight'] / (individual['beta_weight'] + individual['gamma_weight'])
            individual['beta_weight'] = remaining_weight * beta_ratio
            individual['gamma_weight'] = remaining_weight * (1 - beta_ratio)

    def evolve(self, train_info: Dict[str, Any], train_files: List[str], bandwidth: int, logger, generations: int = 30) -> Dict[str, float]:

        logger.info(f'Start evolving，population size: {self.population_size}, generation: {generations}')
        self.initialize_population()
        consecutive_num = 0
        eval_num = max(int(len(train_files) * 0.1), 10)
        logger.info(f'评估文件个数: {eval_num}')
        for generation in range(generations):
            eval_files = random.sample(train_files, eval_num)
            self.fitness_scores = []
            futures = []
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for individual in self.population:
                    futures.append(executor.submit(evaluate_fitness, individual, train_info, eval_files, bandwidth, logger))

                for res in as_completed(futures):
                    fitness = res.result()
                    self.fitness_scores.append(fitness)
            
            best_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[best_idx] == self.best_fitness:
                consecutive_num += 1
            else:
                consecutive_num = 0
            if self.fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness_scores[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            current_best = max(self.fitness_scores)
            avg_fitness = np.mean(self.fitness_scores)
            
            if generation % 5 == 0:
                logger.info(f'Generation {generation}: best fitness = {current_best:.4f}, average fitness = {avg_fitness:.4f}')
            
            selected = self.selection()
            new_population = []
            new_population.append(self.best_individual.copy())
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutation(child1)
                self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]

            if consecutive_num >= CONSEC_GEN:
                logger.info(f'Current generation: {generation}, consecutive {CONSEC_GEN} generation is the best，stop evolution.')
                break
        
        logger.info(f'Genetic algorithm finishes，the best fitness: {self.best_fitness:.4f}')
        return self.best_individual

def evaluate_fitness(individual: Dict[str, float], train_info: Dict[str, Any], eval_files: List[str], bandwidth: int, logger) -> float:

    try:
        total_score = 0
        valid_count = 0
        df = pl.read_csv(f'../{meta_file}')
        for train_file in eval_files:
            preload_series = compute_genetic_score_series(train_info, individual)
            res = union_preload_process(bandwidth, df, preload_series, train_file, None, False)
            
            if res is not None:
                hit_rate, wait_latency, accuracy = res
                accuracy_bonus = accuracy * 0.1 if accuracy > 0 else 0
                score = hit_rate + accuracy_bonus
                total_score += score
                valid_count += 1
        
        return total_score / valid_count if valid_count > 0 else -float('inf')
        
    except Exception as e:
        logger.warning(f'Wrong when evaluating: {e}')
        return -float('inf')

def compute_genetic_score_series(train_info: Dict[str, Any], individual: Dict[str, float]) -> List[int]:
    
    block_access_num = train_info['access_num']
    block_access_avg_time = train_info['access_avg_time']
    block_access_min_time = train_info['access_min_time']
    max_access_num = train_info['max_access_num']
    max_access_time = train_info['max_access_time']
    
    block_score = {}
    for block_id, access_count in block_access_num.items():
        if block_id not in block_access_avg_time or block_id not in block_access_min_time:
            continue
        normalized_access_count = access_count / max_access_num
        frequency_score = normalized_access_count
        
        normalized_avg_time = (max_access_time - block_access_avg_time[block_id]) / max_access_time
        normalized_min_time = (max_access_time - block_access_min_time[block_id]) / max_access_time
        
        score = individual['alpha_weight'] * frequency_score + individual['beta_weight'] * normalized_avg_time + individual['gamma_weight'] * normalized_min_time
        block_score[block_id] = score
    
    block_score = dict(sorted(block_score.items(), key=lambda x: x[1], reverse=True))
    
    return list(block_score.keys())


def train_version_specific_params(args, all_train_infos, train_files_map, logger):

    version_params = {}
    version_performance = {}
    
    logger.info('Train specific parameters for each version...')
    version_num = len(train_files_map)
    for i, (train_version, train_files) in enumerate(train_files_map.items()):
        logger.info(f'----------Train version: {train_version}----------')
        
        if train_version not in all_train_infos:
            logger.warning(f'No training infors for {train_version}, skip')
            continue
        
        train_infos = all_train_infos[train_version]
        
        ga = GeneticAlgorithm(population_size=15, mutation_rate=0.15, crossover_rate=0.8)
        
        logger.info(f'Train for {train_version}, using {len(train_files)} files [{i+1}/{version_num}]')
        best_params = ga.evolve(train_infos, train_files, args.bandwidth, logger, generations=args.optimization_generations)
        
        version_params[train_version] = best_params
        version_performance[train_version] = ga.best_fitness
        
        logger.info(f'Version {train_version} train complete，best fitness: {ga.best_fitness:.4f}，best param: {best_params}')
    
    return version_params, version_performance


def thinkahead_train(args, train_files_map, logger):

    all_train_infos = np.load(train_info_file, allow_pickle=True).item()
    logger.info('Start genetic training...')

    param_save_path = os.path.join(param_path, f'sufficient_thinkahead_params_{args.bandwidth}MB.npy')
    performance_save_path = os.path.join(res_path, f'sufficient_thinkahead_performance_{args.bandwidth}MB.npy')
    version_params, version_performance = train_version_specific_params(args, all_train_infos, train_files_map, logger)
    np.save(param_save_path, np.array(version_params), allow_pickle=True)
    np.save(performance_save_path, np.array(version_performance), allow_pickle=True)


def thinkahead_test(args, test_files_map, logger):
    df = pl.read_csv(f'../{meta_file}')
    total_test_version = len(test_files_map)
    all_train_infos = np.load(train_info_file, allow_pickle=True).item()
    param_save_path = os.path.join(param_path, f'sufficient_thinkahead_params_{args.bandwidth}MB.npy')
    assert os.path.exists(param_save_path), f'No parameter file found at {param_save_path}, please run training first.'
    version_params = np.load(param_save_path, allow_pickle=True).item()
    avg_hit_rate_res, total_hit_rate_res = {}, {}
    total_wait_latency_res = {}
    total_accuracy_res = {} 
    
    for i, (test_version, test_files) in enumerate(test_files_map.items()): 
        logger.info(f'----------Test version: {test_version} [{i+1}/{total_test_version}]----------')
        avg_hit_rate = 0
        file_num = len(test_files)
        valid_num = 0
        for j, test_file in enumerate(test_files):
            if test_version not in all_train_infos:  # zero-shot
                assert False, f'No train infos for {test_version}, please check the trainset info file {train_info_file}'
            else:  # few-shot
                preload_series = compute_genetic_score_series(all_train_infos[test_version], version_params[test_version])
                logger.info(f'[{j+1}/{file_num}] Test file: {test_file}')
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
    avg_hit_rate_save_path = os.path.join(res_path, f'avg_thinkahead_hit_rate_{args.bandwidth}MB.json')
    all_hit_rate_save_path = os.path.join(res_path, f'all_thinkahead_hit_rate_{args.bandwidth}MB.json')
    all_wait_latency_save_path = os.path.join(res_path, f'all_thinkahead_wait_latency_{args.bandwidth}MB.npy')
    all_accuracy_save_path = os.path.join(res_path, f'all_thinkahead_accuracy_{args.bandwidth}MB.json')
    logger.info('All finished!')

    with open(avg_hit_rate_save_path, 'w') as f:
        json.dump(avg_hit_rate_res, f)
    with open(all_hit_rate_save_path, 'w') as f:
        json.dump(total_hit_rate_res, f)
    with open(all_accuracy_save_path, 'w') as f:
        json.dump(total_accuracy_res, f)
    np.save(all_wait_latency_save_path, np.array(total_wait_latency_res))
    avg_rate = np.mean(list(total_hit_rate_res.values()))
    p50_latency = np.percentile([lat for lat_list in total_wait_latency_res.values() for lat in lat_list], 50)
    print(f'Sufficient ThinkAhead Average Hit Rate: {avg_rate:.4f}, P50 Wait Latency: {p50_latency} us.')
    print(f'Save all and average hit rate results to {os.path.abspath(avg_hit_rate_save_path)}:1 and {os.path.abspath(all_hit_rate_save_path)}:1, save accuracy results to {os.path.abspath(all_accuracy_save_path)}:1, save latency results to {os.path.abspath(all_wait_latency_save_path)}:1 finished!')
    

def thinkahead_exp(args):
    logger = logging.getLogger('file_logger')
    logger.info(f'Parameters: {vars(args)}')
    logger.info(f'OSS bandwidth: {args.bandwidth}MB/s')
    
    train_files_map, test_files_map = train_test_map(args)
    get_map_length(train_files_map, test_files_map, logger)
    
    if args.mode == 'train':  # test after train
        thinkahead_train(args, train_files_map, logger)
        thinkahead_test(args, test_files_map, logger)
    elif args.mode == 'test':
        thinkahead_test(args, test_files_map, logger)
    else:
        raise ValueError(f'Invalid mode {args.mode}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', action='store_true', help='debug mode')
    parser.add_argument('--mode', '-m', choices=['train', 'test'], default='train')
    parser.add_argument('--bandwidth', '-bw', type=int, required=True, choices=[2, 4, 6, 8, 10, 30, 50, 70, 90, 150], help='OSS bandwidth (MB/s)')
    parser.add_argument('--optimization_generations', '-og', type=int, default=60)
    parser.add_argument('--small_interval', '-si', action='store_true')
    parser.set_defaults(func=thinkahead_exp)
    args = parser.parse_args()
    args.data_type = 'few-shot'
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    global res_path
    if args.debug:
        logfile = 'thinkahead_debug.log'
    else:
        res_path = os.path.join(res_path)
        logfile = f'{args.bandwidth}MB_{args.mode}_{cur_time}.log'
        if not os.path.exists(res_path):
            os.makedirs(res_path)

    setup_logging(log_name=logfile, log_path='../log/sufficient')
    args.func(args)

    # Example command to reproduce the results:
    # python sufficient_thinkahead.py -bw 2 -m test
    # Example command to train models from scratch:
    # python sufficient_thinkahead.py -bw 2 -m train


if __name__ == "__main__":
    main() 
    