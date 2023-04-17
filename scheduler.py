import os
import sys
import json
import torch
import torchvision
import threading
import argparse
import numpy as np
from statistics import median, mean
from PIL import Image
from itertools import product
from enum import Enum
import math

from load_info_calc import prepare_for_scheduler

class Frame():
    def __init__(self, gen_time, index):
        self.gen_time = gen_time
        self.index = index
        self.processed = False
        self.processed_time = None
        self.skipped = False

# Constants
NUM_FRAMES_TOTAL = 3000
CUDA_MEM = 0.8 # CUDA uses a certain amount of memory; this should be
# replaced with amount for current GPU

# import these two classes to call run_nexus
class Mem_Level(Enum):
    Min = 0
    Fifty = 1
    Seventy_Five = 2

class Experiment_ID(Enum):
    Baseline = 0
    MaxMerge = 1
    GemelMerge = 2
    NoMerge = 3


"""Helpers for running Nexus"""

# Populate queues with frames for 3000
def populate_queue(q, fps):
    gen_time = 0
    index = 0
    for i in range(NUM_FRAMES_TOTAL):
        for model_q in q:
            f = Frame(gen_time, index)
            model_q.append(f)
        gen_time += (1000.0/fps)
        index += 1


# Calculate time to load a model, given that if it shares layers with model already in gpu, time will be shorter than standalone
def load_time(i, in_gpu, experiment_id, merged_load_info, standalone_load_info):
    # Baseline assumes no load time 
    if experiment_id == Experiment_ID.Baseline or (i in in_gpu):
        return 0.0

    if experiment_id == Experiment_ID.NoMerge:
        return standalone_load_info['LT'][i]

    # Assume it has it's stand-alone loading time unless there's a model in gpu that makes it faster
    min_load_time = standalone_load_info['LT'][i]
    for j in in_gpu:
        if (merged_load_info['times'][str(i)][str(j)]*1000) < min_load_time:
            min_load_time = (merged_load_info['times'][str(i)][str(j)]*1000)
    return min_load_time


# Calculate memory to add a model to GPU, given that if it shares layers with a model already in gpu, added memory will be less than standalone
def load_mem(i, in_gpu, experiment_id, merged_load_info, standalone_load_info):
    # with no merging, the added memory of loading is just the amount of memory the model uses to stay in GPU
    if experiment_id == Experiment_ID.Baseline or experiment_id == Experiment_ID.NoMerge:
        return standalone_load_info['LM'][i]

    # Assume it has its standalone loading mem unless there's a merged model in gpu that makes it add less than that
    min_load_mem = standalone_load_info['LM'][i]
    for j in in_gpu:
        if j == i:
            continue
        if merged_load_info['mems'][str(i)][str(j)] < min_load_mem:
            min_load_mem = merged_load_info['mems'][str(i)][str(j)]
    return min_load_mem


def current_index(i, clock_time, q):
    # if first model has loaded and we're at time 0, let it process one frame
    if clock_time == 0:
        return 1
    index = 0
    while q[i][index].gen_time <= clock_time:
        index += 1
        if index >= NUM_FRAMES_TOTAL:
            return NUM_FRAMES_TOTAL
    return index


def more_frames_to_process(q, stream_len=NUM_FRAMES_TOTAL):
    # if the last frame in any stream has neither been processed nor skipped, keep going 
    for model_q in q:
        if not model_q[stream_len-1].processed and not model_q[stream_len-1].skipped:
            return True
    return False


# Calculates stats for running a workload, given batche sizes, SLAs, memory capacity, merging info, and fps
def throughput(models, batch_sizes, SLA, M_total, experiment_id, merged_load_info, standalone_load_info, fps):
    # Start the clock and step through each model's run time
    clock_time = 0.0

    q = [[] for _ in models]
    populate_queue(q, fps)

    in_gpu = []

    total_load_time = 0.0
    total_num_swap = 0

    while more_frames_to_process(q):
        for i in range(0, len(models)):
            # Check if we need to evict other models first
            sum_mem = CUDA_MEM
            was_already_in_gpu = False
            in_gpu_temp = in_gpu.copy()
            for model in in_gpu:
                if model != i:
                    hypothetical_in_gpu = in_gpu.copy()
                    hypothetical_in_gpu.append(i)
                    sum_mem += load_mem(model, hypothetical_in_gpu, experiment_id, merged_load_info, standalone_load_info)                
                else:
                    # If it's already there, remove it so we can add it to the end
                    in_gpu_temp.remove(i)
                    was_already_in_gpu = True
            in_gpu = in_gpu_temp.copy()
            sum_mem += standalone_load_info['RM'][i][str(batch_sizes[i])]
            while sum_mem > M_total:
                # As long as there are items in gpu, remove them starting from most recently used
                # If there are no items and memory to run still doesn't fit, this batch size doesn't work
                if in_gpu:
                    sum_mem -= load_mem(in_gpu[-1], in_gpu, experiment_id, merged_load_info, standalone_load_info)
                    remove_id = in_gpu[-1]
                    in_gpu = in_gpu[:-1]
                    total_num_swap += 1
                else:
                    print(f'Returning none because memory is too high to run next model and there is nothing left to evict')
                    return None, None, None, None

            # Check whether load time for this model exceeds run time of previous model
            if was_already_in_gpu:
                load_time_curr = 0.0
            else:
                load_time_curr = load_time(i, in_gpu, experiment_id, merged_load_info, standalone_load_info)
            in_gpu.append(i)

            if load_mem(i, in_gpu, experiment_id, merged_load_info, standalone_load_info) + standalone_load_info['RM'][i-1][str(batch_sizes[i-1])] + CUDA_MEM > M_total:
                clock_time += load_time_curr
                total_load_time += load_time_curr
            else:
                if load_time_curr > standalone_load_info['RT'][i-1][str(batch_sizes[i-1])]:
                    clock_time += (load_time_curr - standalone_load_info['RT'][i-1][str(batch_sizes[i-1])])
                    total_load_time += (load_time_curr - standalone_load_info['RT'][i-1][str(batch_sizes[i-1])])
            # Check whether there are enough frames in queue to warrant this batch size and if so, whether they'll meet SLA
            frames_available = True
            while frames_available:
                curr_index_in_queue = current_index(i, clock_time, q)
                if curr_index_in_queue >= NUM_FRAMES_TOTAL:
                    unprocessed_frames = [f for f in q[i] if not f.processed and not f.skipped]
                else:
                    unprocessed_frames = [f for f in q[i][:curr_index_in_queue] if not f.processed and not f.skipped]

                # Process frames in queue starting from current time, either batch_sizes[i] frames or 
                # the number of frames in the queue if that's less
                if len(unprocessed_frames) > batch_sizes[i]:
                    frames_this_batch = unprocessed_frames[-batch_sizes[i]:]
                    for f in unprocessed_frames[:-batch_sizes[i]]:
                        assert(not f.processed)
                        f.skipped = True
                else:
                    frames_this_batch = unprocessed_frames

                do_process = frames_this_batch
                if frames_this_batch:
                    batch_index = 0
                    while frames_this_batch[batch_index].gen_time + SLA < clock_time + standalone_load_info['RT'][i][str(batch_sizes[i])] and frames_this_batch[batch_index].index != NUM_FRAMES_TOTAL-1:
                        frames_this_batch[batch_index].skipped = True
                        batch_index += 1
                        if batch_index > len(frames_this_batch) - 1:
                            do_process = False
                            break
                if do_process:
                    clock_time += standalone_load_info['RT'][i][str(batch_sizes[i])]
                    for f in frames_this_batch:
                        f.processed = True
                        f.processed_time = clock_time
                    frames_available = False
                else:
                    if curr_index_in_queue == NUM_FRAMES_TOTAL:
                        break
                    clock_time += 1

    throughputs = []
    for m, model_queue in enumerate(q):
        frames_processed = 0
        total_frames = 0
        # Only consider frames from rounds that were completed
        for frame in [f for f in model_queue if f.index <= NUM_FRAMES_TOTAL]:
            if frame.processed:
                frames_processed += 1
            total_frames += 1
        throughput = round(frames_processed/total_frames, 2)
        throughputs.append(throughput)

    return throughputs, q, total_load_time/clock_time, total_num_swap

# Return whether the current throughputs are the best we've seen so far
def better_throughput(curr_throughputs, best_throughputs):
    if not best_throughputs:
        return True
    s_curr = sorted(curr_throughputs)
    s_best = sorted(best_throughputs)
    assert(len(s_curr) == len(s_best))
    i = 0
    while i < len(s_curr):
        if s_curr[i] < s_best[i]:
            return False
        elif s_curr[i] > s_best[i]:
            return True
        i += 1
    return False

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def calculate_M_total(models, mem_level, experiment_id, standalone_load_info):
    if experiment_id == Experiment_ID.Baseline:
        return 100.0
    RM = [standalone_load_info['RM'][i]['1'] for i in range(len(models))]
    LM = [standalone_load_info['LM'][j] for j in range(len(models))]

    max_run_mem = max(RM)
    index_max = RM.index(max_run_mem)

    min_mem_needed = round_decimals_up(max_run_mem + CUDA_MEM)
    
    mem_for_all_to_fit = sum(LM) + max_run_mem + CUDA_MEM - LM[index_max]

    if mem_level == Mem_Level.Min:
        return min_mem_needed
    elif mem_level == Mem_Level.Fifty:
        return min_mem_needed + (mem_for_all_to_fit - min_mem_needed)*0.50
    else:
        return min_mem_needed + (mem_for_all_to_fit - min_mem_needed)*0.75

# Nexus must be passed a workload and a file with the load and run times
def run_nexus(workload, mem_level, experiment_id, path_to_load_file, sla, fps):
    fname_nexus_result = workload.scheduler_results_template.format(exp_id=experiment_id, memory=mem_level, sla=sla, fps=fps)
    print(fname_nexus_result)

    if os.path.exists(fname_nexus_result):
        return
    
    # Get model list from workload
    models = [job.model for job in workload.workload]
    
    merged_load_info = None
    if experiment_id == Experiment_ID.MaxMerge or experiment_id == Experiment_ID.GemelMerge:
        fname = workload.scheduler_input_gemel if experiment_id == Experiment_ID.GemelMerge else workload.scheduler_input_max
        load_file = open(fname, 'r')
        merged_load_info = json.load(load_file)


    # Get standalone info and filter for the models in this workload
    # Standalone info should contains load time (LT), runtime (RT) at different batch sizes,
    # loading mem (LM), and running mem(RM) at different batch sizes
    standalone_file = open('/home/ubuntu/scheduler/standalone_load_info.json', 'r')
    standalone_load_info = json.load(standalone_file)
    standalone_filtered = {}
    standalone_filtered['LT'] = [standalone_load_info[m]['LT'] for m in models]
    standalone_filtered['RT'] = [standalone_load_info[m]['RT'] for m in models]
    standalone_filtered['LM'] = [standalone_load_info[m]['LM'] for m in models]
    standalone_filtered['RM'] = [standalone_load_info[m]['RM'] for m in models]

    M_total = calculate_M_total(models, mem_level, experiment_id, standalone_filtered)
    print(f'M_total is {M_total}')

    combs = []
    potential_batch_sizes = [1, 2, 4]
    for i in product(potential_batch_sizes, repeat=len(models)):
        combs.append(list(i))

    print(f'SLA: {sla}, M total: {M_total}, id: {experiment_id}')
    best_throughput = []
    max_q = None
    best_batch_sizes = None

    best_load_time = None
    best_num_swaps = None
    index = 0
    for batch_sizes in combs:
        throughputs, q, total_load_time, num_swaps = throughput(models, batch_sizes, sla, M_total, experiment_id, merged_load_info, standalone_filtered, fps)
        print(f'batch sizes: {batch_sizes}, throughputs: {throughputs}')
        if throughputs == None:
            continue
        if better_throughput(throughputs, best_throughput):
            best_throughput = throughputs
            max_q = q
            best_batch_sizes = batch_sizes
            best_num_swaps = num_swaps
            best_load_time = total_load_time

        index += 1
    print(f'At end, best: batch_sizes: {best_batch_sizes}, best throughput: {best_throughput}, best num swaps: {best_num_swaps}, best load times: {best_load_time}')

    # Package up results into dict and save output. Final result should be throughputs
    results = {}
    results['batch_sizes'] = best_batch_sizes
    results['num_swaps'] = best_num_swaps
    results['load_time'] = best_load_time
    results['throughputs'] = best_throughput
    results['avg_throughput'] = sum(results['throughputs'])/len(results['throughputs'])
    print(results)

    with open(fname_nexus_result, 'w') as final_results_file:
        json.dump(results, final_results_file)
