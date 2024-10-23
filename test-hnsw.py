#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
import random
import itertools
random.seed(108)
from hnsw import HNSW
from hnsw import l2_distance, heuristic
import concurrent.futures


def brute_force_knn_search(distance_func, k, q, data):
        '''
        Return the list of (idx, dist) for k-closest elements to {x} in {data}
        '''
        return sorted(enumerate(map(lambda x: distance_func(q, x) ,data)), key=lambda a: a[1])[:k]

# def calculate_recall_single_query(query, true_neighbors, hnsw, k, ef):
#     observed = [neighbor for neighbor, dist in hnsw.search(q=query, k=k, ef=ef, return_observed = True)]
#     total_calc = total_calc + len(observed)
#     results = observed[:k]
#     intersection = len(set(true_neighbors).intersection(set(results)))
#     # print(f'true_neighbors: {true_neighbors}, results: {results}. Intersection: {intersection}')
#     recall = intersection / k
#     return recall, total_calc



def run_experiment(kg, test, groundtruth, k, ef, m, distance_func):
    recall, avg_calc = calculate_recall(distance_func, kg, test, groundtruth, k, ef, m)
    return recall, avg_calc, ef



def calculate_recall(distance_func, kg: HNSW, test, groundtruth, k, ef, m):
    if groundtruth is None:
        print("Ground truth not found. Calculating ground truth...")
        groundtruth = [ [idx for idx, dist in brute_force_knn_search(distance_func, k, query, kg.data)] for query in tqdm(test)]

    print("Calculating recall...")
    recalls = []
    total_calc = 0

    # median_points = calculate_median_points(kg.data, num_medians=m)
    # print("mp ", median_points)
    # median_points_indices = find_closest_indices(kg.data, median_points)
    # print("mpi ", median_points_indices)
    # print("sdata", kg.data)
    for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
        # true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
        # entry_points = random.sample(range(len(kg.data)), m)
        observed = [neighbor for neighbor, dist in kg.search(q=query, k=k, ef=ef, return_observed = True)]
        # observed = [neighbor for neighbor, dist in kg.beam_search(graph=kg._graphs[0], eps=[kg._enter_point], q=query, k=k, ef=ef, return_observed = True)]
        total_calc = total_calc + len(observed)
        results = observed[:k]
        intersection = len(set(true_neighbors[:k]).intersection(set(results)))
        # print(f'true_neighbors: {true_neighbors}, results: {results}. Intersection: {intersection}')
        recall = intersection / k
        recalls.append(recall)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = []
    #     for query, true_neighbors in zip(test, groundtruth):
    #         future = executor.submit(calculate_recall_single_query, query, true_neighbors[:k], kg, k, ef)
    #         futures.append(future)

    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
    #         recall, calc = future.result()
    #         recalls.append(recall)
    #         total_calc += calc
    # for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
    #     true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
    #     entry_points = random.sample(range(len(kg.data)), m)

    #     recalls.append(recall)

    return np.mean(recalls), total_calc/len(test)

def run_parallel_experiments(kg, test, groundtruth, k, m, ef_list, distance_func):
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, kg, test, groundtruth, k, ef, m, distance_func) for ef in ef_list]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())

    results.sort(key=lambda x: x[2])
    return results

# def experiment_and_save_results(train_data, test_data, groundtruth_data, ef_values, k, m, output_file):
#     # hnsw = HNSW(distance_func=l2_distance, m=50, m0=50, ef=10, ef_construction=30, neighborhood_construction = heuristic)

#     # for x in tqdm(train_data):
#     #     hnsw.add(x)

#     with open(output_file, "w") as f:
#         f.write("ef,recall,avg_calculations\n")
#         for ef in ef_values:
#             recall, avg_calc = calculate_recall(l2_distance, hnsw, test_data, groundtruth_data, k=k, ef=ef, m=m)
#             f.write(f"{ef},{recall},{avg_calc}\n")
#             print(f"ef={ef}, average recall: {recall}, avg calc: {avg_calc}")

def save_results_file(results, file_path):
    with open(file_path, "w") as f:
        f.write("Recall, AvgCalc, ef\n")
        for recall, avg_calc, ef in results:
            f.write(f"{recall},{avg_calc},{ef}\n")
    print("Results saved")

def load_results(filename):
    ef_values = []
    recalls = []
    avg_calculations = []

    with open(filename, "r") as f:
        next(f)
        for line in f:
            ef, recall, avg_calc = map(float, line.strip().split(","))
            ef_values.append(ef)
            recalls.append(recall)
            avg_calculations.append(avg_calc)

    return ef_values, recalls, avg_calculations


def read_fvecs(filename):
    with open(filename, 'rb') as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if not vec_size:
                break
            vec = np.fromfile(f, dtype=np.float32, count=vec_size[0])
            yield vec


def read_ivecs(filename):
    with open(filename, 'rb') as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if not vec_size:
                break
            vec = np.fromfile(f, dtype=np.int32, count=vec_size[0])
            yield vec


def load_sift_dataset():
    train_file = 'datasets/siftsmall/siftsmall_base.fvecs'
    test_file = 'datasets/siftsmall/siftsmall_query.fvecs'
    groundtruth_file = 'datasets/siftsmall/siftsmall_groundtruth.ivecs'

    train_data = np.array(list(read_fvecs(train_file)))
    test_data = np.array(list(read_fvecs(test_file)))
    groundtruth_data = np.array(list(read_ivecs(groundtruth_file)))

    return train_data, test_data, groundtruth_data


def generate_synthetic_data(dim, n, nq):
    train_data = np.random.random((n, dim)).astype(np.float32)
    test_data = np.random.random((nq, dim)).astype(np.float32)
    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Test recall of beam search method with KGraph.')
    parser.add_argument('--dataset', choices=['synthetic', 'sift'], default='synthetic', help="Choose the dataset to use: 'synthetic' or 'sift'.")
    # parser.add_argument('--K', type=int, default=5, help='The size of the neighbourhood')
    parser.add_argument('--M', type=int, default=50, help='Avg number of neighbors')
    parser.add_argument('--M0', type=int, default=50, help='Avg number of neighbors')
    parser.add_argument('--dim', type=int, default=2, help='Dimensionality of synthetic data (ignored for SIFT).')
    parser.add_argument('--n', type=int, default=200, help='Number of training points for synthetic data (ignored for SIFT).')
    parser.add_argument('--nq', type=int, default=50, help='Number of query points for synthetic data (ignored for SIFT).')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to search in the test stage')
    # parser.add_argument('--ef', type=int, default=10, help='Size of the beam for beam search.')
    parser.add_argument('--m', type=int, default=3, help='Number of random entry points.')
    parser.add_argument('--output', type=str, default="results.txt", help="File to save experiments")
    # parser.add_argument('--plot', action="store_true", help="Flag to plot")

    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'sift':
        print("Loading SIFT dataset...")
        train_data, test_data, groundtruth_data = load_sift_dataset()
    else:
        print(f"Generating synthetic dataset with {args.dim}-dimensional space...")
        train_data, test_data = generate_synthetic_data(args.dim, args.n, args.nq)
        groundtruth_data = None

    ef_values = [5, 10, 20, 30, 40, 50]
    
    hnsw = HNSW( distance_func=l2_distance, m=args.M, m0=args.M0, ef=10, ef_construction=30, neighborhood_construction = heuristic)
    # Add data to HNSW
    for x in tqdm(train_data):
        hnsw.add(x)

    results = run_parallel_experiments(hnsw, test_data, groundtruth_data, args.k, args.m, ef_values, l2_distance)
    save_results_file(results, args.output)

    
        # ef_values, recalls, avg_calculations = load_results(args.output)
        # plot_results(ef_values, recalls, avg_calculations)

    # Create HNSW




    # # Calculate recall
    # recall, avg_cal = calculate_recall(l2_distance, hnsw, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m)
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
