import argparse
import math
import time

import networkx as nx
import numpy as np
from hyperopt import hp
import hyperopt.pyll
import pandas as pd

from src.traffic.v3.environment import TrafficModel
from src.traffic.v3.util.network import create_cars, create_random_gnp_graph, UniformLatencyGenerator
from src.traffic.v3.util.utils import save_or_extend_dataframe

a_max = hp.randint('a_max', 3, 5)
a_min = hp.randint('a_min', 1, a_max)
b_max = hp.randint('b_max', 3, 5)
b_min = hp.randint('b_min', 1, b_max)
c_max = hp.randint('c_max', 2, 30)
c_min = hp.randint('c_min', 1, c_max)
cars = hp.randint('cars', 30, 50)
n = (30, 100)
seed = 46


def single_run(sample, task, edge=None, allowed=True, nodes=40, p=0.1, steps=1000):
    network = create_random_gnp_graph(number_of_nodes=nodes, p=p,
                                      latency_generator=UniformLatencyGenerator(a_min=sample['a_min'],
                                                                                a_max=sample['a_max'],
                                                                                b_min=sample['b_min'],
                                                                                b_max=sample['b_max'],
                                                                                c_min=sample['c_min'],
                                                                                c_max=sample['c_max']),
                                      seed=seed)

    model = TrafficModel(network, create_cars(network, car_counts={task: sample['cars']},
                                              seed=seed))

    if edge is not None:
        model.set_edge_restriction(edge, allowed=allowed)

        if not nx.has_path(model.allowed_network, task[0], task[1]):
            print('Skipping as the task got infeasible!')
            return -np.inf, None, None

        model.cars = create_cars(model.allowed_network, car_counts={task: sample['cars']},
                                 seed=seed)

    step_statistics, car_statistics = model.run_sequentially(steps)

    return (-car_statistics["travel_time"][-int(
        len(car_statistics) / 2):]).mean() if 'travel_time' in car_statistics.columns else -np.inf, step_statistics, car_statistics


def get_task(n, p):
    network = create_random_gnp_graph(number_of_nodes=n, p=p,
                                      latency_generator=UniformLatencyGenerator(a_min=0,
                                                                                a_max=1,
                                                                                b_min=0,
                                                                                b_max=1,
                                                                                c_min=0,
                                                                                c_max=1),
                                      seed=seed)

    task = tuple(np.random.choice(network.nodes, 2))
    while not nx.has_path(network, task[0], task[1]) or len(nx.shortest_path(network, task[0], task[1])) <= 2:
        task = tuple(np.random.choice(network.nodes, 2))

    return task


def find_braess(search_space, samples=10, steps=1000):
    # Draw Graph
    graph_nodes = np.random.randint(n[0], n[1])
    graph_p = math.sqrt(graph_nodes) / 100
    task = get_task(graph_nodes, graph_p)
    print(f'Graph setup: G({graph_nodes}, {graph_p}) with task {task}')

    # Find latency and car parameters
    results = pd.DataFrame(columns=list(search_space.keys()))
    for iteration in range(samples):
        # Random parameter sample
        sample = hyperopt.pyll.stochastic.sample(search_space)
        print(f'Sample: {sample} (Iteration {iteration})')

        # Unrestricted evaluation
        unrestricted_performance, step_stats, car_stats = single_run(sample,
                                                                     task=task,
                                                                     steps=steps,
                                                                     nodes=graph_nodes,
                                                                     p=graph_p)

        # Five most used routes
        top_unrestricted_edges = set(
            [(route[index], route[index + 1]) for route in list(
                car_stats.groupby('route').count().sort_values(
                    ascending=False, by='step'
                ).head().index) for index, node in enumerate(route) if index < len(route) - 1])

        # Iterate through edges of most used routes and calculate improvement through removal
        for edge in top_unrestricted_edges:
            print(f'Processing edge {edge}')

            results = pd.concat([results, pd.DataFrame({
                **sample,
                'n': graph_nodes,
                'p': graph_p,
                'task': [task],
                'edge': [edge],
                'seed': [seed],
                'mean_travel_time': [
                    single_run(sample, task, edge, False, graph_nodes, graph_p, steps)[0]
                    - unrestricted_performance]
            })])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--name', type=str, required=False, default='results')
    args = parser.parse_args()
    print(f'Starting braess search with {args.samples} with {args.steps} for {args.seconds} seconds!!')

    space = {
        'a_min': a_min,
        'a_max': a_max,
        'b_min': b_min,
        'b_max': b_max,
        'c_min': c_min,
        'c_max': c_max,
        'cars': cars
    }

    start_time = time.time()
    while time.time() < start_time + args.seconds:
        save_or_extend_dataframe(find_braess(space, args.samples, args.steps), f'{args.name}.pkl')
