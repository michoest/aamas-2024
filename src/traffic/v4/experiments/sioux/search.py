import argparse

import pandas as pd
import networkx as nx
import numpy as np

from src.traffic.v3.util.utils import save_or_extend_dataframe
from src.traffic.v4.src.environment import build_network, create_cars, TrafficModel, Car

seeds = [41, 42, 43, 44, 45, 46]
network_path = 'data/sf1_net.txt'
costs_path = 'data/sf1_c.txt'
cars_path = 'data/sf1_dem.txt'


def create_sioux_falls_network(network_path: str, costs_path: str, capacity=2, scaling: int = 60):
    net = pd.read_csv(network_path, sep='\s+', names=['init_node', 'term_node'],
                      index_col=False)
    n = nx.DiGraph([(edge['init_node'], edge['term_node']) for index, edge in net.iterrows()])

    c = pd.read_csv(costs_path,
                    sep='\s+',
                    names=['free_flow', 'b', 'capacity', 'power'],
                    index_col=False)
    nx.set_edge_attributes(
        n,
        {
            (int(net[net.index == i]['init_node'].iloc[0]),
             int(net[net.index == i]['term_node'].iloc[0])): (scaling * edge['free_flow'],
                                                              scaling * edge['free_flow'] * edge['b']
                                                              / (edge['capacity'] ** edge['power']),
                                                              capacity,
                                                              4) for i, edge in c.iterrows()
        },
        "latency_params",
    )

    return build_network(n)


def create_sioux_falls_cars(path, network, max_steps, scaling: int = 10, seed=42):
    trips = pd.read_csv(path, sep=' ', names=['s', 't', 'cars'], index_col=False)
    routes = {(trip['s'], trip['t']): trip['cars'] for _, trip in trips.iterrows()}

    rng = np.random.RandomState(seed)
    start_times = rng.randint(max_steps, size=3600 * scaling)

    cars = {}
    added_cars = 0
    for route, count in routes.items():
        for car in range(count * scaling):
            if start_times[added_cars] in cars.keys():
                cars[start_times[added_cars]][added_cars] = Car(added_cars, route[0], route[1], 0,
                                                                ((route[0], route[0]), 1.0),
                                                                respawn=False, created_at_step=start_times[added_cars],
                                                                seed=seed)
            else:
                cars[start_times[added_cars]] = {
                    added_cars: Car(added_cars, route[0], route[1], 0, ((route[0], route[0]), 1.0),
                                    respawn=False, created_at_step=start_times[added_cars],
                                    seed=seed)}
            added_cars += 1

    return cars


def single_run(steps=300, edge=None, seed=42):
    network = create_sioux_falls_network(network_path, costs_path, capacity=200, scaling=100)
    model = TrafficModel(network, create_sioux_falls_cars(cars_path, network, 24 * 100, scaling=200, seed=seed),
                         seed=seed)

    if edge is not None:
        model.set_edge_restriction(edge, allowed=False)

    return model.run_sequentially(steps, show_progress=True, return_step_statistics=False)


def find_braess(steps):
    unrestricted_result = np.mean([single_run(steps, seed=seed)['travel_time'].mean() for seed in seeds])
    edges = [(v, w) for v, w in create_sioux_falls_network(network_path, costs_path).edges if v != w]

    results = pd.DataFrame(columns=['edge', 'improvement'])

    for index, edge in enumerate(edges):
        print(f'Blocking edge {edge} ({index})')
        pd.concat([results, pd.DataFrame({
            'edge': edge,
            'improvement': unrestricted_result - np.mean(
                [single_run(steps, edge=edge, seed=seed)['travel_time'].mean() for seed in seeds])
        })])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=False, default='results')
    args = parser.parse_args()
    print(f'Started braess search!!')

    save_or_extend_dataframe(find_braess(300), f'{args.name}.pkl')
