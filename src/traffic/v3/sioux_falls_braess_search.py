import argparse

import pandas as pd

from src.traffic.v3.environment import TrafficModel
from src.traffic.v3.util.network import create_sioux_falls_network, create_sioux_falls_cars
from src.traffic.v3.util.utils import save_or_extend_dataframe


network_path = 'src/traffic/v3/data/SiouxFalls_net.tntp'
costs_path = 'src/traffic/v3/data/sf1_c.txt'
cars_path = 'src/traffic/v3/data/sf1_dem.txt'


def single_run(steps=2000, edge=None):
    network = create_sioux_falls_network(network_path, costs_path)
    cars = create_sioux_falls_cars(cars_path, network)

    model = TrafficModel(network, cars)

    if edge is not None:
        model.set_edge_restriction(edge, allowed=False)

    return model.run_sequentially(steps)[1]


def find_braess(steps):
    # Initialize dataframe with unrestricted performance
    car_statistics = single_run(steps)
    combined_results = car_statistics[car_statistics['step'] >= steps / 2].groupby(
        ['source', 'target'])['travel_time'].mean().reset_index()
    combined_results.columns = ['source', 'target', 'unrestricted']

    # Block each edge and measure effect
    for index, edge in enumerate(create_sioux_falls_network(network_path, costs_path).edges):
        if edge[0] != edge[1]:
            print(f'Blocking edge {edge} ({index})')
            car_statistics = single_run(steps, edge)
            combined_results = pd.merge(combined_results,
                                        car_statistics[car_statistics['step'] >= steps / 2]
                                        .groupby(['source', 'target'])['travel_time'].mean().reset_index(),
                                        how='outer', left_on=['source', 'target'], right_on=['source', 'target'])
            combined_results.columns = list(combined_results.columns)[:-1] + [edge]
        else:
            print(f'Skipping self-edge {edge}')

        print(combined_results.head())

    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--name', type=str, required=False, default='results')
    args = parser.parse_args()
    print(f'Starting braess search with {args.steps} steps!!')

    save_or_extend_dataframe(find_braess(args.steps), f'{args.name}.pkl')
