import random

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import trange


class Car:
    def __init__(self, source, target) -> None:
        self.source = source
        self.target = target

    def __repr__(self) -> str:
        return f"<Car {self.source} -> {self.target}>"

    def act(self, network, verbose=False):
        if self.source == self.target or not nx.has_path(
            network, self.source, self.target
        ):
            return self.source
        else:
            choices = list(
                nx.all_shortest_paths(
                    network, self.source, self.target, weight="anticipated_latency"
                )
            )
            choice = random.choice(choices)

            if verbose:
                print(f"Choose {choice} from {choices}.")

            return choice


class TrafficModel:
    def __init__(self, network, cars) -> None:
        self.network = network
        self.cars = cars

    def update_latencies(self):
        nx.set_edge_attributes(
            self.network,
            {
                (v, w): attr["latency_fn"](attr["utilization"])
                for v, w, attr in self.network.edges(data=True)
            },
            "latency",
        )
        nx.set_edge_attributes(
            self.network,
            {
                (v, w): attr["latency_fn"](attr["utilization"] + (1 / len(self.cars)))
                for v, w, attr in self.network.edges(data=True)
            },
            "anticipated_latency",
        )

    def update_latency(self, edge):
        self.network.edges[edge]["latency"] = self.network.edges[edge]["latency_fn"](
            self.network.edges[edge]["utilization"]
        )
        self.network.edges[edge]["anticipated_latency"] = self.network.edges[edge][
            "latency_fn"
        ](self.network.edges[edge]["utilization"] + (1 / len(self.cars)))

    @property
    def allowed_network(self):
        return nx.restricted_view(
            self.network,
            [],
            [
                (v, w)
                for v, w, allowed in self.network.edges(data="allowed")
                if not allowed
            ],
        )

    def decrease_utilization(self, edge):
        self.set_utilization(
            edge, self.network.edges[edge]["utilization"] - (1 / len(self.cars))
        )

    def increase_utilization(self, edge):
        self.set_utilization(
            edge, self.network.edges[edge]["utilization"] + (1 / len(self.cars))
        )

    def set_utilization(self, edge, utilization):
        self.network.edges[edge]["utilization"] = utilization
        self.update_latency(edge)

    def set_edge_restriction(self, edge, allowed=True):
        self.network.edges[edge]["allowed"] = allowed


def run(model, number_of_steps, verbose=False):
    routes = {id: [] for id in model.cars}

    nx.set_edge_attributes(model.network, 0, "utilization")
    model.update_latencies()

    step_stats = []
    car_stats = []
    for step in (range if verbose else trange)(number_of_steps):
        if verbose:
            print(f"Step {step}:")
            print(
                f'Initial latency = {nx.get_edge_attributes(model.network, "latency")}'
            )

        for id, car in np.random.permutation(list(model.cars.items())):
            for edge in zip(routes[id], routes[id][1:]):
                model.decrease_utilization(edge)

            # Let agents choose a route, given the network with allowed edges only
            routes[id] = car.act(model.allowed_network, verbose=verbose)

            for edge in zip(routes[id], routes[id][1:]):
                model.increase_utilization(edge)

        step_stats.append(
            list(routes.values())
            + list(nx.get_edge_attributes(model.network, "latency").values())
            + [
                nx.path_weight(model.network, route, "latency")
                for car_id, route in routes.items()
            ]
        )

        for id, car in model.cars.items():
            car_stats.append(
                {
                    "step": step,
                    "car_id": id,
                    "source": car.source,
                    "target": car.target,
                    "route": tuple(routes[id]),
                    "travel_time": nx.path_weight(model.network, routes[id], "latency"),
                }
            )

    return pd.DataFrame(
        step_stats,
        columns=pd.MultiIndex.from_tuples(
            [("route", car_id) for car_id in model.cars]
            + [("latency", car_id) for car_id in model.network.edges]
            + [("travel_time", car_id) for car_id in model.cars]
        ),
    ), pd.DataFrame(car_stats)
