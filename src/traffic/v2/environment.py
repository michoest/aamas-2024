import random
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from networkx import get_edge_attributes
from tqdm import trange


class Car:
    def __init__(self, id, source, target, speed=0, created_at_step=0, position=None, verbose=True) -> None:
        self.id = id
        self.source = source
        self.target = target
        self.speed = speed
        self.created_at_step = created_at_step
        self.position = position if position is not None else ((source, source), 1.0)
        self.verbose = verbose

    def __repr__(self) -> str:
        (v, w), p = self.position
        return f'<Car {self.id} ({self.source} -> {self.target}) at {p} of {(v, w)}>'

    def act(self, network, anticipate_all_edges=True):
        current_node = self.position[0][1]
        anticipated_latencies = None
        if current_node == self.target or not nx.has_path(network, current_node, self.target):
            return current_node
        else:
            if anticipate_all_edges:
                choice = random.choice(
                    list(nx.all_shortest_paths(network, current_node, self.target, weight='anticipated_latency_total')))
                self.speed = 1 / (network.edges[(choice[0], choice[1])]['anticipated_latency_total'])
            else:
                anticipated_latencies = {
                    (v, w): attr['latency_fn'](attr['flow'] + (1 if v == self.position[0][1] else 0))
                    for v, w, attr in network.edges(data=True)
                }
                choice = random.choice(
                    list(nx.all_shortest_paths(network, current_node, self.target,
                                               weight=lambda v, w, _: anticipated_latencies[(v, w)])))
                self.speed = 1 / anticipated_latencies[(choice[0], choice[1])]

            if self.verbose:
                print(f'Car {self.id} at {current_node} chooses {choice[1]}.')
                print(f'Latencies: {get_edge_attributes(network, "anticipated_latency_total")}.'
                      if anticipate_all_edges else f'Latencies: {anticipated_latencies}')

            return choice

    def reset(self, source, step):
        self.source = source
        self.created_at_step = step

        self.position = (source, source), 1.0
        self.speed = 0


class TrafficModel:
    def __init__(self, network, cars, enable_tolling=False, beta=0.5, R=0.1,
                 anticipate_all_edges=True, verbose=False) -> None:
        self.network = network
        self.cars = cars
        self.routes = {car_id: [] for car_id in self.cars}
        self.tolling = enable_tolling
        self.beta = beta
        self.R = R
        self.anticipate_all_edges = anticipate_all_edges
        self.verbose = verbose

        self.step_statistics = []
        self.car_statistics = []
        self.current_step = 0
        self._type = 'undefined'

    def update_latencies(self):
        nx.set_edge_attributes(self.network, {
            (v, w): attr['latency_fn'](attr['flow']) for v, w, attr in self.network.edges(data=True)
        }, 'latency')
        if self.anticipate_all_edges:
            nx.set_edge_attributes(self.network, {
                (v, w): attr['latency_fn'](attr['flow'] + 1) for v, w, attr in self.network.edges(data=True)
            }, 'anticipated_latency_total')

    def update_latency(self, edge):
        self.network.edges[edge]['latency'] = self.network.edges[edge]['latency_fn'](self.network.edges[edge]['flow'])
        if self.anticipate_all_edges:
            self.network.edges[edge]['anticipated_latency_total'] = self.network.edges[edge]['latency_fn'](
                self.network.edges[edge]['flow'] + 1)

    @property
    def allowed_network(self):
        return nx.restricted_view(self.network, [],
                                  [(v, w) for v, w, allowed in self.network.edges(data='allowed') if not allowed])

    def decrease_flow(self, edge):
        self.set_flow(edge, self.network.edges[edge]['flow'] - 1)

    def increase_flow(self, edge):
        self.set_flow(edge, self.network.edges[edge]['flow'] + 1)

    def set_flow(self, edge, flow):
        self.network.edges[edge]['flow'] = flow
        self.update_total_cost(edge)

    def set_edge_restriction(self, edge, allowed=True):
        self.network.edges[edge]["allowed"] = allowed

    def update_toll(self, edge):
        new_toll = self.beta * (self.network.edges[edge]['latency'] - self.network.edges[edge]['latency_fn'](0))
        self.network.edges[edge]['total_cost'] = self.R * new_toll + (1 - self.R) * self.network.edges[edge]["toll"]

    def update_tolls(self):
        new_toll = {
            (v, w): self.beta * (attr["latency"] - attr["latency_fn"](0))
            for v, w, attr in self.network.edges(data=True)
        }
        nx.set_edge_attributes(
            self.network,
            {
                (v, w): self.R * new_toll[(v, w)] + (1 - self.R) * attr["toll"]
                for v, w, attr in self.network.edges(data=True)
            },
            "toll",
        )

    def update_total_cost(self, edge):
        self.update_latency(edge)
        if self.tolling:
            self.update_toll(edge)
            self.network.edges[edge]['total_cost'] = \
                self.network.edges[edge]['latency'] + self.network.edges[edge]['toll']
        else:
            self.network.edges[edge]['total_cost'] = self.network.edges[edge]['latency']

    def update_total_costs(self):
        self.update_latencies()
        if self.tolling:
            self.update_tolls()
            nx.set_edge_attributes(
                self.network,
                {
                    (v, w): attr["latency"] + attr["toll"]
                    for v, w, attr in self.network.edges(data=True)
                },
                "total_cost",
            )
        else:
            nx.set_edge_attributes(self.network, get_edge_attributes(self.network, 'latency'), 'total_cost')

    def run_sequentially(self, number_of_steps):
        assert self._type in ['undefined', 'sequentially'], 'Cannot proceed sequentially from a single step model'
        self._type = 'sequentially'

        routes_taken = {car.id: [car.position[0][0], car.position[0][1]] for car in self.cars.values()}
        for step in range(number_of_steps):
            if self.verbose:
                print(f'Step {step}:')
                print(f'Positions before step {step}: {[car.position for car in self.cars.values()]}')

            # Update flow and latencies
            flow_counter = Counter(car.position[0] for car in self.cars.values())
            nx.set_edge_attributes(self.network, {edge: flow_counter[edge] for edge in self.network.edges}, 'flow')
            self.update_latencies()

            self.step_statistics.append(
                list(self.routes.values())
                + list(nx.get_edge_attributes(self.network, "flow").values())
                + list(nx.get_edge_attributes(self.network, "latency").values())
            )

            if self.verbose:
                print(self)

            # Advance agents
            for car in self.cars.values():
                car.position = (car.position[0], min(car.position[1] + car.speed, 1.0))

                if car.position[1] >= 1.0:
                    self.decrease_flow(car.position[0])

            if self.verbose:
                print(f'Positions after step {step}: {[car.position for car in self.cars.values()]}')

            if self.verbose:
                print(self)

            # Re-spawn cars which have arrived
            for car in self.cars.values():
                if car.position[0][0] == car.target or (car.position[0][1] == car.target and car.position[1] == 1.0):
                    self.car_statistics.append(
                        {
                            "step": step,
                            "car_id": car.id,
                            "source": car.source,
                            "target": car.target,
                            "route": tuple(routes_taken[car.id]),
                            "travel_time": step - car.created_at_step,
                        }
                    )

                    if self.verbose:
                        print(f'Car {car.id} reached its target after {step - car.created_at_step} steps.')

                    # source = np.random.randint(3)
                    car.reset(car.source, step)
                    routes_taken[car.id] = [car.source]

            if self.verbose:
                print(f'Positions after re-spawning at step {step}: {[car.position for car in self.cars.values()]}')

            # Let agents make decisions
            for car in np.random.permutation([car for car in self.cars.values() if car.position[1] == 1.0]):
                self.routes[car.id] = car.act(self.allowed_network, anticipate_all_edges=self.anticipate_all_edges)
                car.position = (car.position[0][1], self.routes[car.id][1]), 0.0
                self.increase_flow(car.position[0])
                routes_taken[car.id].append(car.position[0][1])

            if self.verbose:
                print(self)

        return pd.DataFrame(
            self.step_statistics,
            columns=pd.MultiIndex.from_tuples(
                [("route", car_id) for car_id in self.cars]
                + [("flow", edge_id) for edge_id in self.network.edges]
                + [("latency", edge_id) for edge_id in self.network.edges]
            ),
        ), pd.DataFrame(self.car_statistics)

    def run_single_steps(self, number_of_steps):
        assert self._type in ['undefined', 'single_steps'], 'Cannot proceed in single steps from a sequential model'
        self._type = 'single_steps'

        self.update_latencies()
        for step in (range if self.verbose else trange)(number_of_steps):
            if self.verbose:
                print(self)

            for car in np.random.permutation(list(self.cars.values())):
                for edge in zip(self.routes[car.id], self.routes[car.id][1:]):
                    self.decrease_flow(edge)

                # Let agents choose a route, given the network with allowed edges only
                self.routes[car.id] = car.act(self.allowed_network, anticipate_all_edges=self.anticipate_all_edges)

                for edge in zip(self.routes[car.id], self.routes[car.id][1:]):
                    self.increase_flow(edge)

            self.step_statistics.append(
                list(self.routes.values())
                + list(nx.get_edge_attributes(self.network, "latency").values())
                + [
                    nx.path_weight(self.network, route, "latency")
                    for car_id, route in self.routes.items()
                ]
            )

            for car in self.cars.values():
                self.car_statistics.append(
                    {
                        "step": step,
                        "car_id": car.id,
                        "source": car.source,
                        "target": car.target,
                        "route": tuple(self.routes[car.id]),
                        "travel_time": nx.path_weight(self.network, self.routes[car.id], "latency"),
                    }
                )

        return pd.DataFrame(
            self.step_statistics,
            columns=pd.MultiIndex.from_tuples(
                [("route", car_id) for car_id in self.cars]
                + [("latency", car_id) for car_id in self.network.edges]
                + [("travel_time", car_id) for car_id in self.cars]
            ),
        ), pd.DataFrame(self.car_statistics)

    def __repr__(self):
        print(f'Current step: {self.current_step}')
        print(f'{nx.get_edge_attributes(self.network, "flow")=}')
        print(f'{nx.get_edge_attributes(self.network, "latency")=}')
        print(f'{nx.get_edge_attributes(self.network, "anticipated_latency")=}')
