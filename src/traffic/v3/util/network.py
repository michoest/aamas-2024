# Contains methods to build a traffic network as a networkx graph with additional
# attributes.

# The edges of the traffic network have the following attributes:
# (anticipated adds +1 to flow for the subsequent car)
# - flow: Number of cars currently using the edge
# - latency_params: Parameters (a, b, c, d) of the edges' latency function
# - latency_fn: lambda n: a + b * n ** c
# - latency: Current latency value (=latency_fn(flow))
# - toll: Additional cost to use the edge as defined by delta-tolling
# - total_cost: Sum of latency and toll

import random
import math

import networkx as nx
import numpy as np

from environment import Car


def update_latency_functions(network):
    nx.set_edge_attributes(
        network,
        {
            (v, w): lambda n, params=attr["latency_params"]: params[0]
            + params[1] * (n / params[2]) ** params[3]
            for v, w, attr in network.edges(data=True)
        },
        "latency_fn",
    )


def build_network(network):
    # Create latency functions from parameters: l(n) = a + b * (n / c) ** d
    update_latency_functions(network)

    # Set initial utilization
    nx.set_edge_attributes(network, 0, "flow")

    # Set initial latency based on utilization
    nx.set_edge_attributes(
        network,
        {
            (v, w): attr["latency_fn"](attr["flow"])
            for v, w, attr in network.edges(data=True)
        },
        "latency",
    )

    # Allow all edges
    nx.set_edge_attributes(network, True, "allowed")

    # Set tolls to zero
    nx.set_edge_attributes(network, 0.0, "toll")
    nx.set_edge_attributes(network, 0.0, "anticipated_toll")

    return network


def create_braess_network(capacity=100):
    network = nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.set_node_attributes(
        network, {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}, "position"
    )

    nx.set_edge_attributes(
        network,
        {
            (0, 1): (2, 6, capacity, 1),
            (0, 2): (10, 0, 1, 1),
            (1, 2): (1, 0, 1, 1),
            (1, 3): (10, 0, 1, 1),
            (2, 3): (2, 6, capacity, 1),
        },
        "latency_params",
    )

    return build_network(network)


def create_double_braess_network(capacity=100):
    network = nx.DiGraph(
        [("A", 0), ("A", 2), (0, 1), (0, 2), (0, "B"), (1, 2), (1, 3), (2, 3), (2, "B")]
    )

    nx.set_node_attributes(
        network,
        {"A": (0, -1), 0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), "B": (1, -1)},
        "position",
    )

    nx.set_edge_attributes(
        network,
        {
            ("A", 0): (2, 6, capacity, 1),
            ("A", 2): (19, 0, 1, 1),
            (0, 1): (2, 6, capacity, 1),
            (0, 2): (10, 0, 1, 1),
            (0, "B"): (19, 0, 1, 1),
            (1, 2): (1, 0, 1, 1),
            (1, 3): (10, 0, 1, 1),
            (2, 3): (2, 6, capacity, 1),
            (2, "B"): (2, 6, capacity, 1),
        },
        "latency_params",
    )

    return build_network(network)


def create_cars(
    network, cars_per_edge: int = 3, goal_distribution=None, seed: int = 42
):
    if goal_distribution is None:
        goal_distribution = {1.0: (0, 3)}

    rng = np.random.RandomState(seed)
    created_cars = {}
    for v, w, attr in network.edges(data=True):
        for car in range(cars_per_edge):
            # Draw source and target
            s_t = list(goal_distribution.values())[
                rng.choice(
                    len(goal_distribution.values()), p=list(goal_distribution.keys())
                )
            ]
            # Set cars on the edge evenly spaces
            progress = (1 / attr["latency_fn"](cars_per_edge)) * car
            # Add to the list of cars
            created_cars[len(created_cars) + car] = Car(
                len(created_cars) + car,
                s_t[0],
                s_t[1],
                1 / attr["latency_fn"](cars_per_edge),
                position=((v, w), progress if progress < 1.0 else 1.0),
            )

    return created_cars


class LatencyGenerator:
    def __init__(self, *, seed=42) -> None:
        self.rng = np.random.RandomState(seed)
        random.seed(seed)


class ListLatencyGenerator(LatencyGenerator):
    def __init__(self, possible_params, *, seed=42):
        super().__init__(seed=seed)
        self.possible_params = possible_params

    def __call__(self):
        return random.choice(self.possible_params)


class UniformLatencyGenerator(LatencyGenerator):
    def __init__(
        self,
        a_min,
        a_max,
        b_min,
        b_max,
        c_min=1,
        c_max=1,
        d_min=1,
        d_max=1,
        *,
        integer=False,
        seed=42
    ):
        super().__init__(seed=seed)
        (
            self.a_min,
            self.a_max,
            self.b_min,
            self.b_max,
            self.c_min,
            self.c_max,
            self.d_min,
            self.d_max,
        ) = (a_min, a_max, b_min, b_max, c_min, c_max, d_min, d_max)
        self.integer = integer

    def __call__(self):
        if self.integer:
            return tuple(
                self.rng.randint(
                    low=[self.a_min, self.b_min, self.c_min, self.d_min],
                    high=[
                        self.a_max + 1,
                        self.b_max + 1,
                        self.c_max + 1,
                        self.d_max + 1,
                    ],
                )
            )
        else:
            return tuple(
                self.rng.uniform(
                    low=[self.a_min, self.b_min, self.c_min, self.d_min],
                    high=[self.a_max, self.b_max, self.c_max, self.d_max],
                )
            )


class OneXLatencyGenerator(LatencyGenerator):
    """Selects randomly one of two latency functions:
    - l(n) = 1 + n / c
    - l(n) = 2
    Using these values, it is always beneficial to choose a variable-latency edge over
    a constant-latency edge (as in the original Braess Paradox).
    """

    def __init__(self, *, q=0.5, capacity=100, seed=42):
        super().__init__(seed=seed)
        self.q = q
        self.capacity = capacity

    def __call__(self):
        return random.choices(
            [(1, 1, self.capacity, 1), (2, 0, 1, 1)],
            weights=[self.q, 1.0 - self.q],
        )[0]


def create_random_grid_network(number_of_rows, number_of_columns, latency_generator):
    network = nx.grid_2d_graph(
        number_of_rows, number_of_columns, create_using=nx.DiGraph
    )

    nx.set_node_attributes(
        network,
        {
            (i, j): (j / (number_of_columns - 1), 1 - (i / (number_of_rows - 1)))
            for i, j in network.nodes
        },
        "position",
    )
    nx.set_edge_attributes(
        network,
        {edge: latency_generator() for edge in network.edges},
        "latency_params",
    )

    return build_network(network)


def create_random_gnp_graph(number_of_nodes, p, latency_generator, *, seed=42):
    network = nx.gnp_random_graph(number_of_nodes, p, seed=seed, directed=True)

    nx.set_node_attributes(
        network,
        nx.circular_layout(network),
        "position",
    )

    nx.set_edge_attributes(
        network,
        {edge: latency_generator() for edge in network.edges},
        "latency_params",
    )

    return build_network(network)
