# Contains methods to build a traffic network as a networkx graph with additional
# attributes.

# The edges of the traffic network have the following attributes:
# (anticipated adds +1 to flow for the subsequent car)
# - flow: Number of cars currently using the edge
# - latency_params: Parameters (a, b, c, d) of the edges' latency function
# - latency_fn: lambda n: a + b * n ** c
# - latency: Current latency value (=latency_fn(flow))
# - toll: Additional cost to use the edge as defined by delta-tolling

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
    # Add self-loops
    network.add_edges_from(
        [(v, v) for v in network.nodes if (v, v) not in network],
        latency_params=(1, 0, 1, 1),
    )

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


def create_cars(network, car_counts):
    cars = {}  # Dictionary of final cars
    # Dictionary storing counts of cars on each edge for the different goals
    car_distribution = {edge: [0] * len(car_counts) for edge in network.edges}
    # Iterate through (start, goal)-combinations
    for goal_index, ((s, t), count) in enumerate(car_counts.items()):
        # and get their feasible edges and corresponding latencies
        feasible_edges = []
        for path in nx.all_simple_edge_paths(network, s, t):
            feasible_edges.extend(edge for edge in path if edge not in feasible_edges)
        latencies = [network.edges[edge]["latency_fn"](0) for edge in feasible_edges]

        # Randomly distribute the cars on the edges weighted by the latency
        edge_cars = np.random.choice(
            range(len(feasible_edges)), count, p=np.array(latencies) / sum(latencies)
        )

        # Collect the counts for each goal in a separate dictionary
        if len(edge_cars) > 0:
            for car in edge_cars:
                car_distribution[feasible_edges[car]][goal_index] += 1

    # Iterate through all edges with the counts for all goals
    for edge, remaining_cars in car_distribution.items():
        cars_on_edge = 0
        # and distribute them in a random order on the edge
        while np.sum(remaining_cars) != 0:
            choice_of_goal = np.random.choice(
                list(range(len(remaining_cars))),
                p=remaining_cars / np.sum(remaining_cars),
            )
            cars_on_edge += 1
            cars[len(cars)] = Car(
                len(cars),
                list(car_counts.keys())[choice_of_goal][0],
                list(car_counts.keys())[choice_of_goal][1],
                speed=1 / network.edges[edge]["latency_fn"](cars_on_edge),
                position=(edge, 0.0),
            )
            remaining_cars[choice_of_goal] -= 1
    return cars


class LatencyGenerator:
    def __init__(self, *, seed=42) -> None:
        self.rng = np.random.default_rng(seed)


class ListLatencyGenerator(LatencyGenerator):
    def __init__(self, possible_params, *, seed=42):
        super().__init__(seed=seed)
        self.possible_params = possible_params

    def __call__(self):
        return self.rng.choice(self.possible_params)


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
        return self.rng.choice([(1, 1, self.capacity, 1), (2, 0, 1, 1)], p=[self.q, 1.0 - self.q])


def create_random_grid_network(
    number_of_rows, number_of_columns, *, latency_generator, p=0.5, seed=42
):
    rng = np.random.default_rng(seed)

    network = nx.grid_2d_graph(
        number_of_rows, number_of_columns, create_using=nx.DiGraph
    )

    nx.set_node_attributes(
        network,
        {
            (i, j): (j / number_of_columns, 1 - (i / number_of_rows))
            for i, j in network.nodes
        },
        "position",
    )

    network.remove_edges_from(
        [
            edge
            for edge, r in zip(
                network.edges, rng.uniform(size=len(network.edges))
            )
            if r > p
        ]
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
        nx.circular_layout(network, dim=3, scale=10),
        "position",
    )

    nx.set_edge_attributes(
        network,
        {edge: latency_generator() for edge in network.edges},
        "latency_params",
    )

    return build_network(network)
