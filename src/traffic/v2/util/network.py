# Contains methods to build a traffic network as a networkx graph with additional
# attributes.

# The edges of the traffic network have the following attributes:
# - flow: Number of cars currently using the edge
# - latency_params: Parameters (a, b, c) of the edges' latency function
# - latency_fn: lambda n: a + b * n ** c
# - latency: Current latency value (=latency_fn(flow))
# - toll: Additional cost to use the edge as defined by delta-tolling
# - total_cost: Sum of latency and toll

import random

import networkx as nx


def build_network(network):
    # Create latency functions from parameters
    nx.set_edge_attributes(
        network,
        {
            (v, w): lambda u, a=attr["latency_params"][0], b=attr["latency_params"][
                1
            ], c=attr["latency_params"][2]: a
            + b * u**c
            for v, w, attr in network.edges(data=True)
        },
        "latency_fn",
    )

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

    # Set initial total cost
    nx.set_edge_attributes(
        network,
        {
            (v, w): attr["latency"] + attr["toll"]
            for v, w, attr in network.edges(data=True)
        },
        "total_cost",
    )

    return network


def create_braess_network():
    network = nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.set_node_attributes(
        network, {i: (i / 3, 0) for i in range(len(network.nodes))}, "position"
    )

    nx.set_edge_attributes(
        network,
        {
            (0, 1): (1, 8, 1),
            (0, 2): (11, 0, 1),
            (1, 2): (1, 0, 1),
            (1, 3): (11, 0, 1),
            (2, 3): (1, 8, 1),
        },
        "latency_params",
    )

    return build_network(network)


def create_double_braess_network():
    network = nx.DiGraph([('A', 0), ('A', 2), (0, 1), (0, 2), (0, 'B'), (1, 2), (1, 3), (2, 3), (2, 'B')])

    nx.set_edge_attributes(network, {
        ('A', 0): (0, 11, 1),
        ('A', 2): (121, 0, 1),
        (0, 1): (0, 1, 1),
        (0, 2): (11, 0, 1),
        (0, 'B'): (121, 0, 1),
        (1, 2): (1, 0, 1),
        (1, 3): (11, 0, 1),
        (2, 3): (0, 1, 1),
        (2, 'B'): (0, 11, 1)
    }, 'latency_params')

    return build_network(network)


def create_random_grid_network(number_of_rows, number_of_columns):
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
        {
            edge: random.choice([(1, 0, 1), (11, 0, 1), (0, 8, 1)])
            for edge in network.edges
        },
        "latency_params",
    )

    return build_network(network)
