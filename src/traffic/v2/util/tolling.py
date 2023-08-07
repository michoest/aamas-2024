import networkx as nx


def update_tolls(network, beta, R):
    new_toll = {
        (v, w): beta * (attr["latency"] - attr["latency_fn"](0))
        for v, w, attr in network.edges(data=True)
    }
    nx.set_edge_attributes(
        network,
        {
            (v, w): R * new_toll[(v, w)] + (1 - R) * attr["toll"]
            for v, w, attr in network.edges(data=True)
        },
        "toll",
    )
