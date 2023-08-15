def sort_edges_by_latency_increase(model, step_stats):
    free_flow_latencies = {
        (v, w): attr["latency_fn"](0) for v, w, attr in model.network.edges(data=True)
    }
    mean_latencies = step_stats["latency"].mean().to_dict()

    # Note that free flow latencies are always > 0 in the BPR paradigm
    latency_increase = {
        edge: (mean_latencies[edge] / free_flow_latencies[edge]) - 1.0
        for edge in model.network.edges
    }

    return sorted(latency_increase, key=latency_increase.get, reverse=True)
