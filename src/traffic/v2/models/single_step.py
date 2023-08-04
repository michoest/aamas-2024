import random

import networkx as nx


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
