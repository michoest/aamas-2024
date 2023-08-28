# Internal modules
from collections import Counter

# External modules
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import trange


class Car:
    def __init__(
        self,
        id,
        source,
        target,
        speed=0,
        position=None,
        *,
        anticipation_strategy="route",
        created_at_step=0,
        value_of_time=1,
        value_of_money=1,
        verbose=False,
        seed=42
    ) -> None:
        assert anticipation_strategy in [
            "none",
            "edge",
            "route",
        ], "Unknown anticipation strategy"

        self.id = id
        self.source = source
        self.target = target
        self.speed = speed
        self.created_at_step = created_at_step
        self.position = position if position is not None else ((source, source), 1.0)
        self.anticipation_strategy = anticipation_strategy
        self.value_of_time = value_of_time
        self.value_of_money = value_of_money
        self.verbose = verbose
        self.toll = 0.0

        # Initialize random generator
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        (v, w), p = self.position
        return f"<Car {self.id} ({self.source} -> {self.target}) at {p} of {(v, w)}>"

    def act(self, network):
        current_node = self.position[0][1]
        if current_node == self.target or not nx.has_path(
            network, current_node, self.target
        ):
            return [current_node, current_node]
        else:
            # Define edge weights according to anticipation_strategy
            if self.anticipation_strategy == "none":
                # Find the shortest path based on the actual latencies
                latencies = {
                    (v, w): attr["latency"] * self.value_of_time
                    + attr["toll"] * self.value_of_money
                    for v, w, attr in network.edges(data=True)
                }
            elif self.anticipation_strategy == "edge":
                # Find the shortest path based on the actual latencies with only the
                # next edge being anticipated
                latencies = {
                    (v, w): (
                        attr["anticipated_latency"] * self.value_of_time
                        + attr["anticipated_toll"] * self.value_of_money
                        if v == current_node
                        else attr["latency"] * self.value_of_time
                        + attr["toll"] * self.value_of_money
                    )
                    for v, w, attr in network.edges(data=True)
                }
            elif self.anticipation_strategy == "route":
                # Find the shortest path based on the anticipated total cost (including
                # tolls)
                latencies = {
                    (v, w): attr["anticipated_latency"] * self.value_of_time
                    + attr["anticipated_toll"] * self.value_of_money
                    for v, w, attr in network.edges(data=True)
                }
            else:
                raise ValueError()

            # rng.choice cannot handle inhomogeneous shapes, so we need to draw a
            # random number and return the corresponding list element instead
            shortest_routes = list(
                    nx.all_shortest_paths(
                        network,
                        current_node,
                        self.target,
                        weight=lambda v, w, _: latencies[(v, w)],
                    )
                )
            chosen_route = shortest_routes[self.rng.choice(len(shortest_routes))]
            chosen_edge = chosen_route[:2]

            # Always use anticipated attributes for speed and toll
            self.speed = 1 / (network.edges[chosen_edge]["anticipated_latency"])
            self.toll += network.edges[chosen_edge]["anticipated_toll"]

            if self.verbose:
                print(f"Latencies for routing decision: {latencies}.")
                print(f"Car {self.id} at {current_node} chooses {chosen_route[1]}.")

            return chosen_route

    def reset(self, source, target, step):
        self.source = source
        self.target = target
        self.created_at_step = step

        self.position = (source, source), 1.0
        self.speed = 0

        self.toll = 0.0


class TrafficModel:
    def __init__(
        self, network, cars, *, tolls=False, beta=0.5, R=0.1, verbose=False, seed=42
    ) -> None:
        self.network = network
        self.cars = cars
        self.tolls = tolls
        self.beta = beta
        self.R = R
        self.verbose = verbose

        self._type = "undefined"
        self.routes = {car_id: [] for car_id in self.cars}
        self.step_statistics = []
        self.car_statistics = []
        self.current_step = 0

        # Initialize random generator
        self.rng = np.random.default_rng(seed)

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

    def set_edge_restriction(self, edge, allowed=True):
        self.network.edges[edge]["allowed"] = allowed

    def decrease_flow(self, edge):
        self.network.edges[edge]["flow"] = self.network.edges[edge]["flow"] - 1
        self.update_edge_attributes(edge)

    def increase_flow(self, edge):
        self.network.edges[edge]["flow"] = self.network.edges[edge]["flow"] + 1
        self.update_edge_attributes(edge)

    def update_edge_attributes(self, edge):
        # Update latency
        self.network.edges[edge]["latency"] = self.network.edges[edge]["latency_fn"](
            self.network.edges[edge]["flow"]
        )
        self.network.edges[edge]["anticipated_latency"] = self.network.edges[edge][
            "latency_fn"
        ](self.network.edges[edge]["flow"] + 1)

        # Update toll according to delta-tolling algorithm
        if self.tolls:
            new_toll = self.beta * (
                self.network.edges[edge]["latency"]
                - self.network.edges[edge]["latency_fn"](0)
            )
            self.network.edges[edge]["toll"] = (
                self.R * new_toll + (1 - self.R) * self.network.edges[edge]["toll"]
            )
            new_anticipated_toll = self.beta * (
                self.network.edges[edge]["anticipated_latency"]
                - self.network.edges[edge]["latency_fn"](0)
            )
            self.network.edges[edge]["anticipated_toll"] = (
                self.R * new_anticipated_toll
                + (1 - self.R) * self.network.edges[edge]["anticipated_toll"]
            )

    def update_network_attributes(self):
        # Update latencies
        nx.set_edge_attributes(
            self.network,
            {
                (v, w): attr["latency_fn"](attr["flow"])
                for v, w, attr in self.network.edges(data=True)
            },
            "latency",
        )
        nx.set_edge_attributes(
            self.network,
            {
                (v, w): attr["latency_fn"](attr["flow"] + 1)
                for v, w, attr in self.network.edges(data=True)
            },
            "anticipated_latency",
        )

        # Update tolls according to delta-tolling algorithm
        if self.tolls:
            new_tolls = {
                (v, w): self.beta * (attr["latency"] - attr["latency_fn"](0))
                for v, w, attr in self.network.edges(data=True)
            }
            nx.set_edge_attributes(
                self.network,
                {
                    (v, w): self.R * new_tolls[(v, w)] + (1 - self.R) * attr["toll"]
                    for v, w, attr in self.network.edges(data=True)
                },
                "toll",
            )
            new_anticipated_tolls = {
                (v, w): self.beta
                * (attr["anticipated_latency"] - attr["latency_fn"](0))
                for v, w, attr in self.network.edges(data=True)
            }
            nx.set_edge_attributes(
                self.network,
                {
                    (v, w): self.R * new_anticipated_tolls[(v, w)]
                    + (1 - self.R) * attr["anticipated_toll"]
                    for v, w, attr in self.network.edges(data=True)
                },
                "anticipated_toll",
            )

    def run_sequentially(self, number_of_steps, *, show_progress=True):
        assert self._type in [
            "undefined",
            "sequentially",
        ], "Cannot proceed sequentially from a single step model"
        self._type = "sequentially"

        routes_taken = {
            car.id: [car.position[0][0], car.position[0][1]]
            for car in self.cars.values()
        }
        for step in (trange if show_progress else range)(number_of_steps):
            if self.verbose:
                print(f"Step {step}:")
                print(
                    f"Positions before step {step}: {[car.position for car in self.cars.values()]}"
                )

            # Update flow and latencies
            flow_counter = Counter(car.position[0] for car in self.cars.values())
            nx.set_edge_attributes(
                self.network,
                {edge: flow_counter[edge] for edge in self.network.edges},
                "flow",
            )
            self.update_network_attributes()

            self.step_statistics.append(
                list(self.routes.values())
                + list(nx.get_edge_attributes(self.network, "flow").values())
                + list(nx.get_edge_attributes(self.network, "latency").values())
                + list(nx.get_edge_attributes(self.network, "toll").values())
            )

            # if self.verbose:
            #     print(self)

            # Advance agents
            for car in self.cars.values():
                car.position = (car.position[0], min(car.position[1] + car.speed, 1.0))

                if car.position[1] >= 1.0:
                    self.decrease_flow(car.position[0])

            if self.verbose:
                print(
                    f"Positions after step {step}: {[car.position for car in self.cars.values()]}"
                )

            # if self.verbose:
            #     print(self)

            # Re-spawn cars which have arrived
            for car in self.cars.values():
                if car.position[0][0] == car.target or (
                    car.position[0][1] == car.target and car.position[1] == 1.0
                ):
                    self.car_statistics.append(
                        {
                            "step": step,
                            "car_id": car.id,
                            "value_of_time": car.value_of_time,
                            "value_of_money": car.value_of_money,
                            "source": car.source,
                            "target": car.target,
                            "route": tuple(routes_taken[car.id]),
                            "travel_time": step - car.created_at_step,
                            "toll": car.toll,
                            "total_cost": (step - car.created_at_step)
                            * car.value_of_time
                            + car.toll * car.value_of_money,
                        }
                    )

                    if self.verbose:
                        print(
                            f"Car {car.id} reached its target after {step - car.created_at_step} steps."
                        )

                    car.reset(car.source, car.target, step)
                    routes_taken[car.id] = [car.source]

            if self.verbose:
                print(
                    f"Positions after re-spawning at step {step}: {[car.position for car in self.cars.values()]}"
                )

            # Let agents make decisions
            for car in self.rng.permutation(
                [car for car in self.cars.values() if car.position[1] == 1.0]
            ):
                self.routes[car.id] = car.act(self.allowed_network)
                car.position = (car.position[0][1], self.routes[car.id][1]), 0.0
                self.increase_flow(car.position[0])
                routes_taken[car.id].append(car.position[0][1])

            # if self.verbose:
            #     print(self)

        return pd.DataFrame(
            self.step_statistics,
            columns=pd.MultiIndex.from_tuples(
                [("route", car_id) for car_id in self.cars]
                + [("flow", edge_id) for edge_id in self.network.edges]
                + [("latency", edge_id) for edge_id in self.network.edges]
                + [("toll", edge_id) for edge_id in self.network.edges]
            ),
        ), pd.DataFrame(self.car_statistics)

    def run_single_steps(self, number_of_steps):
        assert self._type in [
            "undefined",
            "single_steps",
        ], "Cannot proceed in single steps from a sequential model"
        self._type = "single_steps"

        self.update_network_attributes()
        for step in (range if self.verbose else trange)(number_of_steps):
            if self.verbose:
                print(self)

            for car in self.rng.permutation(list(self.cars.values())):
                for edge in zip(self.routes[car.id], self.routes[car.id][1:]):
                    self.decrease_flow(edge)

                # Let agents choose a route, given the network with allowed edges only
                self.routes[car.id] = car.act(self.allowed_network)

                for edge in zip(self.routes[car.id], self.routes[car.id][1:]):
                    self.increase_flow(edge)

            self.step_statistics.append(
                list(self.routes.values())
                + list(nx.get_edge_attributes(self.network, "flow").values())
                + list(nx.get_edge_attributes(self.network, "latency").values())
                + list(nx.get_edge_attributes(self.network, "toll").values())
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
                        "travel_time": nx.path_weight(
                            self.network, self.routes[car.id], "latency"
                        ),
                    }
                )

        return pd.DataFrame(
            self.step_statistics,
            columns=pd.MultiIndex.from_tuples(
                [("route", car_id) for car_id in self.cars]
                + [("flow", edge) for edge in self.network.edges]
                + [("latency", edge) for edge in self.network.edges]
                + [("toll", edge) for edge in self.network.edges]
                + [("travel_time", car_id) for car_id in self.cars]
            ),
        ), pd.DataFrame(self.car_statistics)

    def __repr__(self):
        print(f"Current step: {self.current_step}")
        print(f'{nx.get_edge_attributes(self.network, "flow")=}')
        print(f'{nx.get_edge_attributes(self.network, "latency")=}')
        print(f'{nx.get_edge_attributes(self.network, "anticipated_latency")=}')

    def run_sequentially_with_phases(self, phases, *, show_progress=True):
        assert self._type in [
            "undefined",
            "sequentially",
        ], "Cannot proceed sequentially from a single step model"
        self._type = "sequentially"

        routes_taken = {
            car.id: [car.position[0][0], car.position[0][1]]
            for car in self.cars.values()
        }

        cumulated_step = 0
        for car_counts, number_of_steps in phases:
            print(f'Running demand {car_counts} for {number_of_steps} steps...')
            demand = [item for s_t, count in car_counts.items() for item in [s_t] * count]
            self.rng.shuffle(demand)
            demand = dict(enumerate(demand))

            for phase_step in range(number_of_steps):
                # Update flow and latencies
                flow_counter = Counter(car.position[0] for car in self.cars.values())
                nx.set_edge_attributes(
                    self.network,
                    {edge: flow_counter[edge] for edge in self.network.edges},
                    "flow",
                )
                self.update_network_attributes()

                self.step_statistics.append(
                    list(self.routes.values())
                    + list(nx.get_edge_attributes(self.network, "flow").values())
                    + list(nx.get_edge_attributes(self.network, "latency").values())
                    + list(nx.get_edge_attributes(self.network, "toll").values())
                )

                # Advance agents
                for car in self.cars.values():
                    car.position = (car.position[0], min(car.position[1] + car.speed, 1.0))

                    if car.position[1] >= 1.0:
                        self.decrease_flow(car.position[0])

                # Re-spawn cars which have arrived
                for car in self.cars.values():
                    if car.position[0][0] == car.target or (
                        car.position[0][1] == car.target and car.position[1] == 1.0
                    ):
                        self.car_statistics.append(
                            {
                                "step": cumulated_step,
                                "car_id": car.id,
                                "value_of_time": car.value_of_time,
                                "value_of_money": car.value_of_money,
                                "source": car.source,
                                "target": car.target,
                                "route": tuple(routes_taken[car.id]),
                                "travel_time": cumulated_step - car.created_at_step,
                                "toll": car.toll,
                                "total_cost": (cumulated_step - car.created_at_step)
                                * car.value_of_time
                                + car.toll * car.value_of_money,
                            }
                        )

                        if self.verbose:
                            print(
                                f"Car {car.id} reached its target after {cumulated_step - car.created_at_step} steps."
                            )

                        car.reset(*demand[car.id], cumulated_step)
                        routes_taken[car.id] = [car.source]

                if self.verbose:
                    print(
                        f"Positions after re-spawning at step {cumulated_step}: {[car.position for car in self.cars.values()]}"
                    )

                # Let agents make decisions
                for car in self.rng.permutation(
                    [car for car in self.cars.values() if car.position[1] == 1.0]
                ):
                    self.routes[car.id] = car.act(self.allowed_network)

                    car.position = (car.position[0][1], self.routes[car.id][1]), 0.0
                    self.increase_flow(car.position[0])
                    routes_taken[car.id].append(car.position[0][1])

                cumulated_step += 1

        return pd.DataFrame(
            self.step_statistics,
            columns=pd.MultiIndex.from_tuples(
                [("route", car_id) for car_id in self.cars]
                + [("flow", edge_id) for edge_id in self.network.edges]
                + [("latency", edge_id) for edge_id in self.network.edges]
                + [("toll", edge_id) for edge_id in self.network.edges]
            ),
        ), pd.DataFrame(self.car_statistics)


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
        seed=42,
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
        return self.rng.choice(
            [(1, 1, self.capacity, 1), (2, 0, 1, 1)], p=[self.q, 1.0 - self.q]
        )


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


def create_cars(network, car_counts, seed=42):
    rng = np.random.default_rng(seed)
    cars = {}  # Dictionary of final cars
    # Dictionary storing counts of cars on each edge for the different goals
    car_distribution = {edge: [0] * len(car_counts) for edge in network.edges}
    # Iterate through (start, goal)-combinations
    for goal_index, ((s, t), count) in enumerate(car_counts.items()):
        # and get their feasible edges and corresponding latencies
        feasible_edges = [
            edge
            for edge in network.edges
            if nx.has_path(network, s, edge[0]) and nx.has_path(network, edge[1], t) and edge[0] != t
        ]
        latencies = [network.edges[edge]["latency_fn"](0) for edge in feasible_edges]

        # Randomly distribute the cars on the edges weighted by the latency
        edge_cars = rng.choice(
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
            choice_of_goal = rng.choice(
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
