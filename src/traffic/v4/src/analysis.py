# External modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import networkx as nx


def compute_regression(data):
    lrr = LinearRegression()
    est = lrr.fit(data.index.to_numpy().reshape(-1, 1), data.to_numpy())
    error = mean_squared_error(
        data.to_numpy(), lrr.predict(data.index.to_numpy().reshape(-1, 1))
    )

    return est.coef_[0], est.intercept_, error


def analyze_fairness(car_stats, cutoff=0):
    data = (
        car_stats[car_stats["step"] >= cutoff]
        .groupby("value_of_money")["travel_time"]
        .mean()
    )
    slope, intercept, error = compute_regression(data)

    return {"slope": slope, "error": error}


def draw_network(network, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.set_title("Network")
    ax.figure.set_size_inches(5, 5)
    nx.draw(
        network,
        ax=ax,
        pos=nx.get_node_attributes(network, "position"),
        with_labels=True,
        font_size=8,
        edgelist=[(v, w) for v, w in network.edges if v != w]
    )
