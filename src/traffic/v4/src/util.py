# External modules
import numpy as np


def change_value_of_money(cars, possible_values, *, seed=42):
    rng = np.random.default_rng(seed)

    for car in cars.values():
        car.value_of_money = rng.choice(possible_values)
