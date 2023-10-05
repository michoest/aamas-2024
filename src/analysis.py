# External modules
import statsmodels.formula.api as sm
from statsmodels.tools.eval_measures import mse


def compute_regression(data):
    ols = sm.ols(formula='travel_time ~ value_of_money', data=data).fit()
    return ols.params['value_of_money'], ols.params['Intercept'], mse(
        data['travel_time'],
        ols.predict(data['value_of_money'])
    ), ols.pvalues["value_of_money"]


def analyze_fairness(car_stats, cutoff=0):
    data = (
        car_stats[car_stats["step"] >= cutoff]
        .groupby("value_of_money")["travel_time"]
        .mean().reset_index()
    )
    slope, intercept, error, p_value = compute_regression(data)

    return {"slope": slope, "error": error, 'p': p_value}
