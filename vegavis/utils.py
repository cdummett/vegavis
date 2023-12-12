import json
import requests

from math import log, sqrt, exp
from scipy.stats import lognorm
from typing import Optional

from enum import Enum


class Side(Enum):
    SIDE_BUY = 0
    SIDE_SELL = 1


def probability_of_trading(
    side: Side,
    price: float,
    best_bid_price: float,
    best_ask_price: float,
    min_valid_price: Optional[float],
    max_valid_price: Optional[float],
    mu: float,
    tau: float,
    sigma: float,
    min_probability_of_trading: float,
    tau_scaling: Optional[float] = None,
) -> float:
    """Compute the probability of trading of a given order.

    Probability of trading follows the cdf of a lognormal distribution. The distribution
    is bound between the min_valid_price and the best_bid_price for buy orders, and the
    best_ask_price and max_valid_price for sell orders. All probabilities are scaled by
    a half and cannot be lower than the network parameter, min_probability_of_trading.

    Args:
        side (Side):
            Side of the order to evaluate.
        price (float):
            Price of the order to evaluate.
        best_bid_price (float):
            Best bid price on the order book.
        best_ask_price (float):
            Best ask price on the order book.
        min_valid_price (float):
            Minimum valid price from price monitoring bounds.
        max_valid_price (float):
            Maximum valid price from price monitoring bounds.
        mu (float):
            Market parameter for the risk model.
        tau (float):
            Market parameter for the risk model.
        sigma (float):
            Market parameter for the risk model.
        min_probability_of_trading (float):
            Network parameter defining the minimum value to return.
        tau_scaling (Optional[float]):
            Optional scaling factor for tau.

    Returns:
        float:
            Probability of trading of the order.
    """

    if price > best_bid_price and price < best_ask_price:
        return 0.5
    elif (
        min_valid_price is None
        or max_valid_price is None
        or price < min_valid_price
        or price > max_valid_price
    ):
        return min_probability_of_trading

    if side == Side.SIDE_BUY:
        best_price = best_bid_price

        lower_bound = min_valid_price
        upper_bound = best_bid_price

    else:
        best_price = best_ask_price

        lower_bound = best_ask_price
        upper_bound = max_valid_price

    stdev = sigma * sqrt(tau * tau_scaling)
    m = log(best_price) + (mu - 0.5 * sigma**2) * tau * tau_scaling

    rv = lognorm(s=stdev, scale=exp(m))

    min = rv.cdf(lower_bound)
    max = rv.cdf(upper_bound)
    z = max - min

    if side == Side.SIDE_BUY:
        p = 0.5 * (rv.cdf(price) - min) / z
    else:
        p = 0.5 * (max - rv.cdf(price)) / z

    if p < min_probability_of_trading:
        return min_probability_of_trading
    else:
        return p


def cumulative_probability_of_trading(
    side: Side,
    price: float,
    best_bid_price: float,
    best_ask_price: float,
    min_valid_price: Optional[float],
    max_valid_price: Optional[float],
    mu: float,
    tau: float,
    sigma: float,
    min_probability_of_trading: float,
    tau_scaling: Optional[float] = None,
) -> float:
    """Compute the probability of trading of a given order.

    Probability of trading follows the cdf of a lognormal distribution. The distribution
    is bound between the min_valid_price and the best_bid_price for buy orders, and the
    best_ask_price and max_valid_price for sell orders. All probabilities are scaled by
    a half and cannot be lower than the network parameter, min_probability_of_trading.

    Args:
        side (Side):
            Side of the order to evaluate.
        price (float):
            Price of the order to evaluate.
        best_bid_price (float):
            Best bid price on the order book.
        best_ask_price (float):
            Best ask price on the order book.
        min_valid_price (float):
            Minimum valid price from price monitoring bounds.
        max_valid_price (float):
            Maximum valid price from price monitoring bounds.
        mu (float):
            Market parameter for the risk model.
        tau (float):
            Market parameter for the risk model.
        sigma (float):
            Market parameter for the risk model.
        min_probability_of_trading (float):
            Network parameter defining the minimum value to return.
        tau_scaling (Optional[float]):
            Optional scaling factor for tau.

    Returns:
        float:
            Probability of trading of the order.
    """

    if price > best_bid_price and price < best_ask_price:
        return 0.5
    elif (
        min_valid_price is None
        or max_valid_price is None
        or price < min_valid_price
        or price > max_valid_price
    ):
        return min_probability_of_trading

    if side == Side.SIDE_BUY:
        best_price = best_bid_price

        lower_bound = min_valid_price
        upper_bound = best_bid_price

    else:
        best_price = best_ask_price

        lower_bound = best_ask_price
        upper_bound = max_valid_price

    stdev = sigma * sqrt(tau * tau_scaling)
    m = log(best_price) + (mu - 0.5 * sigma**2) * tau * tau_scaling

    rv = lognorm(s=stdev, scale=exp(m))

    min = rv.cdf(lower_bound)
    max = rv.cdf(upper_bound)
    z = max - min

    if side == Side.SIDE_BUY:
        p = 0.5 * (rv.cdf(price) - min) / z
    else:
        p = 0.5 * (max - rv.cdf(price)) / z

    if p < min_probability_of_trading:
        return min_probability_of_trading
    else:
        return p


def num_from_padded_int(to_convert: int, decimals: int) -> float:
    if not to_convert:
        return 0
    to_convert = int(to_convert) if isinstance(to_convert, str) else to_convert
    return float(to_convert) / 10**decimals
