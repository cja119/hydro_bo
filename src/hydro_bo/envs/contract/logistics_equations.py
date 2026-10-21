"""
Equations for the logistics problem: export storage, shipping and the buyer's inventory
at daily resolution. These are implemented in a format which can be solved by pyomo.
"""

from pyomo.environ import Constraint, value


def vector_storage_balance(m, t):
    """
    Vector storage balance equation.
    """
    if t == 0:
        if m.fixed.value is True:
            return Constraint.Skip
        else:
            return m.vector_storage[t] == 0.5 * m.vector_storage_capacity

    eqn = 0

    eqn += m.vector_storage[t]
    eqn -= m.vector_storage[t - 24]
    eqn -= m.vector_delivered[t] / 1000  # site production
    eqn += (
        sum(
            m.n_ship_sent[s, t] * m.ship_capacity[s]
            for s in m.ships
        )
        / 1000
    )  # shipping (outflow)
    eqn += m.vector_flared[t]  # flaring (outflow)

    return eqn == 0


def port_capacity(m, s, t):
    """
    Constraint on the port capacity.
    """
    if t == 0:
        if m.fixed.value is True:
            return Constraint.Skip
        else:
            return m.n_ship_sent[s, t] + m.waiting_ships[s, t] == 0

    eqn = 0

    eqn += m.waiting_ships[s, t]
    eqn -= m.waiting_ships[s, t - 24]
    if t == 24:
        eqn -= m.ship_arrived[s]

    eqn += m.n_ship_sent[s, t]

    if t >= (m.mean_ship_arrival_time + m.expected_arrival_offset) * 24:
        eqn -= m.n_ship_ordered[s, t - (m.mean_ship_arrival_time + m.expected_arrival_offset) * 24]

    if t > 24:
        eqn -= m.expected_ships[s, t]

    return eqn == 0


def ship_holding_balance(m, s, t):
    """
    Ship holding balance equation.
    """
    if t == 0:
        if m.fixed.value is True:
            return Constraint.Skip
        else:
            return m.ships_holding[s, t] == 0

    eqn = 0

    eqn += m.ships_holding[s, t]
    eqn -= m.ships_holding[s, t - 24]
    if t >= m.laden_transit_time.value * 24:
        eqn -= m.n_ship_sent[s, t - m.laden_transit_time.value * 24]
    eqn -= m.laden_arrivals[s, t]

    eqn += m.n_ship_discharged[s, t]

    return eqn == 0


def buyer_inventory_balance(m, t):
    """
    Buyer inventory balance equation.
    """
    if t == 0:
        if m.fixed.value is True:
            return Constraint.Skip
        else:
            return m.buyer_inventory[t] == 0.5 * (m.inventory_lower + m.inventory_upper)

    eqn = 0

    eqn += m.buyer_inventory[t]
    eqn -= m.buyer_inventory[t - 24]
    eqn -= sum(
        m.n_ship_discharged[s, t]
        * (m.ship_capacity[s] - m.ship_fuel_rate[s] * (m.laden_transit_time + m.ballast_transit_time))
        * m.calorific_value
        / 120
        / 1000
        for s in m.ships
    )
    eqn -= m.spot_bought[t]

    eqn += m.spot_sold[t]
    eqn += m.commitment / 30

    return eqn == 0


def initial_spot_trades(m, t):
    """
    Initial spot trades equation.
    """
    if t == 0:
        if m.fixed.value is True:
            return Constraint.Skip
        else:
            return m.spot_bought[t] + m.spot_sold[t] == 0
    return Constraint.Skip


def lower_vector_storage_limit(m, t):
    """
    Lower vector storage limit.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    if t < m.forecast_horizon.value:
        return Constraint.Skip
    cons = 0

    cons += m.vector_storage_capacity * m.vector_storage_lower_backoff
    cons -= m.vector_storage[t]

    return cons <= 0


def upper_vector_storage_limit(m, t):
    """
    Upper vector storage limit.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    cons = 0
    cons += m.vector_storage[t]

    if t < m.forecast_horizon.value:
        cons -= m.vector_storage_capacity
    else:
        cons -= m.vector_storage_capacity * m.vector_storage_upper_backoff

    return cons <= 0


def lower_buyer_inventory_limit(m, t):
    """
    Lower buyer inventory limit.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    cons = 0

    cons += m.inventory_lower
    cons -= m.buyer_inventory[t]

    return cons <= 0


def upper_buyer_inventory_limit(m, t):
    """
    Upper buyer inventory limit.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    cons = 0

    cons += m.buyer_inventory[t]
    cons -= m.inventory_upper

    return cons <= 0


def daily_profit(m, t):
    """
    Daily profit equation.
    """
    eqn = 0
    eqn += m.cumulative_profit[t]
    if t != m.grid0.at(-1):
        eqn -= m.cumulative_profit[t + 24] * m.daily_discount

    eqn += (
        sum(
            m.waiting_ships[s, t]
            * (m.ship_berthing_rate[s] + m.ship_charter_rate[s])
            * 24
            for s in m.ships
        )
        / 1_000_000
    )
    eqn += (
        sum(
            m.ships_holding[s, t]
            * m.ship_charter_rate[s]
            * 24
            for s in m.ships
        )
        / 1_000_000
    )
    eqn += sum(
        m.holding_loss_rate
        * m.ships_holding[s, t]
        * m.ship_capacity[s]
        * m.calorific_value
        / 120
        / 1000
        * m.spot_forecast[t]
        for s in m.ships
    )
    eqn += (
        sum(
            m.n_ship_ordered[s, t]
            * m.ship_charter_rate[s]
            * m.mean_ship_arrival_time
            * 24
            for s in m.ships
        )
        / 1_000_000
    )
    eqn += (
        sum(
            m.n_ship_sent[s, t]
            * (m.ship_charter_rate[s] * m.laden_transit_time * 24 + m.port_fee[s])
            for s in m.ships
        )
        / 1_000_000
    )
    eqn += (
        sum(
            m.n_ship_discharged[s, t]
            * (m.ship_charter_rate[s] * (m.ballast_transit_time + m.port_time) * 24 + m.port_fee[s])
            for s in m.ships
        )
        / 1_000_000
    )
    eqn += (
        m.spot_bought[t]
        * (1 + m.half_spread)
        * m.spot_forecast[t]
    )
    if t != 0:
        eqn += (
            m.vector_delivered[t]
            / 1000
            * m.calorific_value
            / 120
            * m.fundamental_price[t]
        )

    eqn -= (
        m.commitment
        / 30
        * m.fundamental_price[t]
    )
    eqn -= (
        m.spot_sold[t]
        * (1 - m.half_spread)
        * m.spot_forecast[t]
    )

    return eqn == 0


def logistics_profit_target(m):
    """
    Logistics profit target equation.
    """
    T = m.grid0.at(-1)
    # Valuation of stock held at the start and end of the horizon (terminal_valuation):
    #   0 - mid price, forecast at T (original)
    #   1 - mid price, today's netback
    #   2 - storage netback at the bid (logistics_profit_target_storage_netback)
    #   3 - storage netback at the marginal bid on the day's spot volume, floored at 0 (values set by LogisticsDynamics)
    valuation = int(value(m.terminal_valuation))
    if valuation in (2, 3):
        return logistics_profit_target_storage_netback(m)
    terminal_price = m.spot_forecast[0] if valuation == 1 else m.spot_forecast[T]
    bid = 1

    return (
        m.cumulative_profit[m.grid0.at(1)]
        + 0.99999 ** T
        * (
            m.vector_storage[T]
            * m.calorific_value
            / 120
            * (
                (1 - m.ship_fuel_rate["large"] * (m.laden_transit_time + m.ballast_transit_time) / m.ship_capacity["large"])
                * bid * terminal_price
                - (
                    m.ship_charter_rate["large"]
                    * (m.mean_ship_arrival_time + m.laden_transit_time + m.ballast_transit_time + m.port_time)
                    * 24
                    + 2 * m.port_fee["large"]
                )
                / 1_000_000
                / (m.ship_capacity["large"] * m.calorific_value / 120 / 1000)
            )
            + sum(
                (
                    m.ships_holding[s, T]
                    + sum(
                        m.n_ship_sent[s, t]
                        for t in range(T - (m.laden_transit_time.value - 1) * 24, T + 1, 24)
                    )
                )
                * (
                    (m.ship_capacity[s] - m.ship_fuel_rate[s] * (m.laden_transit_time + m.ballast_transit_time))
                    * m.calorific_value
                    / 120
                    / 1000
                    * bid * terminal_price
                    - (m.ship_charter_rate[s] * (m.ballast_transit_time + m.port_time) * 24 + m.port_fee[s])
                    / 1_000_000
                )
                for s in m.ships
            )
            + bid * terminal_price
            * m.buyer_inventory[T]
        )
        - (
            m.vector_storage[0]
            * m.calorific_value
            / 120
            * (
                (1 - m.ship_fuel_rate["large"] * (m.laden_transit_time + m.ballast_transit_time) / m.ship_capacity["large"])
                * bid * m.spot_forecast[0]
                - (
                    m.ship_charter_rate["large"]
                    * (m.mean_ship_arrival_time + m.laden_transit_time + m.ballast_transit_time + m.port_time)
                    * 24
                    + 2 * m.port_fee["large"]
                )
                / 1_000_000
                / (m.ship_capacity["large"] * m.calorific_value / 120 / 1000)
            )
            + sum(
                (
                    m.ships_holding[s, 0]
                    + sum(m.laden_arrivals[s, t] for t in m.grid1)
                    + m.n_ship_sent[s, 0]
                )
                * (
                    (m.ship_capacity[s] - m.ship_fuel_rate[s] * (m.laden_transit_time + m.ballast_transit_time))
                    * m.calorific_value
                    / 120
                    / 1000
                    * bid * m.spot_forecast[0]
                    - (m.ship_charter_rate[s] * (m.ballast_transit_time + m.port_time) * 24 + m.port_fee[s])
                    / 1_000_000
                )
                for s in m.ships
            )
            + bid * m.spot_forecast[0]
            * m.buyer_inventory[0]
        )
        - sum(
            m.daily_discount ** (t / 24)
            * m.half_spread
            * m.spot_forecast[t]
            * (m.spot_bought[t] ** 2 + m.spot_sold[t] ** 2)
            / m.reference_volume
            for t in m.grid0
        )
    )


def logistics_profit_target_storage_netback(m):
    """
    Logistics profit target with stock held at the start and end of the horizon valued at its storage netback: the bid
    (1 - eps) at the forecast netback on the expected sale date, less the costs still to pay to reach the buyer, discounted
    at the project rate (tv0_* for day 0, tv_* for day T, set daily by LogisticsDynamics.storage_netback). Cash flows
    within the horizon are discounted by daily_discount = exp(-r/365) per day.
    """
    T = m.grid0.at(-1)
    laden = int(value(m.laden_transit_time))
    h2 = m.calorific_value / 120

    def ship(prefix, s, k, n):
        """Value of n ships of size s with k laden days still to sail [$M]."""
        landed = (m.ship_capacity[s] - m.ship_fuel_rate[s] * (m.laden_transit_time + m.ballast_transit_time)) * h2 / 1000
        charge = (m.ship_charter_rate[s] * (m.ballast_transit_time + m.port_time) * 24 + m.port_fee[s]) / 1_000_000
        key = f"{s}:{min(max(k, 0), laden)}"
        return n * (landed * getattr(m, f"{prefix}_sea")[key] - charge * getattr(m, f"{prefix}_sea_disc")[key])

    end = (
        m.vector_storage[T] * h2 * m.tv_origin
        + sum(
            ship("tv", s, 0, m.ships_holding[s, T])
            + sum(ship("tv", s, laden - (T - t) // 24, m.n_ship_sent[s, t]) for t in range(T - (laden - 1) * 24, T + 1, 24))
            for s in m.ships
        )
        + m.buyer_inventory[T] * m.tv_buyer
    )
    start = (
        m.vector_storage[0] * h2 * m.tv0_origin
        + sum(
            ship("tv0", s, 0, m.ships_holding[s, 0])
            + sum(ship("tv0", s, t // 24 - 1, m.laden_arrivals[s, t]) for t in m.grid1)
            + ship("tv0", s, laden, m.n_ship_sent[s, 0])
            for s in m.ships
        )
        + m.buyer_inventory[0] * m.tv0_buyer
    )
    spread = sum(
        m.daily_discount ** (t / 24)
        * m.half_spread
        * m.spot_forecast[t]
        * (m.spot_bought[t] ** 2 + m.spot_sold[t] ** 2)
        / m.reference_volume
        for t in m.grid0
    )
    return m.cumulative_profit[m.grid0.at(1)] + end - start - spread
