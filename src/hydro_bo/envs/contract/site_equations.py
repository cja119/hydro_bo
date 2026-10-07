"""
Equations for the site production problem. These are the production equations of the
shipping problem, without vector storage, shipping or profit, implemented in a format
which can be solved by pyomo.
"""

from pyomo.environ import Constraint


def energy_balance(m, t):
    """
    Energy balance equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip

    eqn = 0

    eqn -= m.energy_curtailed[t]
    eqn -= m.energy_compression[t]
    eqn -= m.energy_electrolysis[t]
    eqn -= m.energy_conversion[t]

    eqn += m.energy_fuelcell[t]

    if m.renewables.value == "wind":
        eqn += m.energy_wind[t] * m.renewable_energy_capacity
    if m.renewables.value == "solar":
        eqn += m.energy_solar[t] * m.renewable_energy_capacity
    return eqn == 0


def hydrogen_production(m, t):
    """
    Hydrogen production equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    eqn = 0

    eqn += m.energy_electrolysis[t] * m.electrolysis_efficiency
    eqn -= m.hydrogen_produced[t]

    return eqn == 0


def compression_balance(m, t):
    """
    Compression balance equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    eqn = 0

    eqn += m.energy_compression[t] * 120

    eqn -= m.hydrogen_produced[t] * m.electrolysis_compression_penalty
    eqn -= m.hydrogen_used[t] * m.production_compression_penalty
    eqn -= m.hydrogen_stored[t] * m.storage_compression_penalty

    return eqn == 0


def influent_hydrogen_balance(m, t):
    """
    Influent hydrogen balance equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    eqn = 0

    eqn += m.hydrogen_produced[t] * m.compression_efficiency
    eqn -= m.hydrogen_stored[t]
    eqn -= m.hydrogen_used[t]

    return eqn == 0


def effluent_hydrogen_balance(m, t):
    """
    Effluent hydrogen balance equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    eqn = 0

    eqn += m.vector_flux[t] / m.vector_synthetic_efficiency
    eqn += m.hydrogen_consumed_fuelcell[t]
    eqn -= m.hydrogen_used[t]
    eqn -= m.hydrogen_removed[t]

    return eqn == 0


def fuel_cell_production(m, t):
    """
    Fuel cell production equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    eqn = 0

    eqn += m.hydrogen_consumed_fuelcell[t] * m.fuelcell_efficiency
    eqn -= m.energy_fuelcell[t]

    return eqn == 0


def hydrogen_storage_balance(m, t):
    """
    Hydrogen storage balance equation.
    """
    if t == 0:
        if m.fixed.value is True:
            return Constraint.Skip
        else:
            return m.hydrogen_storage[t] == 0.5 * m.hydrogen_storage_capacity
    eqn = 0
    eqn += m.hydrogen_storage[t]
    eqn -= m.hydrogen_storage[t - 1]
    eqn += m.hydrogen_removed[t]
    eqn -= m.hydrogen_stored[t] * m.compression_efficiency
    return eqn == 0


def vector_production(m, t):
    """
    Vector production equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip

    eqn = 0

    eqn += (
        m.vector_flux[t]
        * (m.variable_energy_penalty_conversion / m.calorific_value)
        * (1 - m.fixed_energy_penalty_conversion)
    )

    eqn += (
        m.n_active_trains_conversion[t]
        * m.fixed_energy_penalty_conversion
        * m.variable_energy_penalty_conversion
        * m.single_train_limit_conversion
    )

    eqn -= m.energy_conversion[t]

    return eqn == 0


def electrolyser_production_limit(m, t):
    """
    Upper limit on the production capacity of the electrolyser
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip

    cons = 0
    cons += m.energy_electrolysis[t]
    cons -= m.electrolyser_capacity

    return cons <= 0


def fuelcell_production_limit(m, t):
    """
    Upper limit on the capacity of the hydrogen fuel cell
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip

    cons = 0
    cons += m.energy_fuelcell[t]
    cons -= m.fuelcell_capacity

    return cons <= 0


def compression_upper_limit(m, t):
    """
    Upper limit on the capacity of the hydrogen fuel cell
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip

    cons = 0
    cons += m.energy_compression[t]
    cons -= m.compression_capacity

    return cons <= 0


def lower_hydrogen_storage_limit(m, t):
    """
    Lower hydrogen storage limit. 
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    if t < m.backoff_start.value:
        return Constraint.Skip

    cons = 0
    cons += m.hydrogen_storage_capacity * m.hydrogen_storage_lower_backoff
    cons -= m.hydrogen_storage[t]

    return cons <= 0


def upper_hydrogen_storage_limit(m, t):
    """
    Upper hydrogen storage limit. 
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    cons = 0
    cons += m.hydrogen_storage[t]

    if t < m.backoff_start.value:
        cons -= m.hydrogen_storage_capacity
    else:
        cons -= m.hydrogen_storage_capacity * m.hydrogen_storage_upper_backoff

    return cons <= 0


def lower_vector_ramping_limit(m, t):
    """
    Lower vector ramping limit equation.
    """
    if t == 0:
        return Constraint.Skip

    cons = 0
    cons += m.vector_flux[t - 1]
    cons -= m.vector_flux[t]
    cons /= m.calorific_value
    cons -= (
        m.n_active_trains_conversion[t - 1]
        * m.single_train_limit_conversion
        * m.ramp_down_limit
    )

    return cons <= 0


def upper_vector_ramping_limit(m, t):
    """
    Upper vector ramping limit equation.
    """
    if t == 0:
        return Constraint.Skip
    cons = 0

    cons += m.vector_flux[t]
    cons -= m.vector_flux[t - 1]
    cons /= m.calorific_value
    cons -= (
        (m.conversion_trains_number + 1 - m.n_active_trains_conversion[t - 1])
        * m.single_train_limit_conversion
        * m.ramp_up_limit
    )

    return cons <= 0


def lower_vector_production_limit(m, t):
    """
    Lower vector production limit equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    cons = 0

    cons -= m.vector_flux[t] / m.calorific_value
    cons += (
        m.single_train_limit_conversion
        * m.n_active_trains_conversion[t]
        * m.minimum_train_throughput
    )

    return cons <= 0


def upper_vector_production_limit(m, t):
    """
    Upper vector production limit equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    cons = 0

    cons += m.vector_flux[t] / m.calorific_value
    cons -= m.single_train_limit_conversion * m.n_active_trains_conversion[t]

    return cons <= 0


def active_trains_limit(m, t):
    """
    Active trains limit equation.
    """
    if t == 0 and m.fixed.value is True:
        return Constraint.Skip
    cons = 0

    cons += m.n_active_trains_conversion[t]
    cons -= m.conversion_trains_number

    return cons <= 0


def storage_addition_limit(m, t):
    """
    Prevents model from adding to and removing from storage simultaneously.
    This avoids loops that exploit numerical noise by blowing up these values.
    """
    cons = 0
    cons += m.hydrogen_stored[t]
    cons -= m.hydrogen_storage_capacity * m.storage_bool[t]
    return cons <= 0


def storage_removal_limit(m, t):
    """
    Prevents model from adding to and removing from storage simultaneously.
    This avoids loops that exploit numerical noise by blowing up these values.
    """
    cons = 0
    cons += m.hydrogen_removed[t]
    cons -= m.hydrogen_storage_capacity * (1 - m.storage_bool[t])
    return cons <= 0


def hydrogen_production_maximisation(m):
    """
    Hydrogen production maximisation equation.
    """
    obj = 0
    obj += sum(m.vector_flux[t] for t in m.grid0)
    return obj
