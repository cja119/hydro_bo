"""MPC parameters that are structural: used in constraint-rule control
flow or set indexing, so changing them requires an instance rebuild.
Shared by `mpc._build_params` (which creates everything else mutable)
and the theta registry (which refuses in-place sinks on these).
"""

IMMUTABLE_MPC_PARAMS = frozenset({
    "mean_ship_arrival_time",
    "mean_ship_transit_time",
    "std_ship_transit_time",
    "expected_arrival_offset",
    "forecast_horizon",
})
