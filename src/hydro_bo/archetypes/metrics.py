"""Per-rollout financial metrics for a ContractEnv run under the storage-netback terminal value (terminal_valuation 2/3).

`d` is the run record: the ContractEnv daily records stacked into arrays, plus the logistics parameters a run needs for
its accounting (ship capacities and rates, transit times, port fees, carrier calorific value, warm-up days, ...); see
`rollout.record_of`. NPV = one evaluated year's net cash flow annuitised over the plant life at the project rate, less
the plant capex + opex (present values, USD M) and the export jetty. End-of-year stock (origin tank, at sea, buyer tank)
is valued with the per-day storage-netback factors the controller itself used (tv0_*), so the accounting and the
controller agree.
"""

from __future__ import annotations

import numpy as np

YEARS = 30
TOL = 1e-6


def plant_costs(capex: float, opex: float, jetty_capex: float, n_berths: int, discount_rate: float) -> dict:
    """Plant totals [USD M] and the annuity factor that turns one year's cash flow into a present value."""
    r_hourly = (1 + discount_rate) ** (1 / 8760) - 1
    hours = YEARS * 8760
    crf = (r_hourly * (1 + r_hourly) ** hours) / ((1 + r_hourly) ** hours - 1)
    total = capex + opex + jetty_capex * n_berths
    return {"capex": capex, "opex": opex, "jetty": jetty_capex * n_berths, "total": total,
            "annual": 8760 * total * crf, "annuity": 1 / (8760 * crf)}


def voyage(d: dict) -> dict:
    """Per ship size: H2 per t carrier, cargo, voyage fuel and landed cargo [kt H2], and the discharge charge [USD M]."""
    h2 = float(d["calorific_value"]) / 120
    rate = d["ship_charter_rate"]
    fuel_t = d["ship_fuel_rate"] * (float(d["laden_transit_time"]) + float(d["ballast_transit_time"]))
    return {
        "h2": h2,
        "cargo": d["ship_capacity"] * h2 / 1000,
        "fuel": fuel_t * h2 / 1000,
        "landed": (d["ship_capacity"] - fuel_t) * h2 / 1000,
        "fuel_share": fuel_t / d["ship_capacity"],
        "discharge_charge": (rate * (float(d["ballast_transit_time"]) + float(d["port_time"])) * 24 + d["port_fee"]) / 1e6,
    }


def charges(d: dict) -> dict:
    """Daily charter split, port fees and holding loss [USD M]."""
    rate, v = d["ship_charter_rate"], voyage(d)
    return {
        "positioning": d["ordered"] @ (rate * float(d["mean_ship_arrival_time"]) * 24) / 1e6,
        "laden": d["sent"] @ (rate * float(d["laden_transit_time"]) * 24) / 1e6,
        "ballast_port": d["discharged"] @ (rate * (float(d["ballast_transit_time"]) + float(d["port_time"])) * 24) / 1e6,
        "waiting_holding": (d["waiting_model"] @ ((d["ship_berthing_rate"] + rate) * 24) + d["holding_model"] @ (rate * 24)) / 1e6,
        "port_fees": (d["sent"] + d["discharged"]) @ d["port_fee"] / 1e6,
        "holding_loss": float(d["holding_loss_rate"]) * (d["holding_model"] @ v["cargo"]) * d["spot_price"],
    }


def _afloat(d: dict, day: int) -> list[tuple[str, int]]:
    """Ships laden at sea or holding at the buyer at the end of `day`: (size, laden days left)."""
    return [(str(z), max(int(arr) - day, 0)) for z, sent, arr, dis in
            zip(d["ship_size"], d["ship_sent"], d["ship_buyer_arrival"], d["ship_discharged"])
            if 0 <= sent <= day and (dis < 0 or dis > day)]


def stock_split(d: dict, day: int) -> dict:
    """End-of-day stock [kt H2]: origin tank, at sea (landed equivalent), buyer tank."""
    v = voyage(d)
    sizes = [str(s) for s in d["ships"]]
    return {"origin": float(d["storage_end"][day] * v["h2"]),
            "sea": float(sum(v["landed"][sizes.index(s)] for s, _ in _afloat(d, day))),
            "buyer": float(d["buyer_inventory"][day])}


def stock_netback(d: dict, day: int) -> float:
    """Storage-netback value of the end-of-day stock [USD M], from the factors the controller used that day."""
    v = voyage(d)
    sizes = [str(s) for s in d["ships"]]
    laden = int(d["laden_transit_time"])
    sea = 0.0
    for s, k in _afloat(d, day):
        j = sizes.index(s)
        i = j * (laden + 1) + min(k, laden)
        sea += v["landed"][j] * d["tv0_sea"][day][i] - v["discharge_charge"][j] * d["tv0_sea_disc"][day][i]
    x = stock_split(d, day)
    return x["origin"] * float(d["tv0_origin"][day]) + sea + x["buyer"] * float(d["tv0_buyer"][day])


def rollout_metrics(d: dict, costs: dict) -> dict:
    """NPV and its parts for one rollout [USD M unless stated]."""
    w = int(d["warm_up_days"])
    last = int(d["day"][-1])
    ev = d["day"] >= w
    eps, q_ref = float(d["half_spread"]), float(d["reference_volume"])
    spot, sold, bought = d["spot_price"], d["spot_sold"], d["spot_bought"]
    v, ch = voyage(d), charges(d)
    out = {k: float(x[ev].sum()) for k, x in ch.items()}
    out["charter"] = out["positioning"] + out["laden"] + out["ballast_port"] + out["waiting_holding"]
    start, end = stock_split(d, w - 1), stock_split(d, last)
    produced = float(d["delivered_t"][ev].sum() * v["h2"] / 1000)
    out.update({
        "contract": float((d["fundamental"] * float(d["commitment"]) / 30)[ev].sum()),
        "sales": float((spot * sold)[ev].sum()),
        "purchases": float((spot * bought)[ev].sum()),
        "linear": float((eps * spot * (sold + bought))[ev].sum()),
        "impact": float((eps * spot * (sold**2 + bought**2) / q_ref)[ev].sum()),
        "stock": stock_netback(d, last) - stock_netback(d, w - 1),
        "landed_kt": float((d["discharged"] @ v["landed"])[ev].sum()),
        "sold_kt": float(sold[ev].sum()),
        "bought_kt": float(bought[ev].sum()),
        "produced_kt": produced,
        "unsold_share": (sum(end.values()) - sum(start.values())) / max(produced, TOL),
        "ship_days_held": int(d["holding"][ev].sum()),
        "flared_kt": float((d["flared"] * v["h2"])[ev].sum()),
        "time_limited_days": int((d["status"][ev] != "optimal").sum() + (d["site_status"][ev] != "optimal").sum()),
    })
    out["net"] = (out["contract"] + out["sales"] - out["purchases"] - out["linear"] - out["impact"] - out["charter"]
                  - out["port_fees"] - out["holding_loss"] + out["stock"])
    out["npv"] = out["net"] * costs["annuity"] - costs["total"]
    return out
