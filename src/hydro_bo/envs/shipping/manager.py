"""
Env Manager for Shipping Environment
"""

from hydro_bo.envs.contract.core import ContractEnv
from hydro_bo.envs.shipping.core import ShippingEnv, ShippingEnvPlot


def ShippingEnvManager(version: str = "v1"):
    """
    Factory for creating the desired version of the shipping environment.
    """
    if version == "shipping":
        return ShippingEnv()
    elif version == "shipping-plot":
        return ShippingEnvPlot()
    elif version == "contract":
        return ContractEnv()
    else:
        raise ValueError(f"Unsupported version: {version}")
