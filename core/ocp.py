from typing import Callable

import pybamm
from numpy import exp, ndarray, tanh, arctan

#全局参数
parameter_values = pybamm.ParameterValues("OKane2022")

Cp_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
Cn_max = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]

#正极OCV计算公式
def get_nmc_ocp(Cs: float | ndarray) -> Callable:
    """
    NMC-811 open circuit potential as a function of concentration. OCP function obtained
    from PyBaMM:
    https://github.com/pybamm-team/PyBaMM/blob/abc42832370268bc765458b4517a53cafdf8d926/pybamm/input/parameters/lithium_ion/Chen2020.py#L76

    Parameters
    ----------
    Cs : float | ndarray
        Particle surface concentration

    Returns
    -------
    Callable
        Open circuit potential as a function of concentration.
    """

    sto = Cs / Cp_max
    return (
            -10.72 * sto ** 4
            + 23.88 * sto ** 3
            - 16.77 * sto ** 2
            + 2.595 * sto
            + 4.563
    )

#负极OCV计算公式
def get_graphite_ocp(Cs: float | ndarray):
    """
    Graphite open circuit potential as a function of concentration. OCP function obtained
    from PyBaMM:
    https://github.com/pybamm-team/PyBaMM/blob/abc42832370268bc765458b4517a53cafdf8d926/pybamm/input/parameters/lithium_ion/Prada2013.py#L5

    Parameters
    ----------
    Cs : float | ndarray
        Particle surface concentration

    Returns
    -------
    Callable
        Open circuit potential as a function of concentration.
    """

    sto = Cs / Cn_max
    return (
            0.1493
            + 0.8493 * exp(-61.79 * sto)
            + 0.3824 * exp(-665.8 * sto)
            - exp(39.42 * sto - 41.92)
            - 0.0313 * arctan(25.59 * sto - 4.099)
            - 0.009434 * arctan(32.49 * sto - 15.74)
    )
