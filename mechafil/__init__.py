"""MechaFIL: Mechanistic model for the Filecoin Economy"""

__version__ = "1.2"
__author__ = "Maria Silva <misilva73@gmail.com>, Tom Mellan <t.mellan@imperial.ac.uk>, Kiran Karra <kiran.karra@gmail.com>"
__all__ = []

import pandas

pandas.set_option("mode.chained_assignment", None)
from .sim import run_simple_sim
