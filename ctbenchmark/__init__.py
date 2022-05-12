"""ctbenchmark module."""
from .CentralTendencyBenchmarkProblem import CentralTendencyBenchmarkProblem
from .CentralTendencyProblem22 import CentralTendencyProblem22
from .CentralTendencyIrregularProblem import CentralTendencyIrregularProblem
from .CentralTendencyBraninProblem import CentralTendencyBraninProblem
from .CentralTendencyGSobolProblem import CentralTendencyGSobolProblem
from .CentralTendencyBenchmark import CentralTendencyBenchmark
from .CentralTendencyCosin2Problem import CentralTendencyCosin2Problem
from .CentralTendencyPeakProblem import CentralTendencyPeakProblem
from .CentralTendencyGaussianPeakProblem2M import CentralTendencyGaussianPeakProblem2M
from .CentralTendencyGaussianPeakProblem2N import CentralTendencyGaussianPeakProblem2N
from .CentralTendencyGaussianPeakProblem5N import CentralTendencyGaussianPeakProblem5N

from .DesignOfExperiments import DesignOfExperiments
from .AKDA import AKDA
from .plotools import *


__all__ = [
    "CentralTendencyBenchmarkProblem",
    "CentralTendencyProblem22",
    "CentralTendencyIrregularProblem", 
    "CentralTendencyBraninProblem", 
    "CentralTendencyGSobolProblem",
    "CentralTendencyBenchmark",
    "DrawFunctions",
    "CentralTendencyCosin2Problem",
    "CentralTendencyPeakProblem",
    "CentralTendencyGaussianPeakProblem2M",
    "CentralTendencyGaussianPeakProblem2N",
    "CentralTendencyGaussianPeakProblem5N",
    "DesignOfExperiments",
    "AKDA", 
    "plotools"
]
__version__ = "1.0"