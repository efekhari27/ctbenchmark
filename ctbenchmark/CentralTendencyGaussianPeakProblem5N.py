#!/usr/bin/python
# coding:utf-8
# Copyright 2022 EDF
"""
Class to define the Gaussian Peak benchmark problem.

@author: efekhari27
"""

from ctbenchmark.CentralTendencyBenchmarkProblem import CentralTendencyBenchmarkProblem
import openturns as ot
import numpy as np


class CentralTendencyGaussianPeakProblem5N(CentralTendencyBenchmarkProblem):
    def __init__(self, dim=5):
        """
        
        Example
        -------
        problem  = CentralTendencyGaussianPeakProblem()
        """
        self.dim = dim
        elements = ["(x{} - 0.35)^2".format(i) for i in range(1, self.dim + 1)]
        sum_formula = " + ".join(elements)
        formula = "exp(- 25 * {})".format(sum_formula)
        inputs = ["x{}".format(i) for i in range(1, self.dim + 1)]
        g = ot.SymbolicFunction(inputs, [formula])

        # Define the distribution
        distributionList = [ot.TruncatedDistribution(ot.Normal(0.5, 0.15), 0., 1.) for i in range(self.dim)]
        #distributionList = [ot.Normal(0.5, 0.3) for i in range(self.dim)]
        myDistribution = ot.ComposedDistribution(distributionList)
        inputRandomVector = ot.RandomVector(myDistribution)
        compositeRandomVector = ot.CompositeRandomVector(
            g, inputRandomVector
        )

        name = "Gaussian Peak 5D (normal input)"
        # References computed by a very large Sobol sample (size 10**8)
        mean = 0.6340250731517876
        std = 0.41751146118991256

        super(CentralTendencyGaussianPeakProblem5N, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
