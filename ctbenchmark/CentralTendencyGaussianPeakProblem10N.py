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


class CentralTendencyGaussianPeakProblem10N(CentralTendencyBenchmarkProblem):
    def __init__(self, dim=10):
        """
        
        Example
        -------
        problem  = CentralTendencyGaussianPeakProblem()
        """
        self.dim = dim
        elements = ["(x{} - 0.3)^2".format(i) for i in range(1, self.dim + 1)]
        sum_formula = " + ".join(elements)
        formula = "exp(- (10^2) * {})".format(sum_formula)
        inputs = ["x{}".format(i) for i in range(1, self.dim + 1)]
        g = ot.SymbolicFunction(inputs, [formula])

        # Define the distribution
        distributionList = [ot.TruncatedDistribution(ot.Normal(0.5, 0.2), 0., 1.) for i in range(self.dim)]
        #distributionList = [ot.Uniform(0., 1.) for i in range(self.dim)]
        myDistribution = ot.ComposedDistribution(distributionList)
        inputRandomVector = ot.RandomVector(myDistribution)
        compositeRandomVector = ot.CompositeRandomVector(
            g, inputRandomVector
        )

        name = "Gaussian Peak 10D (normal input)"
        # References computed by a very large Sobol sample (size 10**8)
        mean = 0.44684171199013906
        std = 0.7155066113804083

        super(CentralTendencyGaussianPeakProblem10N, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
