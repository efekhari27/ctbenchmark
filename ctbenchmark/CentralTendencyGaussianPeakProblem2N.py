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


class CentralTendencyGaussianPeakProblem2N(CentralTendencyBenchmarkProblem):
    def __init__(self):
        """
        
        Example
        -------
        problem  = CentralTendencyGaussianPeakProblem()
        """

        func = '10 * exp(- 25 * (x1 - 0.35)^2 - 25 * (x2 - 0.35)^2)'
        g = ot.SymbolicFunction(['x1', 'x2'], [func])

        X1 = ot.TruncatedDistribution(ot.Normal(0.5, 0.15), 0., 1.)
        X1.setDescription(["X1"])
        X2 = ot.TruncatedDistribution(ot.Normal(0.5, 0.15), 0., 1.)
        X2.setDescription(["X2"])

        myDistribution = ot.ComposedDistribution([X1, X2])
        inputRandomVector = ot.RandomVector(myDistribution)
        compositeRandomVector = ot.CompositeRandomVector(
            g, inputRandomVector
        )

        name = "Gaussian Peak 2D (normal input)"
        # References computed by a very large Sobol sample (size 10**8)
        mean = 2.7761168106934293
        std = 2.778026982859545

        super(CentralTendencyGaussianPeakProblem2N, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
