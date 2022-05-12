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


class CentralTendencyGaussianPeakProblem2M(CentralTendencyBenchmarkProblem):
    def __init__(self):
        """
        
        Example
        -------
        problem  = CentralTendencyGaussianPeakProblem()
        """

        func = '10 * exp(- 25 * (x1 - 0.5)^2 - 25 * (x2 - 0.5)^2)'
        g = ot.SymbolicFunction(['x1', 'x2'], [func])

        modes = [ot.Normal(0.3, 0.12), ot.Normal(0.7, 0.1)]
        weight = [0.4, 1.0]
        mixture = ot.Mixture(modes, weight)
        X1 = ot.TruncatedDistribution(mixture, 0., 1.)
        X1.setDescription(["X1"])
        X2 = ot.TruncatedDistribution(ot.Normal(0.6, 0.15), 0., 1.)
        X2.setDescription(["X2"])

        myDistribution = ot.ComposedDistribution([X1, X2], ot.ClaytonCopula(2.))
        inputRandomVector = ot.RandomVector(myDistribution)
        compositeRandomVector = ot.CompositeRandomVector(
            g, inputRandomVector
        )

        name = "Gaussian Peak 2D (mixture input)"
        # References computed by a very large MC sample
        mean = 2.9439816239704797
        std = 7.883080829122712

        super(CentralTendencyGaussianPeakProblem2M, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
