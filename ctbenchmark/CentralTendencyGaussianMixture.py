#!/usr/bin/python
# coding:utf-8
# Copyright 2022 EDF
"""
Class to define the Gaussian Mixture benchmark problem.

@author: efekhari27
"""

from ctbenchmark.CentralTendencyBenchmarkProblem import CentralTendencyBenchmarkProblem
import openturns as ot
import numpy as np


class CentralTendencyGaussianMixture(CentralTendencyBenchmarkProblem):
    def __init__(self):
        """

        Example
        -------
        problem  = CentralTendencyGaussianMixture()
        """

        func = 'x1 + x2'
        g = ot.SymbolicFunction(['x1', 'x2'], [func])


        modes1 = [ot.Normal(0.22, 0.08), ot.Normal(0.4, 0.066), ot.Normal(0.66, 0.1), ot.Normal(0.85, 0.045)]
        weight1 = [0.4, 1.0, 1.2, 0.6]
        mixture1 = ot.Mixture(modes1, weight1)
        X1 = ot.TruncatedDistribution(mixture1, 0., 1.)
        X2 = ot.TruncatedDistribution(ot.Normal(0.6, 0.15), 0., 1.)
        distribution1 = ot.ComposedDistribution([X1, X2], ot.ClaytonCopula(2.))
        distribution2 = ot.ComposedDistribution([ot.TruncatedDistribution(ot.Normal(0.6, 0.12), 0., 1.), ot.TruncatedDistribution(ot.Normal(0.3, 0.1), 0., 1.)], ot.GalambosCopula(0.9))
        distribution3 = ot.ComposedDistribution([ot.TruncatedDistribution(ot.Normal(0.2, 0.05), 0., 1.), ot.TruncatedDistribution(ot.Normal(0.8, 0.05), 0., 1.)], ot.ClaytonCopula(0.5))

        myDistribution = ot.Distribution(ot.Mixture([distribution1, distribution2, distribution3], [1., 0.5, 0.2]))

        inputRandomVector = ot.RandomVector(myDistribution)
        compositeRandomVector = ot.CompositeRandomVector(
            g, inputRandomVector
        )

        name = "Gaussian Mixture"
        # References computed by a very large MC sample (size 10**8)
        mean = 1.0634174346117014
        std = 0.3047704459456222

        super(CentralTendencyGaussianMixture, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
