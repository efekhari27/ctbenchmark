#!/usr/bin/python
# coding:utf-8
# Copyright 2022 EDF
"""
Class to define the Peak benchmark problem.

@author: efekhari27
"""

from ctbenchmark.CentralTendencyBenchmarkProblem import CentralTendencyBenchmarkProblem
import openturns as ot
import numpy as np


class CentralTendencyPeakProblem(CentralTendencyBenchmarkProblem):
    def __init__(self):
        """
        
        Example
        -------
        problem  = CentralTendencyPeakProblem()
        """

        func = '10 * exp(- 5 * abs(x1^2 - 0.5) - 5 * abs(x2 - 0.5))'
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

        name = "CT_Peak"
        # References computed by a very large Sobol sample (size 10**8)
        mean = 2.510688027976321
        #Trials with large MC have a hard time converging
        #      2.5111244511326256
        #      2.5107382492861676
        std = 1.8291704220412988

        super(CentralTendencyPeakProblem, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
