#!/usr/bin/python
# coding:utf-8
# Copyright 2022 EDF
"""
Class to define the Cosin2 benchmark problem.

@author: efekhari27
"""

from ctbenchmark.CentralTendencyBenchmarkProblem import CentralTendencyBenchmarkProblem
import openturns as ot
import numpy as np


class CentralTendencyCosin2Problem(CentralTendencyBenchmarkProblem):
    def __init__(self):
        """

        The function is the following:
        

        We have x1 ~ Normal(0.5, 0.15) and x2 ~ Normal(0.5, 0.15) respectively truncated between 0. and 1.
        
        Example
        -------
        problem  = CentralTendencyCosin2Problem()
        """

        func = 'cos(10*x1) + sin(10*x2) + x1*x2'
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

        name = "CT_Cosin2"
        # References computed by a very large MC sample (size 10**8)
        mean = 0
        std = 0

        super(CentralTendencyCosin2Problem, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
