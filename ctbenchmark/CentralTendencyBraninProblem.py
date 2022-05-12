#!/usr/bin/python
# coding:utf-8
# Copyright 2021 EDF
"""
Class to define the Branin benchmark problem.

@author: efekhari27
"""

from ctbenchmark.CentralTendencyBenchmarkProblem import CentralTendencyBenchmarkProblem
import openturns as ot
import numpy as np


class CentralTendencyBraninProblem(CentralTendencyBenchmarkProblem):
    def __init__(self):
        """
        Creates a central tendency problem from the Branin function described in https://www.sfu.ca/~ssurjano/branin.html

        The function is the following:

        g(x_1, x_2) = h \circ  t(x_1, x_2)\\
        t(x_1, x2) = (15 x_1 - 5, 15 x_2)^T\\
        h(u_1, u_2) = \frac{\left(u_2-5.1\frac{u_1^2}{4\pi^2}+5\frac{u_1}{\pi}-6\right)^2+10\left(1-\frac{1}{8 \pi}\right)  \cos(u_1)+10-54.8104}{51.9496}\\

        We have x1 ~ Normal(0.5, 0.15) and x2 ~ Normal(0.5, 0.15) respectively truncated between 0. and 1.
        
        Example
        -------
        problem  = CentralTendencyBraninProblem()
        """

        noisy_branin = ot.SymbolicFunction(['x1', 'x2'],
                                    ['((x2 - (5.1 / (4 * pi_ ^ 2)) * x1 ^ 2 + 5 * x1 / pi_ - 6) ^ 2 + 10 * (1 - 1 / (8 * pi_)) * cos(x1) + 10 - 54.8104) / 51.9496', str(0.1)])
        transfo = ot.SymbolicFunction(['u1', 'u2'],
                                    ['15 * u1 - 5', '15 * u2'])
        model = ot.ComposedFunction(noisy_branin, transfo)
        g = model.getMarginal(0)

        X1 = ot.TruncatedDistribution(ot.Normal(0.5, 0.15), 0., 1.)
        X1.setDescription(["X1"])
        X2 = ot.TruncatedDistribution(ot.Normal(0.5, 0.15), 0., 1.)
        X2.setDescription(["X2"])

        myDistribution = ot.ComposedDistribution([X1, X2])
        inputRandomVector = ot.RandomVector(myDistribution)
        compositeRandomVector = ot.CompositeRandomVector(
            g, inputRandomVector
        )

        name = "CT_Branin_Problem"
        # References computed by a very large MC sample (size 10**8)
        mean = -0.3645488428936036
        std = 0.48805729585614865

        super(CentralTendencyBraninProblem, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
