#!/usr/bin/python
# coding:utf-8
# Copyright 2022 EDF
"""
@authors: efekhari27
File description TO DO 
"""

from copy import deepcopy
import pandas as pd
import numpy as np
import openturns as ot
from itertools import product
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

#from KernelHerding import KernelHerding
import ctbenchmark as ctb
import otkerneldesign as otkd

ot.Log.Show(ot.Log.NONE)

class CentralTendencyBenchmark:
    """Class to define a central tendency benchmark"""
    def __init__(self, methods=['sobol', 'kernel herding','support points'], sizes=[10, 20, 50, 100, 500]):
        self.methods = methods
        self.sizes = sizes
        self.mc_ref_size = 10 ** 6
        self.scale_coefficient = 0.2
        self.level_names = ["Problem", "Method", "Size"]

    def set_kernel(self, dimension):
        ker_list = [ot.MaternModel([self.scale_coefficient], [1.0], 2.5)] * dimension
        self.kernel = ot.ProductCovarianceModel(ker_list)

    def set_candidate_pts(self, candidate_points):
        self.candidate_set = candidate_points

    def empty_ct_df(self, problem_name, method):
        muti_index = pd.MultiIndex.from_product([[problem_name], [method], self.sizes], names=self.level_names)
        df_ct_benchmark = pd.DataFrame([], index=muti_index, columns=['mu', 'm', 'm*', 'MMD', 'weights sum'])
        return df_ct_benchmark

    def generate_sample(self, method, size, distribution, candidate_points):
            if distribution is not None:
                dim = distribution.getDimension()
            else: 
                dim = candidate_points.getDimension()
            # Generate sample
            if method == 'monte carlo':
                x_sample = distribution.getSample(size)
            elif method == 'sobol':
                seq = ot.SobolSequence(dim)
                sobol_experiment = ot.LowDiscrepancyExperiment(seq, distribution, size, False)
                x_sample = sobol_experiment.generate()
            elif method == 'kernel herding':
                #scale = 1 / (size ** (1 / dim))
                self.set_kernel(dim)
                # Kernel herding design
                kh = otkd.KernelHerding(kernel=self.kernel, distribution=distribution, candidate_set=candidate_points)
                x_sample, _ = kh.select_design(size)
            elif method == 'support points':
                # Greedy support points design
                sp = otkd.GreedySupportPoints(distribution=distribution, candidate_set=candidate_points)
                x_sample = ot.Sample(sp.select_design(size))
            else :
                raise ValueError('The specified methods do not match the possible list ["monte carlo", "sobol", "kernel herding","support points"]')
            return x_sample

    def run_method(self, random_problem, method):
        function = random_problem.getFunction()
        distribution = random_problem.getDistribution()
        problem_name = random_problem.getName()
        print('START: problem=' + problem_name + ' | method=' + method)
        df_ct_benchmark = self.empty_ct_df(problem_name, method)
        df_ct_benchmark['mu'] = random_problem.getMean()
        if method in ['monte carlo', 'sobol']:
            # Generate sample of the max(sizes)
            full_x_sample = self.generate_sample(method, max(self.sizes), distribution, candidate_points=None)
        else:
            full_x_sample = self.generate_sample(method, max(self.sizes), distribution=None, candidate_points=self.candidate_set)
        # Compute observations
        full_y_sample = function(full_x_sample)
        # Compute iterative statistics and save them
        for size in self.sizes:
            y_sample = full_y_sample[:size]
            mean = np.mean(y_sample)
            x_sample = full_x_sample[:size]
            weights = self.compute_bayesian_quadrature_weights(x_sample, self.candidate_set)
            meanw = np.squeeze(y_sample) @ weights
            df_ct_benchmark.loc[(problem_name, method, int(size)), ['m', 'm*', 'weights sum']] = mean, meanw, np.sum(weights)
            #df_ct_benchmark.loc[(problem_name, method, size), ['m', 'm*', 'MMD', 'weights sum']] = mean, meanw, mmd, w_sum
        print('DONE: problem=' + problem_name + ' | method=' + method)
        return df_ct_benchmark

    def run_benchmark(self, random_problem, candidate_points):
        self.set_candidate_pts(candidate_points)
        p = Pool(cpu_count()-2)
        ct_results = p.starmap(self.run_method, product(random_problem, self.methods))
        p.close()
        df_ct_general_benchmark = pd.concat(ct_results)
        return df_ct_general_benchmark.sort_index(level=self.level_names)

    def compute_bayesian_quadrature_weights(self, x_sample, candidate_points):
        dim = x_sample.getDimension()
        size = x_sample.getSize()
        self.set_kernel(dim)
        global_sample = ot.Sample(x_sample)
        global_sample.add(candidate_points)
        cov_rows = np.zeros((size, global_sample.getSize()))
        for idx in range(size):
            cov_rows[idx, :] = self.kernel.discretizeRow(global_sample, int(idx)).asPoint()
        covmatrix = cov_rows[:, :size] + np.identity(size) * 1e-4
        # TODO: add a test on the condtitionning using np.cond(covmatrix)
        potentials = cov_rows[:, size:].mean(axis=1)
        return np.linalg.solve(covmatrix.T, potentials.T).T
    

######################################
    def plot_ct_benchmark(self, df_benchmark, function_label, methods=['sobol','support points'], save_file=None):
        df_benchmark = df_benchmark.reset_index()
        markers = "XD^Xo^v."
        fig = plt.figure(figsize=(8, 6))
        plt.title(function_label + ' problem', fontsize=20)
        for method in methods:
            df = df_benchmark[(df_benchmark["Problem"]==function_label) &
                                (df_benchmark["Method"]==method)]
            mu_ref = df['mu'].tolist()[0]
            idx = methods.index(method)
            plt.plot(df["Size"], df["m"], marker=markers[idx], label=method, color='C'+str(idx), zorder=2)
            plt.plot(df["Size"], df["m*"], marker=markers[idx], linestyle='dashed', label=method + ' weighted', color='C' + str(idx))
        #plt.scatter(df["Size"].max() + 1000, df['mu'].tolist()[0], marker='D', color='k')
        plt.axhline(y=mu_ref, color='k', linewidth=1.5, zorder=0)

        plt.grid(which='both')
        plt.xlabel('$n$ (log scale)', fontsize=18)
        plt.ylabel('Mean', fontsize=18)
        plt.xscale('log')
        plt.ylim([0.7 * mu_ref, 1.4 * mu_ref])
        legend = plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=3, columnspacing=1.2)
        if save_file is not None:
            fig.savefig(save_file, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig


    @staticmethod
    def build_kriging(x_learning_sample, y_learning_sample):
        """
        Creates an openturns kriging result from an imput and output sample

        Parameters
        ----------
        x_learning_sample: np.array
            Input sample
        y_learning_sample: np.array
            Output sample

        Returns
        -------
        ot.KrigingResult
            Object containing all the Kriging results 

        """
        x_learning_sample = ot.Sample(x_learning_sample)
        y_learning_sample = ot.Sample(y_learning_sample)
        dim = x_learning_sample.getDimension()
        covarianceModel = ot.MaternModel([1.0] * dim, 2.5)
        basis = ot.ConstantBasisFactory(dim).build()
        
        # Define Kriging
        my_kriging = ot.KrigingAlgorithm(x_learning_sample, y_learning_sample, covarianceModel, basis)
        my_kriging.setOptimizeParameters(True)
        my_solver = my_kriging.getOptimizationAlgorithm()
        
        # Optimization bounds
        my_bounds = my_kriging.getOptimizationBounds()
        lbounds = my_bounds.getLowerBound()
        ubounds = my_bounds.getUpperBound()
        
        # Uniform distribution for multistart starting points DoE
        uni_dist = ot.DistributionCollection()
        for i in range(dim):
            uni_dist.add(ot.Uniform(lbounds[i], ubounds[i]))
        uni_dist = ot.ComposedDistribution(uni_dist)
        
        # Multistart starting points DoE
        K = 25 # design size
        LHS = ot.LHSExperiment(uni_dist, K)
        LHS.setAlwaysShuffle(False)
        SA_profile = ot.GeometricProfile(10., 0.95, 20000)
        LHS_optimization_algo = ot.SimulatedAnnealingLHS(LHS, ot.SpaceFillingC2(), SA_profile)
        
        # Generate starting points DoE
        LHS_optimization_algo.generate()
        LHS_design = LHS_optimization_algo.getResult()
        starting_points = LHS_design.getOptimalDesign()
        
        # Define optimization solver
        my_solver.setMaximumIterationNumber(10000)
        multiStartSolver = ot.MultiStart(my_solver, starting_points)
        my_kriging.setOptimizationAlgorithm(multiStartSolver)
        my_kriging.run()
        return my_kriging.getResult()
    

if __name__=="__main__":
    peak_problem = ctb.CentralTendencyPicProblem()
    peak_function = peak_problem.getFunction()
    peak_dist = peak_problem.getDistribution()

    doe_generator = ctb.CentralTendencyBenchmark()
    candidate_points = doe_generator.generate_sample('sobol', 2**12, peak_dist, None)

    x_bench_sizes = list(range(5, 100, 5)) + list(range(100, 550, 50))
    bench = ctb.CentralTendencyBenchmark(['sobol', 'kernel herding'], x_bench_sizes)
    df_benchmark = bench.run_benchmark([peak_problem], candidate_points)