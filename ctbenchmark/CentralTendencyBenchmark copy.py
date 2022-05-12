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
        self.candidate_size = 2 ** 14
        self.scale_coefficient = 0.2
        self.level_names = ["Problem", "Method", "Size"]

    def empty_ct_df(self, problem_name, method):
        muti_index = pd.MultiIndex.from_product([[problem_name], [method], self.sizes], names=self.level_names)
        df_ct_benchmark = pd.DataFrame([], index=muti_index, columns=['mu', 'm', 'm*', 'MMD', 'weights sum'])
        return df_ct_benchmark

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
    
    def generate_sample(self, method, size, distribution, x_init=None, candidate_points=None):
            doe = ctb.DesignOfExperiments()
            if method == 'monte carlo':
                x_sample = np.array(distribution.getSample(size))
            elif method == 'sobol':
                x_sample = doe.sobol_sample(size, distribution)

            elif method == 'kernel herding':
                #dim = candidate_points.shape[1]
                #scale = 1 / (size ** (1 / dim))
                x_sample = doe.kernel_herding_sample(size, distribution, x_init, candidate_points, scale_coeff=self.scale_coefficient)
            elif method == 'support points':
                #x_test, _ = doe.support_points_sample(size, distribution, x_learn, candidate_points)
                x_sample, _ = doe.greedy_support_points_sample(size, distribution, x_init, candidate_points)
            else :
                raise ValueError('The specified methods do not match the possible list ["monte carlo", "sobol", "fssf", "kernel herding","support points"]')
            return x_sample

    def run_method(self, random_problem, method):
        function = random_problem.getFunction()
        distribution = random_problem.getDistribution()
        problem_name = random_problem.getName()
        print('START: problem=' + problem_name + ' | method=' + method)
        df_ct_benchmark = self.empty_ct_df(problem_name, method)
        df_ct_benchmark['mu'] = random_problem.getMean()
        # IF non adaptive (the design can only be built on the inputs)
        if method in ['monte carlo', 'sobol', 'kernel herding','support points']:
            # Generate sample of the max(sizes)
            full_x_sample = self.generate_sample(method, max(self.sizes), distribution)
            full_y_sample = function(full_x_sample)
            # Compute iterative statistics and save them
            for size in self.sizes:
                y_sample = full_y_sample[:size]
                mean = np.mean(y_sample)
                df_ct_benchmark.loc[(problem_name, method, int(size)), ['m']] = mean
                #df_ct_benchmark.loc[(problem_name, method, size), ['m', 'm*', 'MMD', 'weights sum']] = mean, meanw, mmd, w_sum

        # ELSE (the design construction depends iteratively on the output)


        print('DONE: problem=' + problem_name + ' | method=' + method)
        return df_ct_benchmark

    def run_benchmarks(self, random_problems):
        p = Pool(cpu_count()-2)
        ct_results = p.starmap(self.run_method, product(random_problems, self.methods))
        p.close()
        df_ct_general_benchmark = pd.concat(ct_results)
        return df_ct_general_benchmark.sort_index(level=self.level_names)

    
    @staticmethod
    def compute_weights(x_test, x_learn, candidate_points, dim, scale_coeff, nu_coeff=2.5, my_seed=1, residuals=None):
        learn_size = len(x_learn)
        test_size = len(x_test)
        candidate_size = len(candidate_points)
        # Build KernelHerding object
        ot.RandomGenerator.SetSeed(my_seed)
        #scale = [scale_coeff] * dim
        #kernel = ot.MaternModel(scale, nu_coeff)
        ker_list = [ot.MaternModel([scale_coeff], [1.0], nu_coeff)] * dim
        kernel = ot.ProductCovarianceModel(ker_list)
        kh = KernelHerding(kernel, candidate_points)
        # Add learning set and compute their indexes within the candidate points
        kh.add_candidate_points(x_learn)
        learn_indexes = np.arange(candidate_size, candidate_size + learn_size).tolist()
        # Add test set and compute their indexes within the candidate points
        kh.add_candidate_points(x_test)
        test_indexes = np.arange(candidate_size + learn_size, candidate_size + learn_size + test_size).tolist()
        #kh.recompute_covmatrix(I_kh=learn_indexes)
        weights = kh.compute_weigths(test_indexes, 'noconstraint', conditioned_to=learn_indexes, residuals=residuals)
        return weights

##############################################
    def plot_ct_benchmark(self, df_benchmark, function_label, methods=['sobol','support points'], save_file=None):
        df_benchmark = df_benchmark.reset_index()
        markers = "XD^Xo^v."
        #methods = df_benchmark["Validation method"].unique().tolist()
        fig = plt.figure(figsize=(8, 6))
        plt.title(function_label + ' problem central tendency estimation', fontsize=16)

        for method in methods:
            df = df_benchmark[(df_benchmark["Problem"]==function_label) &
                                (df_benchmark["Method"]==method)]
            idx = methods.index(method)
            plt.plot(df["Size"], df["m"], marker=markers[idx], label=method, color='C'+str(idx), zorder=2)
            #plt.plot(df["Size"], df["Q2*"], marker=markers[idx], linestyle='dashed', label=method + ' weighted', color='C' + str(idx))
            #plt.plot(df["Test size"], df["Q2**"], marker=markers[idx], linestyle='dotted', label=method+' weighted and variance on learn and test sample', color='C5')
        #plt.scatter(df["Size"].max() + 1000, df['mu'].tolist()[0], marker='D', color='k')
        plt.axhline(y=df['mu'].tolist()[0], color='k', linewidth=1.5, zorder=0)

        plt.grid(which='both')
        plt.xlabel('$n$ (log scale)', fontsize=16)
        plt.ylabel('${a}_n(g)$', fontsize=16)
        plt.xscale('log')
        legend = plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=3, columnspacing=1.2)
        if save_file is not None:
            fig.savefig(save_file, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig

    def plot_weights_sum(self, df_benchmark, learn_size, function_label, methods=['kernel herding','support points', 'fssf'], save_file=None):
        df_benchmark = df_benchmark.reset_index()
        markers = "Xo^v."
        #methods = df_benchmark["Validation method"].unique().tolist()
        fig = plt.figure(figsize=(8, 6))
        if function_label=='Irregular':
            f_label = 'Test case 1'
        elif function_label=='Cosin2':
            f_label = 'Test case 2'
        elif function_label=='gSobol8D':
            f_label = 'Test case 3'
        else :
            f_label = function_label
        plt.title(f_label + ' ($m={}$)'.format(learn_size), fontsize=16)
        for method in methods:
            try :
                df = df_benchmark[(df_benchmark["Learn size"]==learn_size) &
                                    (df_benchmark["Function"]==function_label) &
                                    (df_benchmark["Validation method"]==method)]
            except:
                df = df_benchmark[(df_benchmark["Learn size"]==learn_size) &
                                    (df_benchmark["Validation method"]==method)]
            idx = methods.index(method)
            if method=='fssf':
                method ='FSSF'
            plt.plot(df["Test size"], df["weights sum"], marker=markers[idx], label=method, color='C'+str(idx))
        #plt.axhline(y=df['Q2 MC'].tolist()[0], color='k', linewidth=1)
        test_values = df["Test size"].tolist()
        ticks = test_values
        ticks_labels = [str(x) for x in test_values]
        plt.grid(which='both')
        plt.xticks(ticks, ticks_labels)
        plt.xlabel('$n$', fontsize=16)
        plt.ylabel('$\sum_{i}^{n} w_i^*$', fontsize=16)
        #plt.ylim([0.2, 1.15])
        legend = plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=3, columnspacing=1.2)
        if save_file is not None:
            fig.savefig(save_file, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig
