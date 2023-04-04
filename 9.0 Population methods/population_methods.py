import numpy as np
from numpy.random import standard_cauchy, multivariate_normal, randint
from typing import Callable
from random import getrandbits, sample
import random


class SelectionMethods():

    def __init__(self, k: int) -> None:

        self.k = k
        pass

    def truncation_selection(self, y: list) -> list:
        index = np.argsort(y)
        index_pairs = [[index[randint(0, self.k)]
                        for j in range(2)] for i in y]
        return index_pairs

    def tournament_selection(self, y: list) -> list:

        def getparent(y: list):
            y = np.array(y)
            p = np.random.permutation(len(y))
            index = p[np.argmin(y[p[:self.k]])]
            return index

        index_pairs = [[getparent(y), getparent(y)] for i in range(len(y))]
        return index_pairs

    def roulette_wheel_selection(self, y: list) -> list:
        y = [(max(y) - yi) for yi in y]
        cum_probs = np.cumsum(y)/sum(y)
        index_pairs = [[np.searchsorted(cum_probs, np.random.random()), np.searchsorted(
            cum_probs, np.random.random())] for i in y]
        return index_pairs


class CrossoverMethods():

    def __init__(self) -> None:
        pass

    def single_point_crossover(self, parent_1: list, parent_2: list) -> list:
        i = random.randint(0, len(parent_1))
        new_one = np.concatenate((parent_1[:i], parent_2[i:]))
        return new_one

    def two_point_crossover(self, parent_1: list, parent_2: list) -> list:
        n = len(parent_1)
        i, j = sorted(random.sample(range(n), 2))
        new_one = parent_1[:i] + parent_2[i:j] + parent_1[j:]
        return new_one

    def uniform_crossover(self, parent_1: list, parent_2: list) -> list:
        new_one = [parent_2[i] if random.random() < 0.5 else parent_1[i]
                   for i in range(len(parent_1))]
        return new_one


class MutationMethods():

    def __init__(self) -> None:
        pass

    def bit_wise_mutation(self, child: list, r: float):
        mut_child = [abs(1-v) if np.random.random() <
                     r else v for v in child]
        return mut_child

    # def gaussian_mutation(self, child: list, sigma: float):
    #     mut_child = child + np.random.randn(len(child)) * sigma
    #     return mut_child


class GeneticAlgorithm(SelectionMethods, CrossoverMethods, MutationMethods):

    def __init__(self, k: int) -> None:
        super().__init__(k=k)
        pass

    def rand_population_normal(self: 'GeneticAlgorithm', m: int, mean: list, cov: list) -> np.array:
        """
        TBD

        """
        samples = multivariate_normal(mean, cov, size=m)
        return samples

    def rand_population_uniform(self: 'GeneticAlgorithm', m: int, a: list, b: list) -> np.array:
        d = len(a)
        _samples = ([a[i] + np.random.random(d) * (b[i] - a[i])
                    for i in range(d) for j in range(m)])
        samples = np.array([arr.tolist() for arr in _samples])
        # samples = np.array(_).flatten().tolist()
        return samples

    def rand_population_cauchy(self, m: int, mean: list, scale: list) -> np.array:
        n = len(mean)
        sample = np.array([[standard_cauchy() * scale[j] + mean[j]
                          for j in range(n)] for i in range(m)])
        return sample

    def genetic_algorithm(f: Callable, select: Callable, crossover: Callable, mutate: Callable, population: list, k_max: int) -> list:
        for k in range(k_max):
            parents = select(y=f(population))
            children = [crossover(parent_1=population[p[0]], parent_2=population[p[1]]) for p in parents]
            population = [mutate(child, r=0.5) for child in children]
        # new_poulation = population[min(range(len(population)), key=lambda i: f(population[i]))]
        new_poulation = population[np.argmin(f(population))]
        return new_poulation

    def generate_rand_bit_population(self, m: int) -> list:
        rand_bit_pop = [getrandbits(1) for i in range(m)]
        return rand_bit_pop
