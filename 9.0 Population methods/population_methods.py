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
        index_pairs = [[index[randint(0, self.k - 1)] for j in range(2)] for i in range(len(y))]
        return index_pairs
    
    def tournament_selection(self, y: list) -> list:
        
        def getparent():
            p = random.sample(range(len(y)), self.k)
            index = p[np.argmin([y[i] for i in p])]
            return index 
        
        index_pairs = [[getparent(), getparent()] for i in y]

        return index_pairs

    def roulette_wheel_selection(self, y: list) -> list:
        y = [(max(y) - yi) for yi in y]
        cum_probs = np.cumsum(y)/sum(y)
        index_pairs = [(np.searchsorted(cum_probs, np.random.random()), np.searchsorted(cum_probs, np.random.random())) for i in y]
        return index_pairs


class PopulationMethods(SelectionMethods):

    def __init__(self) -> None:
        super().__init__(k=10)
        pass

    def rand_population_normal(self: 'PopulationMethods', m: int, mean: list, cov: list) -> np.array:
        """
        TBD
        
        """
        samples = multivariate_normal(mean, cov, size=m)
        return samples

    def rand_population_uniform(self: 'PopulationMethods', m: int, a: list, b: list) -> np.array:
        d = len(a)
        _samples = ([a[i] + np.random.random(d) * (b[i] - a[i]) for i in range(d) for j in range(m)])
        samples = np.array([arr.tolist() for arr in _samples])
        # samples = np.array(_).flatten().tolist()
        return samples

    def rand_population_cauchy(self, m: int, mean: list, scale: list) -> np.array:
        n = len(mean)
        sample = np.array([[standard_cauchy() * scale[j] + mean[j] for j in range(n)] for i in range(m)])
        return sample

    def genetic_algorithm(f: Callable, select: Callable, crossover: Callable, mutate: Callable, population: list, k_max: int, S: int, C: float, M: float) -> list:
        for k in range(k_max):
            parents = select(S, [f(p) for p in population])
            children = [crossover(C, population[p[0]], population[p[1]]) for p in parents]
            population = [mutate(M, child) for child in children]
        new_poulation = population[min(range(len(population)), key=lambda i: f(population[i]))]
        return new_poulation

    def generate_rand_bit_population(self, m: int) -> list:
        rand_bit_pop = [getrandbits(1) for i in range(m)]
        return rand_bit_pop
    

