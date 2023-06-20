import numpy as np
from numpy.random import standard_cauchy, multivariate_normal, randint
from typing import Callable
from random import getrandbits
import random
import matplotlib.pyplot as plt
import math


class TestFunctions():

    def __init__(self) -> None:
        pass
    
    def michalewicz(x:list, m:int=10):
        
        # x = [2.20, 1.57] -> result = -1.8011
        result = 0
        for i, v in enumerate(x):
            result -= np.sin(v) * np.sin((i+1) * v**2 / np.pi)**(2 * m)
        return result

class SelectionMethods():

    def __init__(self, k: int) -> None:

        self.k = k
        pass

    def truncation_selection(self, y: list) -> list:
        """
        Este método seleciona os melhores k indivíduos da população (melhor score da  função objetiva) para então 
        selecionar os pais dentro deste grupo de k. Pode haver repetição de pais.
        
        Args:
            - y: lista de score da população

        """
        
        index = np.argsort(y)
        index_pairs = [[index[randint(0, self.k)] for j in range(2)] for i in y]
        return index_pairs

    def tournament_selection(self, y: list) -> list:
        """
        Este método seleciona k indivíduos da população de forma aleatória e seleciona o melhor indivíduo deste
        grupo de k. Pode haver repetição de pais.
        
        Args:
            - y: lista de score da população


        """

        def getparent(y: list):
            y = np.array(y)
            p = np.random.permutation(len(y))
            index = p[np.argmin(y[p[:self.k]])]
            return index

        index_pairs = [[getparent(y), getparent(y)] for i in range(len(y))]
        return index_pairs

    def roulette_wheel_selection(self, y: list) -> list:
        """
        Este método atribui probabilidade de seleção para cada indivíduo de acordo com o seu score. Para função
        objetiva do tipo minimização, a probabilidade de seleção é inversamente proporsional ao score do indivíduo 
        Para função objetiva do tipo maximização, é diretamente proporcional ao score do indivíduo.
        
        Args:
            - y: lista de score da população

        """
        
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
        new_one = [parent_2[i] if random.random() < 0.5 else parent_1[i] for i in range(len(parent_1))]
        return new_one


class MutationMethods():

    def __init__(self) -> None:
        pass

    def bit_wise_mutation(self, child: list, r: float) -> list:
        """
        Realiza uma mutação de bits em um indivíduo de uma população.

        Args:
            child (list): Indivíduo a ser mutado, representado como uma lista de bits.
            r (float): Taxa de mutação, que determina a probabilidade de um bit ser mutado.

        Returns:
            list: Indivíduo mutado, representado como uma lista de bits.

        """
        
        mut_child = [abs(1-v) if np.random.random() < r else v for v in child]
        return mut_child

    def gauss_mutation(self, sigma: float, child: list) -> list:
        return child + np.random.randn(len(child)) * sigma


class GeneticAlgorithm(SelectionMethods, CrossoverMethods, MutationMethods):

    def __init__(self, k: int) -> None:
        super().__init__(k=k)
        pass

    def rand_population_binary(self, m: int) -> list:
        rand_bit_pop = [getrandbits(1) for i in range(m)]
        return rand_bit_pop

    def genetic_algorithm(self, f: Callable, select: Callable, crossover: Callable, mutate: Callable, population: list, k_max: int) -> list:
        for k in range(k_max):
            y = [f(i) for i in population]
            parents = select(y=y)
            children = [crossover(parent_1=population[p[0]], parent_2=population[p[1]]) for p in parents]
            population = [mutate(sigma=0.1, child=child) for child in children]
        # new_poulation = population[min(range(len(population)), key=lambda i: f(population[i]))]
        y_final = [f(i) for i in population]
        new_poulation = population[np.argmin(y_final)]
        return new_poulation



class GeneratePopulations():

    def __init__(self) -> None:
        pass

    def rand_population_normal(self: 'GeneratePopulations', m: int, mean: list, cov: list) -> np.array:
        """
        TBD

        """
        samples = multivariate_normal(mean, cov, size=m)
        return samples

    def rand_population_uniform(self: 'GeneticAlgorithm', m: int, a: list, b: list) -> np.array:
        d = len(a)
        _samples = ([a[i] + np.random.random(d) * (b[i] - a[i]) for i in range(d) for _ in range(m)])
        samples = np.array([arr.tolist() for arr in _samples])
        return samples

    def rand_population_cauchy(self, m: int, mean: list, scale: list) -> np.array:
        n = len(mean)
        samples = np.array([[standard_cauchy() * scale[j] + mean[j] for j in range(n)] for i in range(m)])
        return samples

    def plot_distribution(self, pop_data: list, title: str) -> None:

        plt.scatter(pop_data[:, 0], pop_data[:, 1], s=10)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title(title)
        plt.show()
        return None
