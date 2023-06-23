import numpy as np
from numpy.random import standard_cauchy, multivariate_normal, randint
from random import getrandbits
import random
import matplotlib.pyplot as plt

class TestFunctions():

    def __init__(self) -> None:
        pass
    
    def michalewicz(x:list, m:int=10):
        """
        Método para calcular função de Michalewicz. Esta função possui múltiplos mínimos locais

        """
        # Para teste: x = [2.20, 1.57] -> result = -1.8011
        result = 0
        for i, v in enumerate(x):
            result -= np.sin(v) * np.sin((i+1) * v**2 / np.pi)**(2 * m)
        return result

class SelectionMethods():

    def __init__(self) -> None:
        pass

    def truncation_selection(self, y: list, k: int) -> list:
        """
        Este método seleciona os melhores k indivíduos da população (melhor score da  função objetiva) para então 
        selecionar os pais dentro deste grupo de k. Pode haver repetição de pais.
        
        Args:
            - y (list): lista de score da população
            - k (int): quantidade de melhores indivíduos a ser selecionado
        
        Returns:
            - index_pairs (list): índice dos pais  
        """
        
        index = np.argsort(y)
        index_pairs = [[index[randint(0, k)] for j in range(2)] for i in y]
        return index_pairs

    def tournament_selection(self, y: list, k: int) -> list:
        """
        Este método seleciona k indivíduos da população de forma aleatória e seleciona o melhor indivíduo deste
        grupo de k. Pode haver repetição de pais.
        
        Args:
            - y (list): lista de score da população
            - k (int): quantidade de indivíduos a ser selecionado para torneio
        
        Returns:
            - index_pairs (list): índice dos pais
        """

        def getparent(y: list):
            y = np.array(y)
            p = np.random.permutation(len(y))
            index = p[np.argmin(y[p[:k]])]
            return index

        index_pairs = [[getparent(y), getparent(y)] for i in range(len(y))]
        return index_pairs

    def roulette_wheel_selection(self, y: list) -> list:
        """
        Este método atribui probabilidade de seleção para cada indivíduo de acordo com o seu score. Para função
        objetiva do tipo minimização, a probabilidade de seleção é inversamente proporsional ao score do indivíduo 
        Para função objetiva do tipo maximização, é diretamente proporcional ao score do indivíduo.
        
        Args:
            - y (list): lista de score da população
        
        Returns:
            - index_pairs (list): índice dos pais  
        """
        
        y = [(max(y) - yi) for yi in y]
        cum_probs = np.cumsum(y)/sum(y)
        index_pairs = [[np.searchsorted(cum_probs, np.random.random()), 
                        np.searchsorted(cum_probs, np.random.random())] for i in y]
        return index_pairs

class CrossoverMethods():

    def __init__(self) -> None:
        pass

    def single_point_crossover(self, parent_1: list, parent_2: list) -> list:
        """
        Realiza a operação de cruzamento de um ponto entre dois pais.
    
        Args:
            - parent_1 (list): Lista representando o primeiro pai.
            - parent_2 (list): Lista representando o segundo pai.
            
        Returns:
            - list: Lista resultante após a operação de cruzamento de um ponto.
                        
        """
        i = random.randint(0, len(parent_1))
        new_one = np.concatenate((parent_1[:i], parent_2[i:]))
        return new_one

    def two_point_crossover(self, parent_1: list, parent_2: list) -> list:
        
        """
        Realiza a operação de cruzamento de dois pontos entre dois pais.

        Args:
           - parent_1 (list): Lista representando o primeiro pai.
           - parent_2 (list): Lista representando o segundo pai.

        Returns:
           - list: Lista resultante após a operação de cruzamento de dois pontos.

        """
        
        n = len(parent_1)
        i, j = sorted(random.sample(range(n), 2))
        new_one = parent_1[:i] + parent_2[i:j] + parent_1[j:]
        return new_one

    def uniform_crossover(self, parent_1: list, parent_2: list) -> list:
        """
        Realiza o cruzamento uniforme entre dois indivíduos.

        Essa função recebe dois indivíduos, representados por listas `parent_1` e `parent_2`,
        e realiza o cruzamento uniforme entre eles. O cruzamento uniforme combina os genes
        dos pais selecionando aleatoriamente os genes de cada pai com igual probabilidade.

        Args:
            - parent_1 (list): O primeiro pai, representado por uma lista de genes.
            - parent_2 (list): O segundo pai, representado por uma lista de genes.

        Returns:
            - new_one (list): O indivíduo resultante do cruzamento uniforme.

        """

        new_one = [parent_2[i] if random.random() < 0.5 else parent_1[i] for i in range(len(parent_1))]
        return new_one


class MutationMethods():

    def __init__(self) -> None:
        pass

    def gauss_mutation(self, sigma: float, child: list) -> list:
        """
        Realiza uma mutação gaussiana em um indivíduo.

        Essa função realiza uma mutação gaussiana em um indivíduo representado pela lista `child`.
        A mutação gaussiana adiciona um ruído aleatório de distribuição normal (gaussiana) aos genes
        do indivíduo. O parâmetro `sigma` controla a magnitude do ruído adicionado.

        Args:
            - sigma (float): O desvio padrão (sigma) da distribuição normal para a mutação.
            - child (list): O indivíduo a ser mutado, representado por uma lista de genes.

        Returns:
            - mut_child (list): O indivíduo mutado.

        """
        mut_child = child + np.random.randn(len(child)) * sigma 
        return mut_child

class GeneratePopulations():

    def __init__(self) -> None:
        pass

    def rand_population_normal(self: 'GeneratePopulations', m: int, mean: list, cov: list) -> np.array:
        """
        Gera uma população aleatória com base em uma distribuição normal multivariada.

        Essa função gera uma população aleatória com `m` indivíduos, utilizando uma distribuição
        normal multivariada com média `mean` e matriz de covariância `cov`.

        Args:
            - m (int): O número de indivíduos a serem gerados na população.
            - mean (list): A lista de médias para cada variável do indivíduo.
            - cov (list): A matriz de covariância que descreve as relações entre as variáveis.

        Returns:
            - samples (np.array): Uma matriz numpy representando a população gerada.

        """
        samples = multivariate_normal(mean, cov, size=m)
        return samples

    def rand_population_uniform(self: 'GeneratePopulations', m: int, a: list, b: list) -> np.array:
        """
        Gera uma população aleatória com base em uma distribuição uniforme.

        Essa função gera uma população aleatória com `m` indivíduos, utilizando uma distribuição
        uniforme para cada variável do indivíduo. Cada variável é gerada dentro do intervalo definido
        pelos vetores `a` e `b`, onde `a` representa os valores mínimos e `b` representa os valores
        máximos permitidos para cada variável.

        Args:
            - m (int): O número de indivíduos a serem gerados na população.
            - a (list): O vetor de valores mínimos para cada variável do indivíduo.
            - b (list): O vetor de valores máximos para cada variável do indivíduo.

        Returns:
            - samples (np.array): Uma matriz numpy representando a população gerada.

        """
        
        d = len(a)
        _samples = ([a[i] + np.random.random(d) * (b[i] - a[i]) for i in range(d) for _ in range(m)])
        samples = np.array([arr.tolist() for arr in _samples])
        return samples

    def rand_population_cauchy(self, m: int, mean: list, scale: list) -> np.array:
        
        """
        Gera uma população aleatória com base em uma distribuição Cauchy.

        Essa função gera uma população aleatória com `m` indivíduos, utilizando uma distribuição
        Cauchy para cada variável do indivíduo. Cada variável é gerada com base na média `mean`
        e escala `scale` especificadas para cada variável.

        Args:
            - m (int): O número de indivíduos a serem gerados na população.
            - mean (list): A lista de médias para cada variável do indivíduo.
            - scale (list): A lista de escalas para cada variável do indivíduo.

        Returns:
            - samples (np.array): Uma matriz numpy representando a população gerada.

        """
        
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
