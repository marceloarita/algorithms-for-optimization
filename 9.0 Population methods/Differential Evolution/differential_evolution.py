

import numpy as np
import matplotlib.pyplot as plt
import random

# Reprodutibilidade
random.seed(0)

def rand_population_uniform(m: int, a: list, b: list) -> np.array:
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

def plot_distribution(pop_data: list, title: str) -> None:
    plt.scatter(pop_data[:, 0], pop_data[:, 1], s=10)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title(title)
    plt.show()
    return None


def ackley(x, a=20, b=0.2, c=2*np.pi):
    # Mínimo global fica na origem (0,0)
    d = len(x)
    _sum_1 = np.sum(np.power(x, 2))
    _sum_2 = np.sum(np.cos(c * x))
    result = -a * np.exp(-b * np.sqrt(_sum_1 / d)) - np.exp(_sum_2 / d) + a + np.exp(1)
    return result

def michalewicz(x:list, m:int=10):
    # Para teste: x = [2.20, 1.57] -> result = -1.8011
    result = 0
    for i, v in enumerate(x):
        result -= np.sin(v) * np.sin((i+1) * v**2 / np.pi)**(2 * m)
    return result

m = 500
a = [-5, -5]
b = [5, 5]

pop = rand_population_uniform(m=m, a=a, b=b)
plot_distribution(pop_data=pop, title='uniform')

def select_three_candidates(population: np.array, j: int):
    candidates_idx = [c for c in range(population.shape[0]) if c!=j]
    idx = np.random.choice(a=candidates_idx, size=3, replace=False)
    a, b, c = pop[idx]
    return a, b, c

def mutation(a: list, b: list, c: list, w: float):
    z = a + w*(b-c)
    return z
    
def crossover(z, target, dims, cr):
    
    p = np.random.random(size=dims)
    new_one = [z[i] if p[i] < cr else target[i] for i in range(dims)]
    return new_one

population = pop
k_max = 20

y = [ackley(i) for i in population]

prev_score = min(y)
best_vector = population[np.argmin(y)]

for k in range(k_max):
    for j in range(population.shape[0]):
        target = population[j]
        a, b, c = select_three_candidates(population=population, j=j)
        z = mutation(a=a, b=b, c=c, w=0.2)
        trial = crossover(z=z, target=target, dims=len(pop[0]), cr=0.5)
        
        y_trial = ackley(x=np.array(trial))
        y_target = ackley(x=target) 
        if  y_trial < y_target:
            population[j] = trial
            y[j] = y_trial
    
    best_score = min(y)
    if best_score < prev_score:
        best_vector = population[np.argmin(y)]
        prev_score = best_score
        
    print(f'Geração {k+1} - menor y: {best_score:.3f}') 


ackley(np.array([0,0]))