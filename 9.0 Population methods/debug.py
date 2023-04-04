%load_ext autoreload
%autoreload 2

from population_methods import *
import matplotlib.pyplot as plt
random.seed(42)

ga = GeneticAlgorithm(k=10)
f = lambda x: np.apply_along_axis(lambda y: np.linalg.norm(y), 1, x)
m = 1000
k_max = 10
population = ga.rand_population_uniform(m,a=[-3,3], b=[3,3])


def plot_distribution(pop_data: list, title: str) -> None:
    plt.scatter(pop_data[:,0], pop_data[:,1], s=10)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title(title)
    plt.show()
    return None

plot_distribution(population, title='Uniform distribution')

ga.genetic_algorithm(f=f,select=ga.truncation_selection, crossover=ga.single_point_crossover, mutate=ga.bit_wise_mutation, population=population, k_max=k_max)


for k in range(k_max):
    parents = ga.truncation_selection(y=f(population))
    children = [ga.single_point_crossover(parent_1=population[p[0]], parent_2=population[p[1]]) for p in parents]
    population = [ga.bit_wise_mutation(child, r=0.5) for child in children]
new_poulation = population[np.argmin(f(population))]





# pop_uniform = ga.rand_population_uniform(m, a, b)
# plot_distribution(pop_data=pop_uniform, title='Uniform distribution')

# # Uniform
# m = 1000
# a = [-2, -2]
# b = [2, 2]
# pop_uniform = ga.rand_population_uniform(m, a, b)
# plot_distribution(pop_data=pop_uniform, title='Uniform distribution')

# # Normal
# mean = [0,0]
# cov = [[1,0],[0,1]]
# pop_normal = ga.rand_population_normal(m, mean, cov)
# plot_distribution(pop_data=pop_normal, title='Normal distribution')

# # Cauchy
# scale = [1, 1]
# pop_cauchy = ga.rand_population_cauchy(m, mean, scale)
# plot_distribution(pop_data=pop_cauchy, title='Cauchy distribution')

rand_bit_pop = ga.generate_rand_bit_population(m=10)
y = [random.randint(0,100) for i in range(10)]

ga.truncation_selection(y=y)
ga.tournament_selection(y=y)
ga.roulette_wheel_selection(y=y)


parent_1 = ga.generate_rand_bit_population(m=10)
parent_2 = ga.generate_rand_bit_population(m=10)
child = parent_1
mut_child = [abs(1-v) if np.random.random() < 0.5 else v for v in child]

import numpy as np

def michalewicz(x, m=10):
    res = -np.sum(np.sin(x) * np.power(np.sin(np.arange(1, len(x)+1) * np.power(x, 2) / np.pi), 2*m))
    return res

x = [2.20, 1.57]
michalewicz(x)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly.offline import iplot

# Definir intervalo de valores para x e y
x = np.linspace(0, np.pi, 50)
y = np.linspace(0, np.pi, 50)

# Calcular o valor da função para cada ponto na grade
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        Z[i,j] = michalewicz([X[i,j], Y[i,j]])

# Criar um traçado 3D
surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')

# Criar um traçado para o marcador
point = go.Scatter3d(
    x=[2.20],
    y=[1.57],
    z=[michalewicz([2.20, 1.57])],
    mode='markers',
    marker=dict(
        size=5,
        color='red'
    )
)

# Definir o layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='x'),
        yaxis=dict(title='y'),
        zaxis=dict(title='michalewicz(x)')
    )
)

# Plotar o gráfico iterativo
fig = go.Figure(data=[surface, point], layout=layout)
iplot(fig)

