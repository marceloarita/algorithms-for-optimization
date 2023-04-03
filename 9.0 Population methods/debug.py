%load_ext autoreload
%autoreload 2

from population_methods import *
import matplotlib.pyplot as plt

pm = PopulationMethods()

def plot_distribution(pop_data: list, title: str) -> None:
    plt.scatter(pop_data[:,0], pop_data[:,1], s=10)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title(title)``
    plt.show()
    return None


# Uniform
m = 1000
a = [-2, -2]
b = [2, 2]
pop_uniform = pm.rand_population_uniform(m, a, b)
plot_distribution(pop_data=pop_uniform, title='Uniform distribution')

# Normal
mean = [0,0]
cov = [[1,0],[0,1]]
pop_normal = pm.rand_population_normal(m, mean, cov)
plot_distribution(pop_data=pop_normal, title='Normal distribution')

# Cauchy
scale = [1, 1]
pop_cauchy = pm.rand_population_cauchy(m, mean, scale)
plot_distribution(pop_data=pop_cauchy, title='Cauchy distribution')

rand_bit_pop = pm.generate_rand_bit_population(m=10)
y = [random.randint(0,100) for i in range(10)]

pm.truncation_selection(y=y)
pm.tournament_selection(y=y)
pm.roulette_wheel_selection(y=y)
y = [5, 2, 7, 3, 9, 1, 8, 4, 6]
def getparent():
        p = random.sample(range(len(y)), 9)
        return p[np.argmin([y[i] for i in p])]
[[getparent(), getparent()] for i in y]



from numpy.random import standard_cauchy, multivariate_normal, random, randint
