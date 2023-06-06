%load_ext autoreload
%autoreload 2

from population_methods import *
random.seed(42)

gp = GeneratePopulations()

# Uniform
m = 1000
a = [-2, -2]
b = [2, 2]
pop_uniform = gp.rand_population_uniform(m, a, b)
gp.plot_distribution(pop_data=pop_uniform, title='Uniform distribution')


# Normal
mean = [0,0]
cov = [[1,0],[0,1]]
pop_normal = gp.rand_population_normal(m, mean, cov)
gp.plot_distribution(pop_data=pop_normal, title='Normal distribution')

# Cauchy
scale = [1, 1]
pop_cauchy = gp.rand_population_cauchy(m, mean, scale)
gp.plot_distribution(pop_data=pop_cauchy, title='Cauchy distribution')


ga = GeneticAlgorithm(k=10)

bit_population = ga.rand_population_binary(m=10)
bit_population