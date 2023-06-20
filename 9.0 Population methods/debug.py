%load_ext autoreload
%autoreload 2

from population_methods import *
random.seed(0)

gp = GeneratePopulations()

m = 100
a = [-3, 3]
b = [3,3]

population = gp.rand_population_uniform(m=1000, a=a, b=b)
gp.plot_distribution(pop_data=population, title='Uniform distribution')


S = SelectionMethods(k=10).roulette_wheel_selection
C = CrossoverMethods().single_point_crossover
M = MutationMethods().gauss_mutation
f = TestFunctions.michalewicz

def genetic_algorithm(f: Callable, select: Callable, crossover: Callable, mutate: Callable, population: list, 
                    k_max: int) -> list:
    for k in range(k_max):
        y = [f(i) for i in population]
        parents = select(y=y)
        children = [crossover(parent_1=population[p[0]], parent_2=population[p[1]]) for p in parents]
        population = [mutate(sigma=0.1, child=child) for child in children]
    y_final = [f(i) for i in population]
    best_population = population[np.argmin(y_final)]
    return best_population

best_population = genetic_algorithm(f=f, select=S, crossover=C, mutate=M, population=population, k_max=100)
best_population

x = [population[i][0] for i in range(len(population))]
y = [population[i][1] for i in range(len(population))]
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
z = [f(i) for i in population]


# Generate multiple sets of dots for each frame
num_frames = 100
# Create a contour plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(x, y, z, cmap='viridis')
cbar = plt.colorbar(contour)
cbar.ax.set_ylabel('Function Value')


