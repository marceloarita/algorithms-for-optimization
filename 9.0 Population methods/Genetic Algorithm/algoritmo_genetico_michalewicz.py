# %load_ext autoreload
# %autoreload 2

# ---------------------------------------------- 1. Importa bibliotecas ------------------------------------------------
import matplotlib.animation as animation
from algoritmo_genetico import *
from typing import Callable

# Reprodutibilidade
random.seed(0)

# ---------------------------------------------- 2. Algoritmo Genético -------------------------------------------------

# Gera populaçãoo inicial
gp = GeneratePopulations()

# 100 indivíduos com distribuição uniforme
m = 100
a = [-3, 3]
b = [3,3]
population = gp.rand_population_uniform(m=m, a=a, b=b)
gp.plot_distribution(pop_data=population, title='Uniform distribution')

# Funções de Seleção (S), Crossover (C), Mutação (M) e função objetiva (f)
S = SelectionMethods().roulette_wheel_selection
C = CrossoverMethods().single_point_crossover
M = MutationMethods().gauss_mutation
f = TestFunctions.michalewicz

# Algoritmo genético
def genetic_algorithm(f: Callable, select: Callable, crossover: Callable, mutate: Callable, population: list, 
                      k_max: int) -> list:
    
    # Salva todas populações em lista para fazer gif depois
    all_popluation_list = list()
    # Loop para iterar sobre as gerações
    for k in range(k_max):
        # Array de output da função objetiva
        y = [f(i) for i in population]
        parents    = select(y=y)
        children   = [crossover(parent_1=population[p[0]], parent_2=population[p[1]]) for p in parents]
        population = [mutate(sigma=0.1, child=child) for child in children]
        all_popluation_list.append(population)

    # Valor final funçã objetiva de todas as populações após k_max gerações
    y_final = [f(i) for i in population]
    # Objetivo é minimizar, logo deve-se obter a população que minimiza o valor (np.armin(y) -> índice da população)
    best_population = population[np.argmin(y_final)]
    return best_population, all_popluation_list

# Roda algoritmo genético
k_max = 20 
best_population, population_list = genetic_algorithm(f=f, select=S, crossover=C, mutate=M, population=population,
                                                     k_max=k_max)
print(f'Valor mínimo obtido: {f(best_population):.3f} (valor analítico: -1.8011)')
print(f'População: {best_population}')

# ------------------------------------------- 3. Cria gif das gearcoes -------------------------------------------------

# Utilizado chatGPT para essa parte

# Array dos eixos X, Y
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# Função que plota gráfico da função michalewicz
def michalewicz_graph(x, y):
    
    michalewicz = TestFunctions.michalewicz
    X, Y = np.meshgrid(x, y)
    Z = michalewicz([X, Y])

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel('Function Value')
    plt.title('Michalewicz Function')
    plt.xlabel('x')
    plt.ylabel('y')
    return fig, ax

# Cria gráfico da função de Michalewicz
fig, ax = michalewicz_graph(x, y)

# População de todas gerações
dot_positions = population_list
dots = ax.scatter([], [], color='black', marker='o')

# Função para criar animação em gif
def animate(frame):
    # População de cada geração (frame = geração, nesse caso)
    dots_x, dots_y = zip(*dot_positions[frame])

    # Atualiza pontos
    dots.set_offsets(np.c_[dots_x, dots_y])

    # Plota    
    ax.set_title(f'Função Michalewicz (Geração {frame+1})')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')

# Cria animação
anim = animation.FuncAnimation(fig, animate, frames=len(dot_positions), interval=100, blit=False)

# Salva animação
anim.save('algoritmo_genetico_michalewicz.gif', writer='pillow', dpi=80, fps=5)