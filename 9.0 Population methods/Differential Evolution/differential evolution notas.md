# Differential Evolution - Notas de estudo

Escrito por Marcelo Arita
Atualizado em: 2023-06-23

Ref: https://machinelearningmastery.com/differential-evolution-from-scratch-in-python/

## 1. Breve introdução

O algoritmo Differential Evolution (DE) é um método numérico para resolver problemas de otimização global. Assim como o 
algoritmo genético, o Differential Evolution (DE) é um algoritmo heurístico que busca uma solução satisfatória, 
embora não necessariamente ótima. Ele pertence à família de algoritmos evolutivos.

A principal vantagem desse algoritmo é sua capacidade de resolver problemas não lineares e não diferenciáveis por meio 
de um conjunto de indivíduos no espaço de busca. Normalmente, o algoritmo começa com uma população inicial de candidatos 
gerados aleatoriamente e itera sobre essa população para explorar novas soluções potenciais.

## 2. Conceitos chaves

- Mutação: Mutação: o algoritmo aplica o processo de mutação a cada indivíduo da população, que consiste na diferença 
ponderada entre vetores individuais selecionados aleatoriamente. A mutação gera um novo vetor no espaço de busca, que 
representa uma solução candidata modificada.
- Crossover: o vetor mutado é recombinado com o vetor original para criar um vetor filho, a fim de criar variabilidade 
no espaço de busca..
-  Seleção: o vetor filho é comparado com o vetor original do indivíduo de acordo com a função objetivo. Aquele com 
melhor desempenho será selecionado (mantido) para a próxima geração.

Essas etapas são repetidas exaustivamente até um número fixo de iterações ou até que um critério de parada seja atingido. 
Em essência, esse loop se aproxima de uma solução satisfatória para o problema.