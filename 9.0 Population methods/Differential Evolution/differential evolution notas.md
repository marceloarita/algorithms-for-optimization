# Differential Evolution - Notas de estudo

Escrito por Marcelo Arita
Atualizado em: 2023-06-23

Ref: https://machinelearningmastery.com/differential-evolution-from-scratch-in-python/

## 1. Breve introdução

O algoritmo de Differential Evolution (DE) é um método numérico para resolver problemas de otimização global.
Assim como no algoritmo genético, o Differential Evolution (DE) é um algoritmo do tipo heurístico (obtém a solução 
satisfatória, mas não ideal/ótima) que pertence à família de algoritmos evolutivos. \
A fortaleza deste algoritmo reside no fato de conseguir resolver problemas não lineares e não diferenciáveis a partir
de um conjunto de indivíduos do espaço do vetor de busca. 
Normalmente, o algoritmo inicia-se com uma população inicial de candidatos gerados de forma randômica de modo a interar 
sobre esta população para explorar novas soluções potenciais.

## 2. Conceitos chaves

- Mutação: o algoritmo aplica o processo de mutação sobre cada indivíduo da população, que consiste na diferença 
ponderada entre vetores individuais selecionados aleatoriamente. A mutação gera um novo vetor no espaço de busca, que 
representa uma solução candidata modificada.
- Crossover: o vetor mutado é recombinado com vetor original para criar um vetor filho a fim de criar variabilidade
no espaço de busca.
-  Seleção:  o vetor filho é comparado com vetor original do indivíduo de acordo com a função objetiva. Aquele com
melhor desempenho será selecionado (mantido) para próxima geração.

Estes 3 passsos são repetiros exaustivamente até um número fixo de iterações ou quando até que um critério de parada
seja atingido. Em essência, esse loop aproximará da solução satisfatória do problema.

