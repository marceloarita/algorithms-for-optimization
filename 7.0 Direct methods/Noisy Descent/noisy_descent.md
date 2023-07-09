# Cyclic Coordinate Search (CCS) - Notas de estudo

Escrito por Marcelo Arita \
Atualizado em: 2023-06-26

Refs: 
- Algorithm for optimization (MIT Press)

## 1. Breve introdução

Este método pertence à classe dos algoritmos diretos que se baseiam exclusivamente na função objetiva. 
O algoritmo de Cyclic Coordinate Search (CCS) é um método de otimização para obter o mínimo (ou máximo) de uma função
do espaço multidimensional. De forma simplificada, o algoritmo realiza ciclos de busca fixando uma variável por vez
enquanto otimiza as demais variáveis. Dentro de cada ciclo, as variáveis não fixas são otimizados usando métodos 
unidimensionais. Isto é repetido até que um critério de parada é atingido.
O CCS é idequado para problemas com variáveis independentese não correlacionadas.

