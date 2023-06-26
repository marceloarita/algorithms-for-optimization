# Particle Swarm Optmization (PSO) - Notas de estudo

Escrito por Marcelo Arita \
Atualizado em: 2023-06-23

Refs: 
- Algorithm for optimization (MIT Press)
- https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/

## 1. Breve introdução

O Particle Swarm Optimization (PSO) pertence à classe dos algoritmos populacionais heurísticos, i.e: é impossível de
provar que a solução encontrada é o mínimo global (em geral não é), porém pode-se considerar como uma solução
satisfatória que é próxima ao mínimo global.
No PSO, cada partícula mantém a memória do melhor resultado já encontrado e através do compartilhamento de informações
entre as demais partículas, o grupo como um todo converge para a melhor solução.
Em cada interação, a velocidade de posição de cada partícula são atualizadas com base em duas influências: o melhor
resultado pessoal e o melhor resultado encontrado pelas demais partículas do enxame. Isso permite que as partículas
consigam explorar novos espaços de busca de forma cooperativa.

## 2. Conceitos chaves

- Posição: representa uma solução candidata para o problema de otimização. Em geral é um vetor.
- Velocidade: é a taxa de mudança da posição ao longo do tempo. Em geral é um vetor. A velocidade é ajustado por 
dois fatores: influência individual e coletiva. A primeira representa a tendência da partícula em se mover em direção à
sua melhor posição pessoal. A segunda indica a tendêcia da partícula em se mover rumo à melhor solução por qualquer 
partícula do enxame. Essas influêncis são ponderadas por por coeficientes.

## 3. Detalhes do algoritmo

Simplificando, o vetor de posição da partícila é atualizado conforme abaixo (I):
> (I) X(t+1) = X(t) + V(t+1)

E o vetor da velocidade é atualizado conforme abaixo (II):
> (II) V(t+1) = w\*V(t) + c1\*r1\*(pbest - X(t)) + c2\*r2\*(gbest - X(t))

onde: 
- t é a iteração
- w é a inércia da partícula alterar a sua velocidade no instante t
- r1 e r2 são randîomicos entre [0,1]
- c1 e c2 são coeficientes de influência pessoal e coletiva, respectivamente. Estes confiecientes controlam a taxa de 
influência que a partícula irá sofrer pela solução pessoal ou coletiva
- pbest e gbest são as posições respectivos às melhores soluções individual e do grupo, respectivamente. Estes também
são atualizados a cada iteração