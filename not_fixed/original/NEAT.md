Here is the description of the NeuroEvolution of Augmenting Topologies (NEAT) algorithm with mathematical formulae, following the original paper by Stanley and Miikkulainen (2002):
1. Generation of the Initial Population
The initial population consists of $\mu$ individuals, each represented by a neural network with a minimal topology. Let $P_0$ denote the initial population.
Each individual $i$ in $P_0$ is characterized by its genotype, which is a tuple $(C_i, E_i, W_i)$, where:
$C_i$ is the set of nodes (neurons) in the network
$E_i$ is the set of edges (connections) between nodes
$W_i$ is the set of weights associated with each edge
The initial population is generated by creating $\mu$ random neural networks with a small number of nodes and edges.
2. Speciation
Speciation is the process of dividing the population into species based on their similarity. In NEAT, species are formed by clustering individuals with similar topologies.
Let $d(i, j)$ be the distance metric between two individuals $i$ and $j$, defined as:
𝑑
(
𝑖
,
𝑗
)
=
𝑐
1
𝐸
+
𝑐
2
𝐷
𝑁
d(i,j)= 
N
c 
1
​
 E+c 
2
​
 D
​
 
where:
$E$ is the number of excess genes (genes that are not present in both individuals)
$D$ is the number of disjoint genes (genes that are present in one individual but not the other)
$N$ is the total number of genes in the larger of the two individuals
$c_1$ and $c_2$ are constants that control the importance of excess and disjoint genes, respectively
Individuals are assigned to the same species if their distance is less than a certain threshold $\delta$. Let $S_k$ denote the $k^{th}$ species.
3. Evaluation of Individuals
Each individual in the population is evaluated using a fitness function $f(i)$, which assigns a fitness score to each individual based on its performance on a task.
Let $F_i$ denote the fitness of individual $i$.
4. Generation of New Populations by Selection, Crossover, and Mutation
The next generation of individuals is created through selection, crossover, and mutation.
Selection
The fittest individuals are selected to reproduce, based on their fitness scores. Let $P_{k+1}$ denote the new population.
The selection probability of individual $i$ is given by:
𝑝
𝑖
=
𝐹
𝑖
∑
𝑗
∈
𝑃
𝑘
𝐹
𝑗
p 
i
​
 = 
∑ 
j∈P 
k
​
 
​
 F 
j
​
 
F 
i
​
 
​
 
Crossover
Two parents are selected from the population, and their genotypes are combined to create a new offspring. Let $i$ and $j$ be the two parents, and let $o$ be the offspring.
The crossover operation is performed by taking the union of the nodes and edges of the two parents, and then randomly selecting which parent's weights to use for each edge.
Let $C_o = C_i \cup C_j$ and $E_o = E_i \cup E_j$.
For each edge $e \in E_o$, the weight is randomly selected from either $W_i$ or $W_j$.
Mutation
Each individual in the new population has a chance to undergo mutation. There are three types of mutations:
Node mutation: a new node is added to the network, and the incoming and outgoing edges are randomly assigned.
Edge mutation: a new edge is added between two existing nodes, and the weight is randomly assigned.
Weight mutation: the weight of an existing edge is perturbed by a small random value.
Let $m$ be the mutation rate, and let $r$ be a random variable uniformly distributed between 0 and 1.
For each individual $i$ in $P_{k+1}$, the mutation operation is performed with probability $m$:
\text{mutate}(i)&\text{if }r<m\\i&\text{otherwise}\end{cases}$$Theprocessisrepeatedformultiplegenerations,withthepopulationsizeremainingconstantat$\mu$.References:Stanley,K.O.,&Miikkulainen,R.(2002).Evolvingneuralnetworksthroughaugmentingtopologies.EvolutionaryComputation,10(2),99-127.