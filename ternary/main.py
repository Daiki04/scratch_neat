import sys
import math
import random
from itertools import count
from random import choice

from genome import DefaultGenomes
from gene import DefaultNodeGene, DefaultConnectionGene
from spices import Species, DefaultSpeciesSet
from network import FeedForwardNetwork
from mutate_crossover import DefaultReproduction

elitism = 20 # それぞれの種である世代から次の世代にコピーされるエリート個体の数
spice_elitism = 3 # 停滞から保護される種の数を示します。例えば3と設定すると、種の適応度の最も高い3つの種が、たとえ改善が示されなかったとしても、停滞により削除されなくなります。
survival_threshold = 0.2 # それぞれの種で交叉に使われるエリート個体の割合
min_species_size = 2 # それぞれの種の最小個体数
species_fitness_func = max # 種の適応度を計算する関数:max, min, meanなど
max_stagnation = 30 # 種が停滞すると見なされる世代数

num_inputs = 2
num_outputs = 1

pop_size = 200
fitness_theashold = 3.9


if __name__ == '__main__':
    population = DefaultGenomes.create_new(DefaultGenomes, pop_size)
    species_set = DefaultSpeciesSet()
    generation = 0

    while True:
        ### 1.種に分ける
        species_set.speciate(population, generation)

        ### 2.適応度の評価
        FeedForwardNetwork.eval_genomes(population)

        best_genome_id, best_genome = max(population.items(), key=lambda x: x[1].fitness)

        print(f'Generation: {generation}, Best Genome: {best_genome_id}, Fitness: {best_genome.fitness}')

        if best_genome.fitness >= fitness_theashold:
            print('Threshold reached')
            break

        ### 3.交叉 & 突然変異
        reproduction = DefaultReproduction()
        population = reproduction.reproduce(species_set, pop_size, generation)

        generation += 1

    winner = best_genome
    winner_id = best_genome_id

    print(f'Winner: {winner_id}\n{winner}')
    print(f'Winner Fitness: {winner.fitness}')
    