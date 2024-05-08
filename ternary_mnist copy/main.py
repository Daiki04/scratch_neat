import sys
import math
import random
from itertools import count
from random import choice
import tensorflow as tf
import numpy as np
import copy

from genome import DefaultGenomes
from gene import DefaultNodeGene, DefaultConnectionGene
from spices import Species, DefaultSpeciesSet
from network import FeedForwardNetwork
from mutate_crossover import DefaultReproduction


elitism = 20 # それぞれの種である世代から次の世代にコピーされるエリート個体の数
spice_elitism = 3 # 停滞から保護される種の数を示します。例えば3と設定すると、種の適応度の最も高い3つの種が、たとえ改善が示されなかったとしても、停滞により削除されなくなります。
survival_threshold = 0.2 # それぞれの種で交叉に使われるエリート個体の割合
min_species_size = 10 # それぞれの種の最小個体数
species_fitness_func = max # 種の適応度を計算する関数:max, min, meanなど
max_stagnation = 60 # 種が停滞すると見なされる世代数

num_inputs = 1
num_outputs = 784

pop_size = 200
fitness_theashold = 0.95

def _create_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    train_indices = np.where((y_train == 0) | (y_train == 1))[0]
    test_indices = np.where((y_test == 0) | (y_test == 1))[0]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]
    # データ数が多いので削減
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    population = DefaultGenomes.create_new(DefaultGenomes, pop_size)
    species_set = DefaultSpeciesSet()
    generation = 0
    (x, y), (x_test, y_test) = _create_dataset()

    while True:
        ### 1.種に分ける
        species_set.speciate(population, generation)

        ### 2.適応度の評価
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, axis=-1, reduction="none")
        acc = tf.keras.metrics.BinaryAccuracy()
        FeedForwardNetwork.eval_genomes(population, x, y, loss, acc)

        best_genome_id, best_genome = max(population.items(), key=lambda x: x[1].fitness)

        print(f'Generation: {generation}, Best Genome: {best_genome_id}, Fitness: {best_genome.fitness}, Acc: {best_genome.acc}')

        if best_genome.acc >= fitness_theashold:
            print('Threshold reached')
            break

        ### 3.交叉 & 突然変異
        reproduction = DefaultReproduction()
        population = reproduction.reproduce(species_set, pop_size, generation)

        if not species_set.species:
            print("Create New Population")
            population = DefaultGenomes.create_new(DefaultGenomes, pop_size)

        generation += 1

    winner = best_genome
    winner_id = best_genome_id

    print(f'Winner: {winner_id}\n{winner}')
    print(f'Winner Fitness: {winner.fitness}')
    print(f'Winner Acc: {winner.acc}')
    