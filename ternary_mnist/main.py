import sys
import math
import random
from itertools import count
from random import choice
import tensorflow as tf
import numpy as np
import copy
import pandas as pd

from genome import DefaultGenomes
from gene import DefaultNodeGene, DefaultConnectionGene
from spices import Species, DefaultSpeciesSet
from network import FeedForwardNetwork
from mutate_crossover import DefaultReproduction
from read_config import read_default_config

config = read_default_config()

species_fitness_func_dict = {
    'max': max,
    'min': min,
}

elitism = int(config.get('elitism'))
spice_elitism = int(config.get('spice_elitism'))
survival_threshold = float(config.get('survival_threshold'))
min_species_size = int(config.get('min_species_size'))
species_fitness_func = species_fitness_func_dict[config.get('species_fitness_func')]
max_stagnation = int(config.get('max_stagnation'))

num_inputs = int(config.get('num_inputs'))
num_outputs = int(config.get('num_outputs'))

pop_size = int(config.get('pop_size'))
fitness_theashold = float(config.get('fitness_threshold'))

def _create_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape([-1, 784]).astype('float32') / 255, x_test.reshape([-1, 784]).astype('float32') / 255

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    return (x_train, y_train), (x_test, y_test)

def print_config():
    print(f'elitism: {elitism}')
    print(f'spice_elitism: {spice_elitism}')
    print(f'survival_threshold: {survival_threshold}')
    print(f'min_species_size: {min_species_size}')
    print(f'species_fitness_func: {species_fitness_func}')
    print(f'max_stagnation: {max_stagnation}')
    print(f'num_inputs: {num_inputs}')
    print(f'num_outputs: {num_outputs}')
    print(f'pop_size: {pop_size}')
    print(f'fitness_theashold: {fitness_theashold}')

if __name__ == '__main__':
    population = DefaultGenomes.create_new(DefaultGenomes, pop_size)
    species_set = DefaultSpeciesSet()
    generation = 0
    (x, y), (x_test, y_test) = _create_dataset()

    best_fitness_hist = []
    best_acc_hist = []
    fitness_hist = []
    acc_hist = []
    best_test_fitness_hist = []
    best_test_acc_hist = []

    print_config()

    while True:
        train_indices = np.random.choice(len(x), 1000, replace=False)
        train_x = x[train_indices]
        train_y = y[train_indices]

        ### 1.種に分ける
        species_set.speciate(population, generation)

        ### 2.適応度の評価
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0,
            axis=-1,
            reduction=tf.keras.losses.Reduction.NONE
        )
        acc = tf.keras.metrics.CategoricalAccuracy()
        FeedForwardNetwork.eval_genomes(population, train_x, train_y, loss, acc)

        best_genome_id, best_genome = max(population.items(), key=lambda x: x[1].fitness)
        test_loss, test_acc = FeedForwardNetwork.test_genome(best_genome, x_test, y_test, loss, acc)

        fitness_hist.append(best_genome.fitness)
        acc_hist.append(best_genome.acc)
        if generation == 0:
            best_fitness_hist.append(best_genome.fitness)
            best_acc_hist.append(best_genome.acc)
            best_test_fitness_hist.append(test_loss)
            best_test_acc_hist.append(test_acc)
        else:
            if best_genome.fitness > best_fitness_hist[-1]:
                best_fitness_hist.append(best_genome.fitness)
                best_acc_hist.append(best_genome.acc)
                best_test_fitness_hist.append(test_loss)
                best_test_acc_hist.append(test_acc)
            else:
                best_fitness_hist.append(best_fitness_hist[-1])
                best_acc_hist.append(best_acc_hist[-1])
                best_test_fitness_hist.append(best_test_fitness_hist[-1])
                best_test_acc_hist.append(best_test_acc_hist[-1])

        print(f'Generation: {generation}, Number of Species: {len(species_set.species)}, Best Genome: {best_genome_id}, Fitness: {best_genome.fitness}, Acc: {best_genome.acc}, Test Loss: {test_loss}, Test Acc: {test_acc}')

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

    test_loss, test_acc = FeedForwardNetwork.test_genome(winner, x_test, y_test, loss, acc)

    # print(f'Winner: {winner_id}\n{winner}')
    print(f'Winner Fitness: {winner.fitness}')
    print(f'Winner Acc: {winner.acc}')
    print(f'Test Loss: {test_loss}')
    print(f'Test Acc: {test_acc}')

    # 結果の保存
    result = pd.DataFrame({
        'best_fitness': best_fitness_hist,
        'best_acc': best_acc_hist,
        'fitness': fitness_hist,
        'acc': acc_hist,
        'best_test_fitness': best_test_fitness_hist,
        'best_test_acc': best_test_acc_hist,
    })

    result.to_csv('result.csv', index=False)
    print('Saved result')

    # ネットワークの保存
    winner.save('winner.pkl')
    print('Saved winner')
    