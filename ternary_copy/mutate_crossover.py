import sys
import math
import random
from itertools import count
from random import choice

from genome import DefaultGenomes
from gene import DefaultNodeGene, DefaultConnectionGene
from spices import Species, DefaultSpeciesSet
from network import FeedForwardNetwork
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

class DefaultReproduction:
    """
    交叉クラス
    """

    def __init__(self):
        self.genome_indexer = count(1) # ゲノム番号のインデクサ

    @staticmethod
    def compute_species_size(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """
        種のサイズを計算する

        :param adjusted_fitness: 調整適応度
        :param previous_sizes: 前の世代の種のサイズ
        :param pop_size: 個体数
        :param min_species_size: 種の最小個体数
        :return: 種のサイズ
        """
        adjusted_fitness_sum = sum(adjusted_fitness)
        next_species_size = []
        for adjusted, previous in zip(adjusted_fitness, previous_sizes):
            size = max(min_species_size, int(adjusted / adjusted_fitness_sum * pop_size))
            next_species_size.append(size)

        return next_species_size
    
    def configure_crossover(self, genome1, genome2, child):
        """
        交叉の設定

        :param genome1: 親1 (適応度が高い方)
        :param genome2: 親2 (適応度が低い方)
        :param child: 子
        """

        # 適応度が高い方を親1、低い方を親2とする
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # 交叉:コネクション遺伝子
        for key, connction_gene1 in parent1.connections.items():
            connction_gene2 = parent2.connections.get(key)
            if connction_gene2 is None:
                child.connections[key] = connction_gene1.copy()
            else:
                child.connections[key] = connction_gene1.crossover(connction_gene2)

        # 交叉:ノード遺伝子
        parent1_nodes = parent1.nodes
        parent2_nodes = parent2.nodes

        for key, node_gene1 in parent1_nodes.items():
            node_gene2 = parent2_nodes.get(key)
            if node_gene2 is None:
                child.nodes[key] = node_gene1.copy()
            else:
                child.nodes[key] = node_gene1.crossover(node_gene2)

    def update(self, species, generation):
        species_data = []
        # print(species)

        for specie_id, specie in species.species.items():
            if specie.fitness_history:
                prev_fitness = max(specie.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            specie.fitness = species_fitness_func(specie.get_fitnesses())
            specie.fitness_history.append(specie.fitness)
            specie.adjusted_fitness = None
            if prev_fitness is None or specie.fitness > prev_fitness:
                specie.last_improved = generation

            species_data.append((specie_id, specie))

        species_data.sort(key=lambda x: x[1].fitness)

        # 停滞の種を求める
        result = []
        num_non_stagnant = len(species_data)
        for idx, (specie_id, specie) in enumerate(species_data):
            stagment_time = generation - specie.last_improved
            is_stagnant = False

            if num_non_stagnant > spice_elitism:
                is_stagnant = stagment_time >= max_stagnation
            
            if (len(species_data) - idx) <= spice_elitism:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((specie_id, specie, is_stagnant))

        return result
    
    def reproduce(self, species, pop_size, generation):
        """
        交叉

        :param species: 種
        :param pop_size: 個体数
        :param generation: 世代
        :return: 次世代のゲノム
        """
        all_fitness = []
        remaining_species = []
        passing = None

        for stagnation_speice_id, stagnation_speice, is_stagnant in self.update(species, generation):
            if is_stagnant:
                # print(f"Species {stagnation_speice_id} is stagnated")
                pass
            else:
                all_fitness.extend(m.fitness for m in stagnation_speice.members.values())
                remaining_species.append(stagnation_speice)

        if not remaining_species:
            print("All species are stagnated.")
            species.species = {}
            return {}
        
        min_fitness = min(all_fitness)
        max_fitness = max(all_fitness)

        fitness_range = max(1.0, max_fitness - min_fitness)

        # 調整適応度
        for spice in remaining_species:
            mean_fitness = sum(m.fitness for m in spice.members.values()) / len(spice.members)
            adjuested_fitness = (mean_fitness - min_fitness) / fitness_range
            spice.adjusted_fitness = adjuested_fitness

        adjuested_fitness = [s.adjusted_fitness for s in remaining_species]
        species_sizes = self.compute_species_size(adjuested_fitness, [len(s.members) for s in remaining_species], pop_size, min_species_size)

        new_genomes = {}
        species.species = {}
        for specie_size, specie in zip(species_sizes, remaining_species):
            specie_size = max(specie_size, elitism)

            old_members = list(specie.members.items())
            specie.members = {}
            species.species[specie.key] = specie

            old_members.sort(key=lambda x: x[1].fitness, reverse=True)

            # エリート個体のコピー
            for idx, genome in old_members[:elitism]:
                new_genomes[idx] = genome
                specie_size -= 1

            if specie_size <= 0:
                continue

            # エリート個体の選択
            elite_count = int(survival_threshold * len(old_members))
            elite_count = max(elite_count, 2)
            old_members = old_members[:elite_count]

            # 交叉
            while specie_size > 0:
                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                child_id = next(self.genome_indexer)
                child = DefaultGenomes(child_id)
                self.configure_crossover(parent1, parent2, child)
                mutate = Mutate()
                mutate.mutate(child)
                new_genomes[child_id] = child
                specie_size -= 1

        return new_genomes

class Mutate:
    """
    突然変異クラス
    """
    def __init__(self):
        pass

    def mutate(self, genome):
        """
        突然変異

        :param genome: ゲノム
        """
        if random.random() < 0.2:
            self.mutate_delete_node(genome)
        
        if random.random() < 0.2:
            self.mutate_add_node(genome)

        if random.random() < 0.5:
            self.mutate_delete_connection(genome)
        
        if random.random() < 0.5:
            self.mutate_add_connection(genome)

        for connection_gene in genome.connections.values():
            connection_gene.mutate()

        for node_gene in genome.nodes.values():
            node_gene.mutate()
    
    def mutate_delete_node(self, genome):
        """
        ノードの削除

        :param genome: ゲノム
        :return: 削除されたノード，削除されなかった場合は-1
        """
        output_keys = [i for i in range(num_outputs)]
        available_nodes = [i for i in genome.nodes.keys() if i not in output_keys]

        if not available_nodes:
            return -1
        
        del_key = choice(available_nodes)

        connections_to_delete = set()

        for connection_key, connection_gene in genome.connections.items():
            if del_key in connection_gene.key:
                connections_to_delete.add(connection_gene.key)

        for connection_key in connections_to_delete:
            del genome.connections[connection_key]

        del genome.nodes[del_key]

        return del_key

    def mutate_add_node(self, genome):
        """
        ノードの追加

        :param genome: ゲノム
        """
        if not genome.connections:
            return
        
        connection_gene = choice(list(genome.connections.values()))
        if genome.node_indexer is None:
            genome.node_indexer = count(max(list(genome.nodes.keys())) + 1)
        new_node_id = next(genome.node_indexer)
        new_node_gene = DefaultGenomes.create_node(new_node_id)
        genome.nodes[new_node_id] = new_node_gene
        connection_gene.enabled = False

        in_node, out_node = connection_gene.key
        self.add_connection(genome, in_node, new_node_id, 1.0, True)
        self.add_connection(genome, new_node_id, out_node, connection_gene.weight, True)

    def add_connection(self, genome, in_node, out_node, weight, enabled):
        """
        接続の追加

        :param genome: ゲノム
        :param in_node: 入力ノード
        :param out_node: 出力ノード
        :param weight: 重み
        :param enabled: 有効かどうか
        """
        key = (in_node, out_node)
        connction = DefaultConnectionGene(key)
        connction.init_attributes()
        connction.weight = weight
        connction.enabled = enabled
        genome.connections[key] = connction

    def mutate_delete_connection(self, genome):
        """
        接続の削除

        :param genome: ゲノム
        """
        if genome.connections:
            del_key = choice(list(genome.connections.keys()))
            del genome.connections[del_key]

    def mutate_add_connection(self, genome):
        """
        接続の追加

        :param genome: ゲノム
        """
        possible_outputs = list(genome.nodes.keys())
        out_node = choice(possible_outputs)
        input_keys = [-i - 1 for i in range(num_inputs)]
        possible_inputs = possible_outputs + input_keys
        in_node = choice(possible_inputs)

        key = (in_node, out_node)
        if key in genome.connections:
            return
        
        if in_node == out_node:
            return
        
        if (in_node, out_node) in genome.connections or (out_node, in_node) in genome.connections:
            return
        
        connection_gene = DefaultGenomes.create_connection(in_node, out_node)
        genome.connections[connection_gene.key] = connection_gene

if __name__ == '__main__':
    genomes = DefaultGenomes.create_new(DefaultGenomes, 10)
    FeedForwardNetwork.eval_genomes(genomes)
    species_set = DefaultSpeciesSet()
    species_set.speciate(genomes, 1)
    print(f"num_species: {len(species_set.species)}")

    reproduction = DefaultReproduction()
    genomes = reproduction.reproduce(species_set, 10, 1)
    FeedForwardNetwork.eval_genomes(genomes)
    species_set.speciate(genomes, 2)
    print(f"num_species: {len(species_set.species)}")