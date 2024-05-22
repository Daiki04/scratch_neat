from itertools import count
import numpy as np

from genome import DefaultGenomes
from read_config import read_default_config

config = read_default_config()

compatibility_threshold = float(config.get('compatibility_threshold'))  # 類似度の閾値
initial_connection = config.get('initial_connection')  # 初期接続の種類
c1 = c2 = float(config.get('c1'))  # 類似度の計算係数
c3 = float(config.get('c3'))  # 類似度の計算係数


class Species:
    """
    種クラス
    """

    def __init__(self, key, generation):
        """
        コンストラクタ

        :param key: 種分化ID
        :param generation: 生成世代
        """
        self.key = key  # 種分化ID
        self.created = generation  # 生成世代
        self.last_improved = generation  # 最終改善世代
        self.representative = None  # 代表ゲノム
        self.members = {}  # ゲノム集団: {ゲノムID: ゲノム}
        self.fitness = None  # 平均適応度など
        self.adjusted_fitness = None  # 適応度の調整値
        self.fitness_history = []  # 適応度の履歴

    def update(self, representative, members):
        """
        種の更新

        :param representative: 代表ゲノム
        :param members: ゲノム集団
        """
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        """
        適応度の取得

        :return: 適応度
        """
        return [genome.fitness for genome in self.members.values()]


class DefaultSpeciesSet:
    """
    種集団のクラス
    """

    def __init__(self):
        """
        コンストラクタ
        """
        self.indexer = count(1)  # 種分化IDのインデクサ
        self.species = {}  # 種集団: {種のID: 種}
        self.genome_to_species = {}  # ゲノムと種の対応: {ゲノムID: 種ID}

    @staticmethod
    def distance(me, other):
        D = 0
        E = 0
        bar_W = 0
        N = max(len(me.connections), len(other.connections))
        N = N if N > 20 else 1

        me_connections_ids = set(me.connections.keys())
        other_connections_ids = set(other.connections.keys())

        D = len(me_connections_ids ^ other_connections_ids)
        E = 0

        for connection_id in me_connections_ids & other_connections_ids:
            bar_W = abs(me.connections[connection_id].weight -
                        other.connections[connection_id].weight)

        if len(me_connections_ids & other_connections_ids) > 0:
            bar_W = bar_W / (len(me_connections_ids & other_connections_ids))

        return (c1 * D) / N + (c2 * E) / N + c3 * bar_W

    def speciate(self, population, generation):
        unspaciated = set(population.keys())
        new_species_members = {}
        new_genome_to_species = {}
        for genome_id in population.keys():
            found = False
            for species_id, species in self.species.items():
                if self.distance(population[genome_id], species.representative) < compatibility_threshold:
                    if species_id not in new_species_members.keys():
                        new_species_members[species_id] = {}
                    new_species_members[species_id][genome_id] = population[genome_id]
                    new_genome_to_species[genome_id] = species_id
                    unspaciated.remove(genome_id)
                    found = True
                    break

            if not found:
                new_species_id = next(self.indexer)
                self.species[new_species_id] = Species(
                    new_species_id, generation)
                self.species[new_species_id].update(
                    population[genome_id], {genome_id: population[genome_id]})

                new_species_members[new_species_id] = {
                    genome_id: population[genome_id]}
                new_genome_to_species[genome_id] = new_species_id

        new_species = {}
        for species_id in new_species_members:
            new_species[species_id] = self.species[species_id]
            new_representative = np.random.choice(
                list(new_species_members[species_id].values()))
            new_species[species_id].update(
                new_representative, new_species_members[species_id])

        self.species = new_species

    def __str__(self):
        """
        文字列化

        種の情報を文字列化して返す
        種のIDと代表ゲノムの情報，ゲノム数，適応度，適応度の調整値を表示
        """
        info = ""
        for species_id, species in self.species.items():
            info += '=' * 120 + '\n'
            info += f'Species {species_id}:\n'
            info += f'Representative: {species.representative}\n'
            info += f'Members: {len(species.members)}\n'
            info += f'Fitness: {species.fitness}\n'
            info += f'Adjusted Fitness: {species.adjusted_fitness}\n'
        
        info += '=' * 120 + '\n'
        return info

if __name__ == '__main__':
    pop_size = 10
    genomes = DefaultGenomes.create_new(DefaultGenomes, pop_size)
    species_set = DefaultSpeciesSet()
    species_set.speciate(genomes, 1)
    print(species_set)