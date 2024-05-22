'''
ゲノムクラス
'''
from itertools import count

from gene import DefaultNodeGene, DefaultConnectionGene

num_inputs = 2
num_outputs = 1

class DefaultGenomes:
    '''
    ゲノム集団のクラス
    '''

    def __init__(self, key):
        '''
        コンストラクタ

        :param key: ゲノムID（ユニーク）
        '''
        self.key = key
        self.connections = {} # コネクション遺伝子を格納
        self.nodes = {} # ノード遺伝子を格納
        self.fitness = None # 適応度
        self.node_indexer = None # ノード番号のインデクサ

    def create_new(genome_type, num_genomes):
        '''
        新規ゲノムの生成

        :param genome_type: ゲノムの種類
        :param num_genomes: ゲノムの数
        :return: ゲノム集団
        '''
        new_genomes = {}
        genome_indexer = count(1)
        for _ in range(num_genomes):
            key = next(genome_indexer)
            g = genome_type(key)
            output_keys = [i for i in range(num_outputs)]
            for node_key in output_keys:
                g.nodes[node_key] = DefaultGenomes.create_node(node_key)
            for input_id, output_id in DefaultGenomes.compute_full_conections(g):
                connection = DefaultGenomes.create_connection(input_id, output_id)
                g.connections[connection.key] = connection
            new_genomes[key] = g
        return new_genomes


    def create_node(node_id):
        '''
        ノードの生成

        :param node_id: ノード番号
        :return: ノード
        '''
        node = DefaultNodeGene(node_id)
        node.init_attributes()
        return node

    def compute_full_conections(genome):
        '''
        全結合の計算

        :param genome: ゲノム
        :return: 全結合
        '''
        output_keys = [i for i in range(num_outputs)]
        input_keys = [-i-1 for i in range(num_inputs)] # 入力ノードの番号は-1からはじまる負の値
        connections = []
        for input_id in input_keys:
            for output_id in output_keys:
                connections.append((input_id, output_id))
        return connections

    def create_connection(input_id, output_id):
        '''
        コネクションの生成

        :param input_id: 入力ノード番号
        :param output_id: 出力ノード番号
        :return: コネクション
        '''
        connection = DefaultConnectionGene((input_id, output_id))
        connection.init_attributes()
        return connection
    
    def size(self):
        '''
        ノード数と有効なコネクションの数を返す
        '''
        return len(self.nodes), sum(1 for c in self.connections.values() if c.enabled)
    
    def __str__(self):
        '''
        文字列化
        '''
        s = f'key: {self.key}, fitness: {self.fitness}\nNodes:'
        for k, ng in self.nodes.items():
            s += f'\n\t{k} {ng}'
        s += '\nConnections:'
        connections = list(self.connections.values())
        connections.sort()
        for cg in connections:
            s += f'\n\t{cg}'
        return s
    
if __name__ == '__main__':
    pop_size = 10
    genomes = DefaultGenomes.create_new(DefaultGenomes, pop_size)
    for a, b in list(genomes.items()):
        print(a)
        print(b)