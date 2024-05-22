import math
import tensorflow as tf

from read_config import read_default_config

config = read_default_config()

xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs = [(0,), (1,), (1,), (0,)]

fitness_theashold = float(config.get('fitness_threshold'))

num_inputs = int(config.get('num_inputs'))
num_outputs = int(config.get('num_outputs'))


class FeedForwardNetwork:
    """
    フィードフォワードネットワーククラス
    """

    def __init__(self, input_nodes, output_nodes, node_evals):
        """
        コンストラクタ

        :param inputs: 入力ノード
        :param outputs: 出力ノード
        :param node_evals: ノードの評価式
        """
        self.input_nodes = input_nodes  # 入力ノード
        self.output_nodes = output_nodes  # 出力ノード
        self.node_evals = node_evals  # ノードの評価式
        self.node_values = dict((key, 0.0)
                                for key in input_nodes + output_nodes)  # ノードの値

    @staticmethod
    def eval_genomes(genomes, x, target, loss, acc, use_x_num=1000):
        use_x_num = min(use_x_num, len(x))
        x = x[:use_x_num]
        target = target[:use_x_num]
        target = tf.convert_to_tensor(target)
        for _, genome in genomes.items():
            outputs = []
            pred_target = []
            net = FeedForwardNetwork.create(genome)
            for xi in x:
                output_values = net.activate(xi)
                outputs.append(output_values)
                pred_target.append(1 if output_values[0] > 0.5 else 0)
            outputs = tf.convert_to_tensor(outputs)
            pred_target = tf.convert_to_tensor(pred_target)
            outputs = tf.cast(outputs, tf.float32)
            pred_target = tf.cast(pred_target, tf.float32)
            genome.fitness = -float(loss(target, outputs).numpy().mean())
            genome.acc = float(acc(target, pred_target).numpy())

    @staticmethod
    def test_genome(genome, x, target, loss, acc):
        target = tf.convert_to_tensor(target)
        outputs = []
        pred_target = []
        net = FeedForwardNetwork.create(genome)
        for xi in x:
            output_values = net.activate(xi)
            outputs.append(output_values)
            pred_target.append(1 if output_values[0] > 0.5 else 0)
        outputs = tf.convert_to_tensor(outputs)
        pred_target = tf.convert_to_tensor(pred_target)
        outputs = tf.cast(outputs, tf.float32)
        pred_target = tf.cast(pred_target, tf.float32)
        fitness = -float(loss(target, outputs).numpy().mean())
        acc = float(acc(target, pred_target).numpy())
        return fitness, acc

    @staticmethod
    def create(genome):
        connections = [connction_genome.key for connction_genome in genome.connections.values() if connction_genome.enabled] # コネクション:使用可能
        output_node_keys = [i for i in range(num_outputs)]
        input_node_keys = [-i-1 for i in range(num_inputs)]
        layers = FeedForwardNetwork.feed_forward_layers(input_node_keys, output_node_keys, connections)
        node_evals = []

        # n層目のレイヤー→n層目のノード→n-1層目のノードとのコネクションをもとにノードを計算する順番でnode_evalsに格納
        for layer in layers:
            for current_node in layer:
                input_this_node = []
                for connection_key in connections:
                    in_node, out_node = connection_key
                    if current_node == out_node:
                        connection_gene = genome.connections[connection_key]
                        input_this_node.append((in_node, connection_gene.weight))

                current_node_gene = genome.nodes[current_node]
                node_evals.append((current_node, "sigmoid", "sum", current_node_gene.bias, 1.0, input_this_node))

        return FeedForwardNetwork(input_node_keys, output_node_keys, node_evals)

    def activate(self, input_values):
        """
        ネットワークの活性化

        :param input_values: 入力値
        :return: 出力値
        """

        # 入力ノードの値を設定：入力値
        for k, v in zip(self.input_nodes, input_values):
            self.node_values[k] = v

        # ノードの値を計算
        # aggregate function: sum
        # activation function: sigmoid
        for node, _, _, bias, response, links in self.node_evals:
            node_input_values = []
            for i, w in links:
                node_input_values.append(self.node_values[i] * w)
            y = sum(node_input_values)
            y = bias + response * y
            self.node_values[node] = 1.0 / (1.0 + math.exp(-y))

        return [self.node_values[i] for i in self.output_nodes]

    @staticmethod
    def required_for_output(input_nodes, output_nodes, connections):
        """
        最終的なネットワーク出力を計算するために計算が必要なノードを収集（出力ノードを含む）

        :param input_nodes: 入力ノード
        :param output_nodes: 出力ノード
        :param connections: コネクション
        """
        required = set(output_nodes)
        s = set(output_nodes)
        while True:
            t = set(a for (a, b) in connections if b in s and a not in s)
            if not t:
                break
            layer_nodes = set(x for x in t if x not in input_nodes)
            if not layer_nodes:
                break

            required = required.union(layer_nodes)
            s = s.union(t)

        return required

    @staticmethod
    def feed_forward_layers(input_nodes, output_nodes, connections):
        """
        フィードフォワードネットワークにおいて、並列評価可能なレイヤを収集

        :param inputs: 入力ノード
        :param outputs: 出力ノード
        :param connections: コネクション

        :return: レイヤの集合
        """
        required = FeedForwardNetwork.required_for_output(input_nodes, output_nodes, connections)
        layers = [] # レイヤの集合
        s = set(input_nodes)

        while 1:
            c = set(b for (a, b) in connections if a in s and b not in s)
            t = set()
            # 次のレイヤーの候補ノードcを見つける
            # s内のノードとs外のノードを繋ぐノードを探す
            for node in c:
                if node in required and all(a in s for (a, b) in connections if b == node):
                    t.add(node)

            if not t:
                break

            layers.append(t)
            s = s.union(t)

        return layers

        
if __name__ == '__main__':
    from genome import DefaultGenomes
    pop_size = 3
    genomes = DefaultGenomes.create_new(DefaultGenomes, pop_size)
    FeedForwardNetwork.eval_genomes(genomes)
    for a, b in list(genomes.items()):
        print(a, b)
        print(b.fitness)
        print()
