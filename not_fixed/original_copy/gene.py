'''
遺伝子クラス

※遺伝子はゲノムに内包される
'''

from random import random, gauss
import numpy as np

# weights = [-1, 0, 1]

class BaseGene:
    '''
    遺伝子の基底クラス
    '''

    def __init__(self, key):
        '''
        コンストラクタ

        :param key: 遺伝子のキー（ノードではノード番号，接続ではイノベーション番号）
        '''
        self.key = key

    def __str__(self):
        '''
        文字列化

        :return: 遺伝子情報の文字列
        '''
        attrib = ['key'] + [a for a in self._gene_attributes]
        attrib = [f'{a} = {getattr(self, a)}' for a in attrib]
        return f'{self.__class__.__name__}({", ".join(attrib)})'

    def __lt__(self, other):
        '''
        比較演算子（<）

        :param other: 比較対象
        :return: 比較結果
        '''
        return self.key < other.key

    def mutate(self):
        '''
        突然変異
        '''
        for a in self._gene_attributes:
            v = getattr(self, a)
            if a == 'enabled':
                r = random()
                if r < 0.01:
                    v = random() < 0.5
                setattr(self, a, v)
            else:
                r = random()
                if r < 0.9:
                    v += gauss(0, 1.0)
                    # v = np.random.choice(weights)
                else:
                    v = gauss(0, 1)
                    # v = np.random.choice(weights)
                setattr(self, a, v)

    def copy(self):
        '''
        複製

        :return: 複製された遺伝子
        '''
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a, getattr(self, a))

        return new_gene

    def crossover(self, other):
        '''
        交叉

        :param other: 交叉対象
        :return: 交叉された遺伝子
        '''
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() < 0.8:
                setattr(new_gene, a, getattr(self, a))
            else:
                setattr(new_gene, a, getattr(other, a))
        return new_gene


class DefaultNodeGene(BaseGene):
    '''
    ノード遺伝子
    '''

    _gene_attributes = ['bias']

    def __init__(self, key):
        '''
        コンストラクタ

        :param key: ノード番号
        '''
        BaseGene.__init__(self, key)

    def init_attributes(self):
        '''
        属性の初期化
        '''
        self.bias = gauss(0, 1)
        # self.bias = np.random.choice(weights)
        # self.bias = np.random.choice(weights)


class DefaultConnectionGene(BaseGene):
    '''
    接続遺伝子
    '''

    _gene_attributes = ['weight', 'enabled']

    def __init__(self, key):
        '''
        コンストラクタ

        :param key: イノベーション番号
        '''
        BaseGene.__init__(self, key)

    def init_attributes(self):
        '''
        属性の初期化
        '''
        self.weight = gauss(0, 1)
        # self.weight = np.random.choice(weights)
        self.enabled = True