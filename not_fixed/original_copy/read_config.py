import configparser
import os
import sys

path_dir = os.path.dirname(os.path.abspath(__file__))

def read_default_config(config_file='config.ini'):
    config_file = os.path.join(path_dir, config_file)
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    default_config = config['DEFAULT']
    return default_config


if __name__ == '__main__':
    conf = read_default_config()
    fitness_threshold = float(conf.get('fitness_threshold'))
    print(type(fitness_threshold))