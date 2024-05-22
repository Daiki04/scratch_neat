import configparser

def read_default_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    default_config = config['DEFAULT']
    return default_config