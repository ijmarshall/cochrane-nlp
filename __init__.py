import os
import configparser # easy_install configparser

config = configparser.ConfigParser()

PATH = os.path.dirname(os.path.abspath(__file__))

config.read(os.path.join(PATH, 'CNLP.INI'))

