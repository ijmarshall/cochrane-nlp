import os
import ConfigParser # easy_install configparser

config = ConfigParser.ConfigParser()

PATH = os.path.dirname(os.path.abspath(__file__))

config.read(os.path.join(PATH, 'CNLP.INI'))
