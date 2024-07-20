import argparse
import math
import matplotlib.pyplot as plt
from utils.training import *
from utils.models import *
from utils.prediction import *

parser = argparse.ArgumentParser(description='OSM2Vec')
parser.add_argument("-c", type=str, required=True, help='City')
parser.add_argument("-t", type=str, required=True, help='Task')

hp = parser.parse_args()
filter_w()
reset_log_file()

check_data_exists(hp.c)
downstream_check(hp.c, hp.t)

buildings_u = False
if hp.t == 'build_func':
    buildings_u = True

cd = CityData(hp.c, buildings_u=buildings_u)
model = read_model(hp.c)

downstream_functions[hp.t](hp.c, cd, model, training=True)
