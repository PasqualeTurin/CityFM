#!/usr/bin/env python3

import argparse
import math

from utils.pre_processing import *
from utils.training import *
from utils.models import *
from utils.properties import *

parser = argparse.ArgumentParser(description='OSM2Vec')
parser.add_argument("-c", type=str, required=True, help='City')

hp = parser.parse_args()
filter_w()

check_data_exists(hp.c)

if not os.path.isdir(hp.c + '/Model'):
    os.mkdir(hp.c + '/Model')

cd = CityData(hp.c)
voc = build_category_vocab(cd.entities)
n_polylines = len(set(cd.data.keys()))
model = SSModel(hp.c, config.lm, 768)

train_SS_im(cd, model)
model = read_model(hp.c)
train_SS_raster(cd, model)
model = read_model(hp.c)
train_SS_rel(cd, model)
