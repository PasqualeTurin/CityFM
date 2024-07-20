import config
from utils.pre_processing import *
import warnings


def preprocessed_data_not_found(city):
    print('Data for ' + str(city) + ' not found. Please use "python OSM2Vec_preprocess -c ' + str(city) + '"')
    exit()


def ds_data_not_found(city, task):
    print('Downstream data (' + task + ') for ' + str(city) + ' not found. Please download it and try again.')
    exit()


def no_data_for(city):
    print('No data found for ' + str(city))


def check_ds_task_exists(task):
    if task not in config.downstream_tasks:
        print('Task ' + str(task) + ' not found. Please use one of the following tasks:')
        for t in config.downstream_tasks:
            print('- ' + str(t))

        exit()

    return


def check_data_exists(city):
    if not os.path.isfile(city + '/Data/data.json') or not os.path.isfile(city + '/Data/entities.json'):
        preprocessed_data_not_found(city)


def check_ds_data_exists(city, task):
    data_name = ''
    ppc_data_name = ''
    if task == 'avg_speed':
        data_name = 'uber_speed_data.csv'
        ppc_data_name = 'uber_speed_data_ppc.csv'

    elif task == 'build_func':
        data_name = 'land_use_data.csv'
        ppc_data_name = 'land_use.json'

    elif task == 'housing_price':
        data_name = 'housing_data.csv'
        ppc_data_name = 'housing_data.json'

    elif task == 'pop_density':
        data_name = 'pop_density.json'
        ppc_data_name = data_name

    if not os.path.isfile(city + '/Data/' + data_name) and not os.path.isfile(city + '/Data/' + ppc_data_name):
        ds_data_not_found(city, task)


def filter_w():
    warnings.filterwarnings('ignore')


def window_size_error():
    print('Window size should be >= 3')
    exit()


def downstream_check(city, task):
    check_ds_task_exists(task)
    check_ds_data_exists(city, task)
    check_data_exists(city)
