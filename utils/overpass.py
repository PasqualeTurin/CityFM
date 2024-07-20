import os
from os import listdir
from os.path import isfile, join
from io import open
import sys
import requests
from http.client import responses
import json
import time
from utils.pre_processing import *
from utils.errors import *
import config


def overpass_dl(city, type):

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = ' [out:json]; area[name="' + str(city) + '"]; (' + str(type) + '(area); ); out; '

    in_place_print('Downloading ' + str(type) + 's for ' + str(city) + '...')
    response = requests.get(overpass_url, params={'data': overpass_query}, stream=True, headers={'Connection':'close'})
    
    while response.status_code != 200:
    
        print(config.flush)
        print('Error ' + str(response.status_code) + ' - ' + responses[response.status_code])
        
        if response.status_code == 429:
            exit()
        
        print('Trying to recontact the server...')
        
        response = requests.get(overpass_url, params={'data': overpass_query})

    with open('tmp_dwl', 'wb') as f:
    
        for i, chunk in enumerate(response.iter_content(chunk_size=16384)):
        
            if chunk:
            
                in_place_print('Downloading ' + str(type) + 's for ' + str(city) + '... (' + str(16*(i)) + 'Kb)')
                f.write(chunk)
                f.flush()
    
    print(config.flush)
    response.connection.close()

    with open('tmp_dwl', 'rb') as f:
        data = json.load(f)
    
    os.remove('tmp_dwl')
    
    if not len(data['elements']):
        no_data_for(city)
        if os.path.isdir(city):
            if len(os.listdir(city)) == 0:
                os.rmdir(city)
            
        exit()
    
    else:
        print('Download of ' + str(len(data['elements'])) + ' ' + str(type) + '(s) completed!')
    
    return data['elements']

