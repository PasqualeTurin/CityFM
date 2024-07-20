import argparse
from utils.overpass import overpass_dl
from utils.pre_processing import *


parser = argparse.ArgumentParser(description='OSM2Vec')
parser.add_argument("-c", type=str, default='Singapore', help='City')

hp = parser.parse_args()
dwl_city = map_city(hp.c)

if not os.path.isdir(hp.c):
    os.mkdir(hp.c)

if not os.path.isdir(hp.c+'/Nodes'):
    
    t0 = time.time()
    
    elements = overpass_dl(dwl_city, 'node')
    
    print('Saving...')
    
    os.mkdir(hp.c+'/Nodes')
    save_n_dict(elements, hp.c+'/Nodes/nodes_all.json')
    save_n_tagged(elements, hp.c+'/Nodes/nodes_tagged.json')
    
    release(elements)
    
    print('Completed! ('+str(round(time.time()-t0, 2))+'s)\n')
    

if not os.path.isdir(hp.c+'/Ways'):
    
    t0 = time.time()
    
    elements = overpass_dl(dwl_city, 'way')
    polygons, polylines = ways_split(elements)
    
    release(elements)
    
    os.mkdir(hp.c+'/Ways')
    
    dl_driving_network(hp.c)
    polygons = way_to_polygon(hp.c, polygons)
    polylines = way_to_polyline(hp.c, polylines)
    
    save(polygons, hp.c+'/Ways/polygons.json')
    save(polylines, hp.c+'/Ways/polylines.json')

    # polygon_to_raster(polygons, hp.c)
    
    print('Saving...')
    
    polygons = None
    release(polylines)
    
    print('Completed! ('+str(round(time.time()-t0, 2))+'s)\n')


if not os.path.isdir(hp.c+'/Relations'):
    
    t0 = time.time()
    
    elements = overpass_dl(dwl_city, 'relation')
    
    os.mkdir(hp.c+'/Relations')
    print('Saving...')
        
    save(elements, hp.c+'/Relations/relations.json')
    release(elements)
    
    print('Completed! ('+str(round(time.time()-t0, 2))+'s)\n')


if not os.path.isdir(hp.c+'/Data'):
    
    os.mkdir(hp.c+'/Data')


if not isfile(hp.c+'/Data/data.json'):

    t0 = time.time()
    
    prepare_data(hp.c)
    
    print('Completed! ('+str(round(time.time()-t0, 2))+'s)\n')
