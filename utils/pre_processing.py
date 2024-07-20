import os
from os import listdir
from os.path import isfile, join

import pandas as pd
# from matplotlib import pyplot as plt
# import pygeohash as pgh
from io import open
import sys
import numpy as np
import time
import math
from math import sin, cos, sqrt, atan2, radians
import json
import pickle
import random
from shapely import geometry, affinity
from shapely.geometry import Polygon, Point, LineString
from pyproj import Geod, Proj, transform
import rasterio.features
from difflib import SequenceMatcher
import config
import osmnx as ox
import networkx as nx
import utils.osm_strings as osm_strings
import utils.landuse_strings as lu_strings
import re


def prepare_data(city):

    with open(str(city) + '/Ways/polylines.json', 'r', encoding='utf-8') as f:
        polylines = json.load(f)

    poly_dictionary = polyline_id_mapping(city, polylines)
    ellipsoid = Geod(ellps='WGS84')

    polylines_dict = {}
    for p in polylines:
        if len(p["points"]) > 1 and check_polyline(p):

            length = 0
            points = p["points"]

            for i in range(len(points) - 1):
                try:
                    length += int(compute_dist(points[i][0], points[i][1], points[i+1][0], points[i+1][1]))
                except ValueError:
                    continue

            if length == 0:
                continue

            p['length'] = length
            polylines_dict[p["polyline"]["id"]] = p

    save(polylines_dict, city + '/Data/polylines.json')

    with open(str(city) + '/Nodes/nodes_tagged.json', 'r', encoding='utf-8') as f:
        nodes = json.load(f)

    with open(str(city) + '/Ways/polygons.json', 'r', encoding='utf-8') as f:
        buildings = json.load(f)

    with open(str(city) + '/Relations/relations.json', 'r', encoding='utf-8') as f:
        rel = json.load(f)

    # read polygons and concat with nodes
    buildings_untagged = {}

    for b in buildings:

        try:
            building = buildings[b]
            if check_building(building) and b not in nodes:
                points = building["points"]
                area = polygon_faceted_area(points, ellipsoid)
                nodes[b] = {"type": "polygon", "id": b, "lat": building['polygon']['center'][0], "lon": building['polygon']['center'][1], "tags": building["polygon"]["tags"], "points": building["points"], "area": area}

            elif check_building_null(building):
                points = building["points"]
                area = polygon_faceted_area(points, ellipsoid)
                buildings_untagged[b] = {"type": "polygon", "id": b, "lat": building['polygon']['center'][0], "lon": building['polygon']['center'][1], "tags": building["polygon"]["tags"], "points": building["points"], "area": area}

        except ValueError:
            continue

    save(buildings_untagged, city + '/Data/buildings_untagged.json')

    b_u_context = {}
    for i, b in enumerate(list(buildings_untagged.keys())):

        b_u = buildings_untagged[b]
        ctx = []
        for node_id in nodes.keys():
            node = nodes[node_id]
            try:
                d = int(compute_dist(b_u['lat'], b_u['lon'], node['lat'], node['lon']))
            except ValueError:
                continue

            if d < config.context_b:
                ctx.append(node_id)

        b_u_context[b] = ctx

        s = 'Processing buildings data for ' + str(city) + ' (' + str(round((i + 1) / len(buildings_untagged) * 100, 2)) + '%)'
        in_place_print(s)

    print(config.flush)
    save(b_u_context, city + '/Data/buildings_context.json')

    rel_of_int = int_rel(rel, poly_dictionary)

    # for n in nodes:
        # node = nodes[n]
        # nodes[n]['geohash'] = pgh.encode(node['lat'], node['lon'], precision=9)

    save(rel_of_int, city + '/Data/relations.json')
    save(nodes, city + '/Data/entities.json')

    data = {}

    for i, polyline in enumerate(polylines):

        if check_polyline(polyline):

            context = []

            for node_id in nodes.keys():

                node = nodes[node_id]

                for rp in polyline['points']:

                    try:
                        d = int(compute_dist(rp[0], rp[1], node['lat'], node['lon']))
                        # d = int(fast_haversine(rp[0], rp[1], node['lat'], node['lon']))
                        # d = int(haversine((rp[0], rp[1]), (node['lat'], node['lon']), unit='m'))

                        if d <= config.context_d:
                            context.append(node_id)
                            break

                        if d >= config.context_out:
                            break

                    except ValueError:
                        continue

            # n_context = random.sample(nodes.keys(), len(p_context))
            data[polyline['polyline']['id']] = {'context': list(set(context)), 'points': polyline['points'], 'tags': polyline['polyline']['tags']}
            # {'pos': p_context, 'neg': n_context}

        s = 'Preparing data for ' + str(city) + ' (' + str(round((i + 1) / len(polylines) * 100, 2)) + '%)'
        in_place_print(s)

    print(config.flush)
    print('Saving...')
    save(data, city + '/Data/data.json')


def save_n_dict(nodes, path):
    s_dict = {}

    for n in nodes:
        s_dict[n['id']] = (n['lat'], n['lon'])

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(s_dict, f, ensure_ascii=False, indent=4)


def check_node(tags, dd, cont_dd, perm_dd):
    if perm_dd in tags:
        return True

    c = 0
    for tag in tags:
        if tag in dd:
            c += 1
        else:
            for c_d in cont_dd:
                if c_d in tag:
                    c += 1
                    break

    if c == len(tags):
        return True

    return False


def string_of_tags(e):

    tags = e['tags']
    rq_tags = {k: tags[k] for k in osm_strings.use_tags if k in tags}
    s_tags = json_string(rq_tags)

    return s_tags


def check_building(building):

    if 'polygon' in building:

        if 'tags' in building['polygon'] and 'center' in building['polygon']:

            if list(set(osm_strings.building_rqd) & set(building['polygon']['tags'].keys())):

                for rqd in osm_strings.building_rqd:
                    if rqd in building['polygon']['tags']:
                        if building['polygon']['tags'][rqd] in osm_strings.building_others:
                            return False

                return True

            if osm_strings.building in building['polygon']['tags'].keys():

                if building['polygon']['tags'][osm_strings.building] not in osm_strings.building_forbid:

                    return True

    return False


def check_building_null(building):

    if 'polygon' in building:

        if 'tags' in building['polygon'] and 'center' in building['polygon']:

            if 'building' in building['polygon']['tags']:

                if building['polygon']['tags']['building'] == 'yes':

                    return True

    return False


def int_rel(relations, poly_dictionary):

    rel_of_int = {}

    for r in relations:

        r_ways = []

        if 'tags' in r and 'members' in r and 'id' in r:

            if 'route' in r['tags']:

                if r['tags']['route'] in osm_strings.int_routes:

                    for m in r['members']:

                        if 'type' in m and 'ref' in m:

                            if m['type'] == 'way':

                                if m['ref'] in poly_dictionary.keys():
                                    r_ways.append(m['ref'])

                    if len(r_ways) >= config.rel_window:
                        rel_of_int[r['id']] = r_ways

    return rel_of_int


def check_polyline(polyline):

    if 'tags' in polyline['polyline'] and 'points' in polyline and 'id' in polyline['polyline']:

        if 'highway' in polyline['polyline']['tags']:

            if polyline['polyline']['tags']['highway'] not in osm_strings.polyline_forbid:

                return True

    return False


def save_n_tagged(nodes, path):
    tagged = {}

    avg_p = False

    for n in nodes:

        if 'tags' in n and 'id' in n and 'lat' in n and 'lon' in n:

            if list(set(osm_strings.reqd) & set(n['tags'].keys())):

                tags = n['tags']
                rq_tags = {k: tags[k] for k in osm_strings.reqd if k in tags}

                if len([neg_v for neg_v in osm_strings.negative_reqd if neg_v in rq_tags.values()]):
                    continue

                if not avg_p:
                    first_x = n['lat']
                    first_y = n['lon']
                    avg_p = True

                if avg_p:

                    try:

                        d = int(compute_dist(n['lat'], n['lon'], first_x, first_y))

                        if d > config.filter_out_d:
                            continue

                    except ValueError:
                        continue

                tagged[n['id']] = n

    print('Saving ' + str(len(tagged)) + ' tagged nodes...')
    save(tagged, path)


def json2geo(elements):

    geo_json = {"type": "FeatureCollection", "features": []}
    ellipsoid = Geod(ellps='WGS84')

    for e in elements:

        element = elements[e]

        if (check_building(element) or check_building_null(element)) and polygon_faceted_area(element['points'], ellipsoid) < 50000:

            geo_element = {"type": "Feature",
                           "geometry": {"type": 'polygon'.capitalize(), "coordinates": [[point[::-1] for point in element['points']]]},
                           "properties": {"id": element['polygon']['id']}}

            geo_json["features"].append(geo_element)

    return geo_json


def map_city(city):

    path = 'utils/name_mappings.json'
    with open(path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    if str(city) in mappings:
        if str(city) != mappings[city]:
            res = input("Do you mean '" + mappings[city] + "'? (y/n): ")
            if res == 'y':
                return mappings[city]
            else:
                return city

    return city


def read_nodes(city, tagged=True):

    if tagged:
        path = city + '/Node/nodes_tagged.json'
        with open(path, 'r') as f:
            nodes = json.load(f)

    else:
        path = city + '/Nodes/nodes_all.json'
        with open(path, 'r') as f:
            nodes = json.load(f)

    return nodes


def dl_driving_network(city):

    print('Saving driving network graph...')
    G = ox.graph_from_place(city, network_type='drive', simplify=True)
    ox.io.save_graphml(G, filepath=str(city) + '/Ways/road_network.graphml', encoding='utf-8')

    G = nx.read_graphml(str(city) + '/Ways/road_network.graphml')

    edge_list = []
    count = 0
    for g in G:

        li = []
        for k in G[g].keys():
            li.append(G[g][k][0]['osmid'])

        for i in range(len(li) - 1):

            for j in range(i + 1, len(li)):

                if '[' in li[i]:
                    a = li[i].replace('[', '').replace(']', '').split(', ')[0]
                else:
                    a = li[i]

                if '[' in li[j]:
                    b = li[j].replace('[', '').replace(']', '').split(', ')[0]
                else:
                    b = li[j]

                edge_list.append([int(a), int(b)])

        for e in li:
            if '[' in e:
                eli = e.replace('[', '').replace(']', '').split(', ')
                for i in range(len(eli) - 1):
                    edge_list.append([int(eli[i]), int(eli[i + 1])])

        count += 1

    inv_g = nx.from_edgelist(edge_list)
    nx.write_edgelist(inv_g, str(city) + '/Ways/road_network.edgelist', data=False)

    with open(str(city) + '/Ways/road_network.edgelist') as f:
        in_edge_list = [list(map(int, line.strip().split())) for line in f]

    with open(str(city) + '/Ways/edge_list', "wb") as fp:
        pickle.dump(in_edge_list, fp)


def ways_split(ways):

    polygons = []
    polylines = []

    for w in ways:
        if w['nodes'][0] == w['nodes'][-1]:

            if 'tags' in w:
                if 'highway' in w['tags']:
                    continue

            polygons.append(w)

        else:
            polylines.append(w)

    return polygons, polylines


def way_to_polyline(city, polylines):

    all_nodes = read_nodes(city, False)
    way_polylines = []

    f_id = list(all_nodes.keys())[0]

    first_x = all_nodes[f_id][0]
    first_y = all_nodes[f_id][1]

    for i, poly in enumerate(polylines):
        points = []
        poly_flag = True
        for n in poly['nodes']:

            if str(n) in all_nodes:

                try:
                    d = int(compute_dist(all_nodes[str(n)][0], all_nodes[str(n)][1], first_x, first_y))
                except ValueError:
                    break

                if d > config.filter_out_d:
                    poly_flag = False
                    break

                points.append(all_nodes[str(n)])

        if len(points) > 1 and poly_flag:
            d = {"polyline": poly, "points": points}
            way_polylines.append(d)

        s = 'Converting ways to polylines... (' + str(round((i + 1) / len(polylines) * 100, 2)) + '%)'
        in_place_print(s)

    print(config.flush)

    return way_polylines


def polyline_id_mapping(city, polylines):

    poly_dictionary = {}
    p_id = 0

    for polyline in polylines:

        if check_polyline(polyline):

            if polyline['polyline']['id'] not in poly_dictionary:
                poly_dictionary[polyline['polyline']['id']] = p_id
                p_id += 1

    save(poly_dictionary, city + '/Data/polyline_to_id.json')

    return poly_dictionary


def way_to_polygon(city, polygons):

    all_nodes = read_nodes(city, False)
    way_polygons = {}

    f_id = list(all_nodes.keys())[0]

    first_x = all_nodes[f_id][0]
    first_y = all_nodes[f_id][1]

    for i, poly in enumerate(polygons):

        if 'id' in poly:

            points = []
            poly_flag = True
            for n in poly['nodes']:

                if str(n) in all_nodes:

                    try:
                        d = int(compute_dist(all_nodes[str(n)][0], all_nodes[str(n)][1], first_x, first_y))
                    except ValueError:
                        break

                    if d > config.filter_out_d:
                        poly_flag = False
                        break

                    points.append(all_nodes[str(n)])

            if len(points) > 2 and poly_flag:
                p_poly = Polygon(points)
                poly['center'] = [p_poly.centroid.x, p_poly.centroid.y]
                d = {"polygon": poly, "points": points}
                way_polygons[poly['id']] = d

            s = 'Converting ways to polygons... (' + str(round((i + 1) / len(polygons) * 100, 2)) + '%)'
            in_place_print(s)

    print(config.flush)

    return way_polygons


def polygon_to_raster(way_polygons, city):

    with open(str(city) + '/Ways/rasters.npy', 'wb') as f:

        for j, wp in enumerate(way_polygons):
            points = []
            p_lat = []
            p_lon = []

            for i in range(len(wp["points"])):
                p_lat.append(wp["points"][i][0])
                p_lon.append(wp["points"][i][1])

            for i in range(len(p_lat)):
                norm_lat = int((p_lat[i] - min(p_lat)) / (max(p_lat) - min(p_lat)) * (
                            config.img_h - 2 * (config.img_h // 10))) + config.img_h // 10
                norm_lon = int((1 - (p_lon[i] - min(p_lon)) / (max(p_lon) - min(p_lon))) * (
                            config.img_w - 2 * (config.img_w // 10))) + config.img_w // 10
                points.append([norm_lat, norm_lon])

            poly = geometry.Polygon([p[0], p[1]] for p in points)
            img = [[0]] # rasterio.features.rasterize([poly], out_shape=(config.img_h, config.img_w))
            img = np.array(img)

            s = 'Converting polygons to rasters... (' + str(round((j + 1) / len(way_polygons) * 100, 2)) + '%)'
            in_place_print(s)

            np.save(f, img)

    print(config.flush)

    return


def show_json_frq(dic):

    frq_dic = {}

    for key in dic.keys():

        if dic[key] not in frq_dic:
            frq_dic[dic[key]] = 1

        else:
            frq_dic[dic[key]] += 1

    sorted_frq_dic = {k: v for k, v in sorted(frq_dic.items(), key=lambda item: -item[1])}

    for key in sorted_frq_dic:
        print(key, sorted_frq_dic[key])

    exit(0)


def hp_to_json(cd):

    hp_df = pd.read_csv(str(cd.city) + '/Data/housing_data.csv')
    hp_df = hp_df.loc[hp_df['TransactionType'] == 'Sale']
    hp_df = hp_df[['ID', 'X', 'Y', 'avgPriceSqm']]

    hp_df = hp_df.dropna(subset=['avgPriceSqm'])
    # 356290.5527 148893.4126

    in_proj = Proj('epsg:32648')
    out_proj = Proj('epsg:4326')
    areas = {}
    area_id = 0

    for i in range(hp_df.shape[0]):

        x, y = hp_df.iloc[i]['X'], hp_df.iloc[i]['Y']
        area_price = hp_df.iloc[i]['avgPriceSqm']

        p_c = transform(in_proj, out_proj, x, y)

        p1 = transform(in_proj, out_proj, x - 500, y - 500)
        p2 = transform(in_proj, out_proj, x + 500, y - 500)
        p3 = transform(in_proj, out_proj, x + 500, y + 500)
        p4 = transform(in_proj, out_proj, x - 500, y + 500)

        area_info = {'center': p_c, 'polygons': [], 'nodes': [], 'price': area_price}

        square = Polygon([p1, p2, p3, p4, p1])
        count = 0
        for e in cd.entities:
            entity = cd.entities[e]
            entity_p = Point(entity['lat'], entity['lon'])
            entity_type = entity['type']

            if square.contains(entity_p):
                count += 1

                if entity_type == 'node':
                    area_info['nodes'].append(e)
                else:
                    area_info['polygons'].append(e)

        if count >= 20:
            areas[area_id] = area_info
            area_id += 1

    save(areas, cd.city + '/Data/housing_price.json')


def reset_log_file():
    open(config.log_file_name, 'w').close()


def land_use_to_json(city):

    lu_df = pd.read_csv(str(city) + '/Data/land_use_data.csv')

    lu_groups = lu_df.groupby('id')

    lu_json = {}

    for group in lu_groups:

        group_id = str(group[0])
        group_max_lu = lu_df.iloc[group[1]['PERCENTAGE'].idxmax()]['LU_DESC']

        if group_max_lu not in lu_strings.land_use_dismiss:

            if group_max_lu in lu_strings.land_use_rename:
                group_max_lu = lu_strings.land_use_rename[group_max_lu]

            lu_json[group_id] = group_max_lu

    # show_json_frq(lu_json)
    save(lu_json, city + '/Data/land_use.json')


def json_string(d):

    return str(d).replace("'", "").replace('"', '').replace('{', '').replace('}', '').replace('_', ' ')


def read_rasters():

    with open('rasters.npy', 'rb') as f:
        while True:
            try:
                r = np.load(f)
                return r

            except ValueError:
                return


def show_polygons(cd):

    keys = list(cd.entities.keys())
    random.shuffle(keys)

    for e in keys:

        entity = cd.entities[e]

        if entity['type'] == 'polygon':
            pass
            '''
            f, ax_arr = plt.subplots(1, 2)
            poly = Polygon(entity['points'])
            tags = json_string({k: entity['tags'][k] for k in osm_strings.use_tags if k in entity['tags']})
            image = poly_to_raster(poly, rotation=False, expand_first=False)
            image_r = poly_to_raster(poly, expand_first=False)
            ax_arr[0].imshow(image*255)
            ax_arr[1].imshow(image_r*255)
            plt.show()
            '''


def polyline_nodes(polyline, train_e, edges):

    poly_id = int(polyline['polyline']['id'])

    n = []
    if str(poly_id) in edges:
        n = edges[str(poly_id)]

    n_s = []
    if n:
        for ne in n:
            if str(ne) in train_e:
                n_s.append(train_e[str(ne)])

    if not n_s:
        n_s.append(0)

    if not n:
        n.append(0)

    n_s = np.mean(n_s)

    points = polyline['points']
    length = 0

    for i in range(len(points) - 1):
        length += int(compute_dist(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]))

    ln = int(length / len(points))
    le = int(length / len(n))

    a = np.ones(config.polyline_size // 4) * ln
    b = np.ones(config.polyline_size // 4) * le
    c = np.ones(config.polyline_size // 2) * n_s

    return np.concatenate((a, b, c))


def polyline_features(polyline):

    features = []

    if 'tags' in polyline['polyline']:
        tags = polyline['polyline']['tags']

        if 'highway' in tags:
            hw = tags['highway'].split('_')[0]

            if hw not in osm_strings.polyline_hw:
                hw = osm_strings.polyline_hw_default

            hw = osm_strings.polyline_hw[hw]
            features.append(hw)

        else:
            features.append(osm_strings.polyline_hw[osm_strings.polyline_hw_default])

        if 'lanes' in tags:

            try:
                lanes = min(int(tags['lanes']), 5)
                features.append(lanes)

            except ValueError:
                features.append(0)

        else:
            features.append(0)

        if osm_strings.polyline_ms in tags:

            ms = tags[osm_strings.polyline_ms]

            try:
                ms = min(int(ms.split(' ')[0])//5, 10)

            except ValueError:
                ms = 0

            features.append(ms)

        else:
            features.append(0)

    else:
        features.append(osm_strings.polyline_hw[osm_strings.polyline_hw_default])
        features.append(0)
        features.append(0)

    return features


def filter_polygons(cd):

    return [k for k in cd.entities.keys() if cd.entities[k]["type"] == osm_strings.polygon and len(cd.entities[k]['points']) > 3]


def largest(cd):

    polygons = filter_polygons(cd)
    return max([cd.entities[k]['area'] for k in polygons])


def norm_surface(s):

    return min(s, config.max_surface)/config.max_surface


def log_surface(s):

    return math.log((1 + math.pow(math.sqrt(1000), 3))/(1 + math.pow(math.sqrt(s), 3)))


def generate_geojson(city):

    with open(str(city) + '/Ways/polygons.json', encoding='utf-8') as f:
        b_u = json.load(f)

    geo_b_u = json2geo(b_u)
    save(geo_b_u, 'geo_buildings_' + str(city).lower() + '.json')


def read_model_im(city):

    if os.path.exists(city + "/Model/model_im.pkl"):
        with open(city + "/Model/model_im.pkl", 'rb') as f:
            model = pickle.load(f)
            return model

    else:
        no_model_im(city)


def read_model_r(city):
    if os.path.exists(city + "/Model/model_rast.pkl"):
        with open(city + "/Model/model_rast.pkl", 'rb') as f:
            model = pickle.load(f)
            return model

    else:
        no_model_im(city)


def read_model_rel(city):
    if os.path.exists(city + "/Model/model_rel.pkl"):
        with open(city + "/Model/model_rel.pkl", 'rb') as f:
            model = pickle.load(f)
            return model

    else:
        no_model_im(city)


def read_model(city):
    if os.path.exists(city + "/Model/model.pkl"):
        with open(city + "/Model/model.pkl", 'rb') as f:
            model = pickle.load(f)
            return model

    else:
        no_model_im(city)


def mape(y_t, p):
    y_t, p = np.array(y_t), np.array(p)
    mape_score = np.mean(np.abs((y_t - p) / (y_t + 1e-6)))*100
    return mape_score


def key_list(voc):
    return list(voc.keys())*config.pd_limu


def read_best_model_im(city):

    models = os.listdir(city + "/Model")
    best = 0
    model_name = ''

    for m in models:

        if 'model_im' in m:

            try:
                if int(re.findall('\d+', m)[0]) > best:
                    best = int(re.findall('\d+', m)[0])
                    model_name = m

            except ValueError:
                continue

    if not model_name:
        no_model_im(city)

    with open(city + '/Model/' + model_name, 'rb') as f:
        model = pickle.load(f)

    return model


def read_best_model_r(city):

    models = os.listdir(city + "/Model")
    best = 0
    model_name = ''

    for m in models:

        if 'model_r' in m:

            try:
                if int(re.findall('\d+', m)[0]) > best:
                    best = int(re.findall('\d+', m)[0])
                    model_name = m

            except ValueError:
                continue

    if not model_name:
        no_model_im(city)

    with open(city + '/Model/' + model_name, 'rb') as f:
        model = pickle.load(f)

    return model


def poly_to_raster(polygon, rotation=True, expand_first=True):

    if rotation and isinstance(polygon, Polygon):
        rectangle = polygon.minimum_rotated_rectangle

        xc = polygon.centroid.x
        yc = polygon.centroid.y

        rec_x = []
        rec_y = []

        try:
            _ = rectangle.exterior.coords
        except (AttributeError, TypeError):
            return []

        for point in rectangle.exterior.coords:
            rec_x.append(point[0])
            rec_y.append(point[1])

        top = np.argmax(rec_y)
        top_left = top - 1 if top > 0 else 3
        top_right = top + 1 if top < 3 else 0

        x0, y0 = rec_x[top], rec_y[top]
        x1, y1 = rec_x[top_left], rec_y[top_left]
        x2, y2 = rec_x[top_right], rec_y[top_right]

        d1 = np.linalg.norm([x0 - x1, y0 - y1])
        d2 = np.linalg.norm([x0 - x2, y0 - y2])

        if d1 > d2:
            cos_p = (x1 - x0)/d1
            sin_p = (y0 - y1)/d1
        else:
            cos_p = (x2 - x0)/d2
            sin_p = (y0 - y2)/d2

        matrix = (cos_p, -sin_p, 0.0, sin_p, cos_p, 0.0, 0.0, 0.0, 1.0, xc - xc*cos_p + yc*sin_p, yc - xc*sin_p - yc*cos_p, 0.0)
        polygon = affinity.affine_transform(polygon, matrix)

    min_x, min_y, max_x, max_y = polygon.bounds
    length_x = max_x - min_x
    length_y = max_y - min_y

    if length_x > length_y:
        min_y -= (length_x - length_y)/2
        max_y += (length_x - length_y)/2

    else:
        min_x -= (length_y - length_x)/2
        max_x += (length_y - length_x)/2

    length = max(length_x, length_y)

    min_x -= length*0.1
    min_y -= length*0.1
    max_x += length*0.1
    max_y += length*0.1

    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, config.img_w, config.img_h)
    image = rasterio.features.rasterize([polygon], out_shape=(config.img_w, config.img_h), transform=transform)

    # return image
    if expand_first:
        return np.repeat(np.expand_dims(image, axis=0), 3, 0)
    else:
        return np.repeat(np.expand_dims(image, axis=-1), 3, -1)


def polygon_faceted_area(points, ellipsoid):

    area = 0

    for i in range(len(points) - 1):

        lon1, lat1 = points[i]
        lon2, lat2 = points[i + 1]

        area += ellipsoid.geometry_area_perimeter(Polygon([(lat1, lon1), (lat2, lon2), (0, 0)]))[0]

    return int(abs(area))


def compute_dist(lat1, lon1, lat2, lon2):

    R = 6373.0

    try:
        lat1 = radians(float(lat1))
    except ValueError:
        return ' '

    try:
        lon1 = radians(float(lon1))
    except ValueError:
        return ' '

    try:
        lat2 = radians(float(lat2))
    except ValueError:
        return ' '

    try:
        lon2 = radians(float(lon2))
    except ValueError:
        return ' '

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return str(round(R * c * 1000))


def fast_haversine(lat1, lon1, lat2, lon2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6373 * c * 1000

    return m


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def no_model_im(city):
    print('No trained model found for ' + str(city))
    exit()


def save(elements, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(elements, f, ensure_ascii=False, indent=4)


def print_epoch(e, max_e):
    print(config.sep_width*"-")
    print("Epoch (" + str(e) + '/' + str(max_e) + ')')
    print(config.sep_width*"-")


def in_place_print(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()


def release(in_list):
    del in_list[:]
    del in_list
