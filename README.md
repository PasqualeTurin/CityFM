# CityFM

### This is the code for 'CityFM' paper (CIKM 2024)

&rarr; Download our pre-trained models here: <a href="https://drive.google.com/file/d/1KauU3-sa-xdQPn2Lb_HyG6B5D5milNjo/view?usp=drive_link" target="_blank">Singapore</a>, <a href="https://drive.google.com/file/d/1fsdl-DI9XyFX4thK2NUbjWwXIr0xUmvy/view?usp=drive_link" target="_blank">New York</a>, <a href="https://drive.google.com/file/d/1lYXPTwAFBOkPCIdona4knostPB9gFlMI/view?usp=drive_link" target="_blank">Seattle</a>

&rarr; Place the model in the Model/ folder of the city

&rarr; Install required packages
```
pip install -r requirements.txt
```

### Downstream tasks

```
python CityFM_downstream.py -c "city" -t "task"
```

Meaning of the flags and possible values:
* ``-c`` (city): Specify the city you wish to use. Possible values are ``Singapore``, ``"New York"``, ``Seattle``.
* ``-t`` (task): Specify the downstream task. Possible values are ``avg_speed`` (NYC and Seattle), ``build_func`` (Singapore), ``pop_density`` (Singapore and NYC).

Please note that the first time a task in a city is run, it may require some time to pre-compute the embeddings (using the pre-trained model). After that, the training will always be very fast.

### Pre-processing

Pre-processing is not required for NYC, Singapore and Seattle, as we shared the pre-processed data. If you wish to do it for a new city, e.g., Toronto, please use:

```
python CityFM_preprocess.py -c "Toronto"
```

This process will query OSM Overpass API, download and pre-process the data for the requested city. Please be patient, the pre-processing may take several hours, the progress percentage will be shown on the terminal. If you wish to (re-)download the data for an existing city, please delete the city folder before running the command above.


### Pre-training

Pre-training is not necessary for NYC, Singapore and Seattle. If you wish to pre-train your own model, use:

```
python CityFM_train.py -c "city"
```

The pre-processing step is required in order to pre-train a model in a new city.

<br><br>
## Use a pre-trained model to encode your own spatial entities

If you want to use a pre-trained model to encode entities (nodes, polygons, polylines) for your own task, simply use:

```
from encoder import encode_nodes, encode_polygons, encode_polylines
```
#### Encode nodes in the list ```my_nodes```, using the pre-trained model in Singapore
```
my_nodes_pos, my_nodes_text = encode_nodes('Singapore', my_nodes)
```
The outputs are the positional and textual embeddings of the nodes, with the following shapes:

&rarr; The shape of my_nodes_pos is ```[len(my_nodes), 256]```

&rarr; The shape of my_nodes_text is ```[len(my_nodes), 768]```

#### Encode polygons in the list ```my_polygons```, using the pre-trained model in Singapore
```
my_polygons_pos, my_polygons_text, my_polygons_vis = encode_polygons('Singapore', my_polygons)
```
The outputs are the positional, textual and visual embeddings of the polygons, with the following shapes:

&rarr; The shape of my_polygons_pos is ```[len(my_polygons), 256]```

&rarr; The shape of my_polygons_text is ```[len(my_polygons), 768]```

&rarr; The shape of my_polygons_vis is ```[len(my_polygons), 768]```


#### Encode polylines in the list ```my_polylines```
```
my_polylines_pos, my_polylines_text = encode_polylines(my_polylines)
```
The outputs are the positional and textual (i.e. polyline tags) embeddings of the polylines, with the following shapes:

&rarr; The shape of my_polylines_pos is ```[len(my_polylines), 256]```

&rarr; The shape of my_polylines_text is ```[len(my_polylines), 192]```


#### Examples of nodes, polygons and polylines formats:

#### Node
```
my_node1 = {
        "type": "node",
        "id": 369128165,
        "lat": 1.3827466,
        "lon": 103.8932886,
        "tags": {
            "brand": "7-Eleven",
            "name": "7-Eleven",
            "operator": "7-11",
            "shop": "convenience",
        },
    }

my_nodes = [my_node1, ...]
```

#### Polygon
```
my_polygon1 = {
        "type": "polygon",
        "id": "153134133",
        "lat": 1.3055275000000002,
        "lon": 103.90792454999998,
        "tags": {
            "addr:housenumber": "32",
            "addr:postcode": "429538",
            "addr:street": "Chapel Road",
            "building": "house",
            "building:levels": "3",
            "roof:shape": "gabled"
        },
        "points": [
            [
                1.3054958,
                103.907833
            ],
            [
                1.3056007,
                103.9079879
            ],
            [
                1.3055592,
                103.9080161
            ],
            [
                1.3054543,
                103.9078612
            ],
            [
                1.3054958,
                103.907833
            ]
        ],

        "area": 115
    }

my_polygons = [my_polygon1, ...]
```

#### Polyline
```
my_polyline1 = {
        "polyline": {
            "type": "way",
            "id": 4634847,
            "nodes": [
                29449867,
                29449866,
                29449865,
                4245082364,
                29449864,
                720880672
            ],
            "tags": {
                "bicycle": "no",
                "highway": "motorway_link",
                "lanes": "3",
                "oneway": "yes",
                "toll": "no",
                "turn:lanes": "left|left|through;right"
            }
        },
        "points": [
            [
                47.6434001,
                -122.3049446
            ],
            [
                47.643476,
                -122.3048108
            ],
            [
                47.643598,
                -122.3046427
            ],
            [
                47.6436406,
                -122.3045933
            ],
            [
                47.6437405,
                -122.3044753
            ],
            [
                47.643776,
                -122.3044187
            ]
        ],
        "length": 58
    }

my_polylines = [my_polyline1, ...]
```

#### Compute the area of a polygon

&rarr; If you want to compute the area of a polygon, please use the function ```polygon_faceted_area```:

```
from pyproj import Geod
from utils.pre_processing import polygon_faceted_area

ellipsoid = Geod(ellps='WGS84')
area = polygon_faceted_area(points, ellipsoid)

```

