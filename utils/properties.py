import torch
import config
from utils.training import *
from sklearn.decomposition import PCA
import random
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt


def raster_cs(cd, model, polygon_id, plot_polygon=False):
    polygons = filter_polygons(cd)
    tok = Tokenizer(config.lm)
    model.eval()
    random.shuffle(polygons)
    shown_tags = []

    polygon = cd.entities[str(polygon_id)]
    image = torch.tensor(poly_to_raster(Polygon(polygon['points']))).unsqueeze(0).float()
    tags = torch.tensor(serialize_data([polygon], tok))
    tags_m = torch.tensor(np.where(tags != 0, 1, 0))
    s = torch.tensor(norm_surface(polygon['area'])).unsqueeze(-1)

    poly_emb = model.projection_p(torch.mean(model.lm(tags, attention_mask=tags_m)[0][:, :, :].squeeze(), 0))

    image_emb = model.projection_img(model.cv(image).squeeze())
    s_emb = model.projection_s(s)
    tot_emb = (image_emb + s_emb) / 2

    print(string_of_tags(polygon))
    print("Area:", str(polygon['area']))
    print("Self Similarity (shape):", model.cos(image_emb, poly_emb).item())
    print("Self Similarity (size):", model.cos(s_emb, poly_emb).item())
    print("Self Similarity (all):", model.cos(tot_emb, poly_emb).item())
    print("--------------------------------------------")

    for p in polygons:

        k_polygon = cd.entities[p]
        tags = torch.tensor(serialize_data([k_polygon], tok))
        tags_m = torch.tensor(np.where(tags != 0, 1, 0))
        poly_emb = model.projection_p(torch.mean(model.lm(tags, attention_mask=tags_m)[0][:, :, :].squeeze(), 0))
        sot = string_of_tags(k_polygon)

        if sot not in shown_tags:
            print(sot, "- Shape:", model.cos(image_emb, poly_emb).item(), "- Size:", model.cos(s_emb, poly_emb).item(),
                  "- All:", model.cos(tot_emb, poly_emb).item())
            shown_tags.append(sot)

        if len(shown_tags) == 40:
            break

    if plot_polygon:
        img = poly_to_raster(Polygon(polygon['points']), expand_first=False)
        plt.imshow(img * 255)
        plt.show()


def plot_poi_pca(cd, model, n):
    tok = Tokenizer(config.lm)
    model.eval()
    keys = list(cd.entities.keys())
    random.shuffle(keys)

    embs = []

    i = 0
    count = 0
    used_tags = []

    while count < n:

        with torch.no_grad():

            entity = cd.entities[keys[i]]
            tags = entity['tags']
            label = ''

            if 'building' in tags:
                label = tags['building']
                if label == 'yes':
                    label = 'building'
            elif 'amenity' in tags:
                label = tags['amenity']
                if label == 'yes':
                    label = 'building'
            elif 'shop' in tags:
                label = tags['shop']
                if label == 'yes':
                    label = 'shop'

            elif 'tourism' in tags:
                label = tags['tourism']
                if label not in ['museum', 'hotel', 'attraction']:
                    label = 'tourism'

            if label and label not in used_tags:
                used_tags.append(label)
                count += 1
                poi = torch.tensor(serialize_data([entity], tok))
                poi_m = torch.tensor(np.where(poi != 0, 1, 0))
                embs.append(
                    model.projection_p(torch.mean(model.lm(poi, attention_mask=poi_m)[0][:, :, :].squeeze(), 0)))

        i += 1

    embs = torch.stack(embs)
    pca = PCA(n_components=2)
    components = np.array(pca.fit_transform(embs))

    fig, ax = plt.subplots()
    ax.scatter(components[:, 0], components[:, 1])

    for i, txt in enumerate(used_tags):
        ax.annotate(txt, (components[i, 0], components[i, 1]))

    plt.show()

    exit()


def plot_relations(cd):

    relations = cd.relations
    polylines = cd.polylines
    count = 0

    print(len(relations), "Relations found")

    # print(len(relations))  # 684sg, 808nyc
    figure(figsize=(24, 11), dpi=240)

    way_voc = {}

    for key in relations:

        count += 1
        r = relations[key]

        for way in r:

            w = str(way)

            if w not in way_voc:
                way_voc[w] = 1
            else:
                way_voc[w] += 1

    max_value = max([way_voc[k] for k in way_voc])
    count = 0
    plotted = []

    for key in relations:

        count += 1
        r = relations[key]

        for way in r:

            w = str(way)

            if w not in polylines or w in plotted:
                continue

            plotted.append(w)

            points_y = [p[0] for p in polylines[w]['points']]
            points_x = [p[1] for p in polylines[w]['points']]

            color = 'k'
            # if way_voc[w] == 2:
            #    color = 'y'
            alpha = way_voc[w] / max_value

            linew = 1
            if alpha > 0.4:
                linew = 2
            if alpha > 0.6:
                linew = 3
            if alpha > 0.8:
                linew = 4
            if alpha > 0.9:
                linew = 5

            alpha *= 1.1
            alpha = min([1, alpha])

            # default
            linew = 1
            alpha = 1

            # plt.plot(points_x, points_y, linestyle='-', color=color)
            plt.plot(points_x, points_y, linestyle='-', color=color, alpha=alpha, linewidth=linew)

    # print(way_voc)
    plt.savefig('bus_loops.pdf')
    plt.show()
    exit()


def topic_search(cd, model, topic):
    model.eval()

    tok = Tokenizer(config.lm)
    topic_t = torch.tensor(
        tok.tokenizer.convert_tokens_to_ids(tok.tokenizer.tokenize('[CLS] ' + topic + ' [SEP]'))).unsqueeze(0)
    topic_m = torch.tensor(np.where(topic_t != 0, 1, 0))
    emb = model.projection_p(torch.mean(model.lm(topic_t, attention_mask=topic_m)[0][:, :, :].squeeze(), 0))

    keys = list(cd.entities.keys())
    random.shuffle(keys)

    for e2 in keys:
        entity = cd.entities[e2]
        poi = torch.tensor(serialize_data([entity], tok))
        poi_m = torch.tensor(np.where(poi != 0, 1, 0))
        emb2 = model.projection_p(torch.mean(model.lm(poi, attention_mask=poi_m)[0][:, :, :].squeeze(), 0))

        print(str(topic), e2, "Similarity:", model.cos(emb, emb2).item())


def poi_cs(cd, model, poi_id):
    tok = Tokenizer(config.lm)
    model.eval()
    keys = list(cd.entities.keys())
    random.shuffle(keys)

    entity = cd.entities[str(poi_id)]
    poi = torch.tensor(serialize_data([entity], tok))
    poi_m = torch.tensor(np.where(poi != 0, 1, 0))
    emb = model.projection_p(torch.mean(model.lm(poi, attention_mask=poi_m)[0][:, :, :].squeeze(), 0))

    similarity_v = {}

    for i, e2 in enumerate(keys):
        entity = cd.entities[e2]
        poi = torch.tensor(serialize_data([entity], tok))
        poi_m = torch.tensor(np.where(poi != 0, 1, 0))
        emb2 = model.projection_p(torch.mean(model.lm(poi, attention_mask=poi_m)[0][:, :, :].squeeze(), 0))

        # print(str(poi_id), e2, "Similarity:", model.cos(emb, emb2).item())
        similarity_v[e2] = model.cos(emb, emb2).item()
        if i == 10000:
            break

    similarity_v = {k: v for k, v in sorted(similarity_v.items(), key=lambda item: -item[1])}
    for key in similarity_v:
        print(key, similarity_v[key])


def road_cs(cd, model, road_id):
    tok = Tokenizer(config.lm)
    model.eval()

    for road in cd.data.keys():
        if int(road) != int(road_id):
            continue

        tags = []
        for entity in cd.data[str(road)]['context']:
            tags.append(
                {k: cd.entities[entity]['tags'][k] for k in osm_strings.use_tags if k in cd.entities[entity]['tags']})
        pois = torch.tensor(serialize_data([cd.entities[k] for k in cd.data[road]['context']], tok))
        pois_m = torch.tensor(np.where(pois != 0, 1, 0))
        emb = model.projection_p(torch.mean(torch.mean(model.lm(pois, attention_mask=pois_m)[0][:, :, :], 0), 0))

        for road_c in cd.data.keys():

            if not cd.data[str(road_c)]['context']:
                continue

            pois = torch.tensor(serialize_data([cd.entities[k] for k in cd.data[road_c]['context']], tok))
            pois_m = torch.tensor(np.where(pois != 0, 1, 0))
            emb_c = model.projection_p(torch.mean(torch.mean(model.lm(pois, attention_mask=pois_m)[0][:, :, :], 0), 0))

            tags_c = []
            for entity in cd.data[str(road_c)]['context']:
                tags_c.append(
                    {k: cd.entities[entity]['tags'][k] for k in osm_strings.use_tags if
                     k in cd.entities[entity]['tags']})

            if model.cos(emb, emb_c).item() > 0.6 or model.cos(emb, emb_c).item() < -0.1:
                print(tags)
                print(tags_c)
                print(road, road_c, "Similarity:", model.cos(emb, emb_c).item())
                time.sleep(0.1)
