import random

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import r2_score
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import config
from utils.pre_processing import *
from utils.training import *


def precompute_embs_pop_d(cd, model, pop_d):

    district_emb = {}
    key = 0
    tok = Tokenizer(config.lm)
    count = 0

    for d in pop_d:

        count += 1

        if random.choice([True, False]):
            continue

        district = pop_d[d]

        d_polygon = Polygon([[p[1], p[0]] for p in district['polygon']])
        semantic_nodes = []
        polygons = []

        pe = PE((d_polygon.centroid.x, d_polygon.centroid.y), config.pe_size)

        for e in cd.entities:

            entity = cd.entities[e]
            entity_p = Point(entity['lat'], entity['lon'])

            if d_polygon.contains(entity_p):

                if entity['type'] == 'node':
                    semantic_nodes.append(e)

                elif entity['type'] == 'polygon':
                    semantic_nodes.append(e)
                    polygons.append(e)

        if len(semantic_nodes) == 0:
            continue

        with torch.no_grad():

            pois = torch.tensor(serialize_data([cd.entities[k] for k in semantic_nodes], tok, add_zero=True))
            pois_m = torch.tensor(np.where(pois != 0, 1, 0))
            sem_emb = model.projection_p(
                torch.mean(torch.mean(model.lm(pois, attention_mask=pois_m)[0][:, :, :], 0), 0))

            poly_embs = []
            for p in polygons:
                entity_p = cd.entities[p]
                raster = poly_to_raster(Polygon(entity_p['points']))
                if not len(raster):
                    continue
                image = torch.tensor(raster).unsqueeze(0).float()
                image_emb = model.projection_img(model.cv(image).squeeze())
                s = torch.tensor(norm_surface(entity_p['area'])).unsqueeze(-1)
                s_emb = model.projection_s(s)

                poly_embs.append((image_emb + s_emb) / 2)

            if len(poly_embs) > 0:
                poly_embs = torch.mean(torch.stack(poly_embs), 0)

            else:
                poly_embs = torch.zeros(model.emb_size)

        # thousands of people per square kilometer

        district_emb[key] = {'pos': list(pe), 'nodes': sem_emb.tolist(), 'polygons': poly_embs.tolist(),
                             'pop_density': district['population'],
                             'nodes_id': semantic_nodes, 'poly_ids': polygons}

        key += 1

        s = 'Precomputing districts embeddings... (' + str(round(count / len(list(pop_d)) * 100, 2)) + '%)'
        in_place_print(s)

    print(config.flush)
    save(district_emb, cd.city + '/Data/pop_d_embeddings.json')


def population_density(city, cd, model, training=True):

    with open(str(city) + '/Data/pop_density.json', encoding='utf-8') as file:
        pop_d = json.load(file)

    if not os.path.isfile(city + '/Data/pop_d_embeddings.json'):
        precompute_embs_pop_d(cd, model, pop_d)

    with open(str(city) + '/Data/pop_d_embeddings.json', encoding='utf-8') as file:
        pop_d_embs = json.load(file)

    keys = list(pop_d_embs.keys())

    n_runs = 10

    rmse, l1, r2, mape_ = [], [], [], []

    for i in range(n_runs):
        random.shuffle(keys)
        train = keys[:len(keys) // 10 * 6]
        valid = keys[len(keys) // 10 * 6:len(keys) // 10 * 8]
        test = keys[len(keys) // 10 * 8:]

        max_p = max([pop_d_embs[h]['pop_density'] for h in train + valid + test])
        # max_p = 1.0

        ds_model = HPRegressor(model.emb_size)
        train_pd_prediction(city, pop_d_embs, ds_model, None, max_p, train, valid, training, 'pd')

        with open(str(city) + '/Model/model_hp.pkl', 'rb') as file:
            ds_model = pickle.load(file)

        this_rmse, this_l1, this_r2 = validate_pd_prediction(pop_d_embs, ds_model, None, config.bs, max_p, test, 0,
                                                             True, 'pd')
        rmse.append(this_rmse)
        l1.append(this_l1)
        r2.append(this_r2)

    print("\nFinal results out of " + str(n_runs) + ' runs:\n')
    print('RMSE: ' + str(round(float(np.mean(np.array(rmse))), 2)) + ' (' + str(
        round(float(np.std(np.array(rmse))), 2)) + ') | MAE: ' + str(
        round(float(np.mean(np.array(l1))), 2)) + ' (' + str(round(float(np.std(np.array(l1))), 2)) + ') | R2: ' + str(
        round(float(np.mean(np.array(r2))), 2)) + ' (' + str(
        round(float(np.std(np.array(r2))), 2)) + ')')


def make_categories(city, keys, hd):
    cd = CityData(city, buildings_u=False)
    cats = {}
    c = 0

    hd_embs = {}

    for k in keys:
        # for e in hd[str(int(k)//2)]['nodes']:
        for e in hd[k]['nodes_id']:
            tags = cd.entities[e]['tags']
            if 'amenity' in tags:
                if tags['amenity'] not in cats:
                    cats[tags['amenity']] = c
                    c += 1

            elif 'shop' in tags:
                if 'shop' not in cats:
                    cats['shop'] = c
                    c += 1

            elif 'tourism' in tags:
                if 'tourism' not in cats:
                    cats['tourism'] = c
                    c += 1

            elif 'office' in tags:
                if tags['office'] not in cats:
                    cats[tags['office']] = c
                    c += 1

            elif 'leisure' in tags:
                if tags['leisure'] not in cats:
                    cats[tags['leisure']] = c
                    c += 1

    for k in keys:
        cat_list = [np.zeros(len(cats))]
        # for e in hd[str(int(k)//2)]['nodes']:
        for e in hd[k]['nodes_id']:
            tags = cd.entities[e]['tags']
            this_cat = np.zeros(len(cats))

            if 'amenity' in tags:
                this_cat[cats[tags['amenity']]] = 1

            elif 'shop' in tags:
                this_cat[cats['shop']] = 1

            elif 'tourism' in tags:
                this_cat[cats['tourism']] = 1

            elif 'office' in tags:
                this_cat[cats[tags['office']]] = 1

            elif 'leisure' in tags:
                this_cat[cats[tags['leisure']]] = 1

            cat_list.append(this_cat)

        cat_list = [sum(sub_list) / len(sub_list) for sub_list in zip(*cat_list)]
        hd_embs[k] = cat_list

    return hd_embs


def train_pd_prediction(city, hd_embs, ds_model, hd, max_p, train, valid, training, task):

    if task == 'hp':
        epochs = config.ds_epochs
        lr = config.ds_lr
    else:
        epochs = config.ds_epochs * 2
        lr = config.ds_lr

    opt = optim.Adam(params=ds_model.parameters(), lr=lr)
    bs = config.ds_bs
    criterion = nn.MSELoss()

    best_rmse = math.inf

    for epoch in range(epochs):

        if not training:
            break

        print_epoch(epoch + 1, epochs)

        opt.zero_grad()

        idx = 0
        step = 0

        while idx < len(train):

            ds_model.train()

            if idx > len(train) - bs:
                batch_l = train[idx:]
                if task == 'hp':
                    batch_y = torch.tensor([hd[str(int(h)//2)]['price']/max_p for h in batch_l])
                else:
                    batch_y = torch.tensor([hd_embs[h]['pop_density'] / max_p for h in batch_l])

            else:
                batch_l = train[idx:idx + bs]
                if task == 'hp':
                    batch_y = torch.tensor([hd[str(int(h)//2)]['price']/max_p for h in batch_l])
                else:
                    batch_y = torch.tensor([hd_embs[h]['pop_density'] / max_p for h in batch_l])

            batch_poly = []
            batch_n = []
            batch_p = []

            for element in batch_l:

                batch_poly.append(torch.tensor(hd_embs[element]['polygons']))
                batch_n.append(torch.tensor(hd_embs[element]['nodes']))
                batch_p.append(torch.tensor([pos/10 for pos in hd_embs[element]['pos']]))

            batch_poly = torch.stack(batch_poly)
            batch_n = torch.stack(batch_n)
            batch_p = torch.stack(batch_p)

            p = ds_model(batch_poly, batch_n, batch_p)

            loss = torch.sqrt(criterion(p, batch_y))
            step = p_step(loss, step)
            loss.backward()
            opt.step()
            opt.zero_grad()

            idx += bs

        rmse_ = validate_pd_prediction(hd_embs, ds_model, hd, bs, max_p, valid, epoch, False, task)

        if rmse_ < best_rmse:
            best_rmse = rmse_
            pickle.dump(ds_model, open(str(city) + '/Model/model_hp.pkl', 'wb'))
            print('[SAVED]')
        print(config.sep_width * "-")


def validate_pd_prediction(hd_embs, ds_model, hd, bs, max_p, valid, epoch, testing, task):

    for _ in range(2):
        print(config.sep_width * "-")

    if not testing:
        print('Validation - Epoch: ' + str(epoch + 1))

    else:
        print('Testing best model...')

    print('(Population Density prediction)')

    print(config.sep_width * "-")

    ds_model.eval()

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    idx = 0

    pred = []

    while idx < len(valid):

        ds_model.train()

        if idx > len(valid) - bs:
            batch_l = valid[idx:]

        else:
            batch_l = valid[idx:idx + bs]

        batch_poly = []
        batch_n = []
        batch_p = []

        for element in batch_l:

            batch_poly.append(torch.tensor(hd_embs[element]['polygons']))
            batch_n.append(torch.tensor(hd_embs[element]['nodes']))
            batch_p.append(torch.tensor([pos/10 for pos in hd_embs[element]['pos']]))

        batch_poly = torch.stack(batch_poly)
        batch_n = torch.stack(batch_n)
        batch_p = torch.stack(batch_p)

        with torch.no_grad():
            p = ds_model(batch_poly, batch_n, batch_p) * max_p

        if not p.shape:
            p = p.unsqueeze(0)

        pred.append(p)

        idx += bs

    pred = torch.cat(pred, 0)

    if task == 'hp':
        y = torch.tensor([hd[str(int(h)//2)]['price'] for h in valid]).float()
    else:
        y = torch.tensor([hd_embs[h]['pop_density'] for h in valid]).float()

    mse_ = mse(pred, y).item()
    rmse_ = sqrt(mse_)
    l1_ = l1(pred, y).item()
    r2 = r2_score(pred, y).item()

    print('RMSE: ' + str(round(rmse_, 2)) + ' | MAE: ' + str(round(l1_, 2)) + ' | R2: ' + str(round(r2, 4)))

    print(config.sep_width * "-")

    if not testing:
        return rmse_
    if testing:
        return rmse_, l1_, r2


def split_data(df):

    train = df.iloc[:int(df.shape[0]/10*6), :]
    valid = df.iloc[int(df.shape[0]/10*6):int(df.shape[0]/10*8), :]
    test = df.iloc[int(df.shape[0]/10*8):, :]

    return train, valid, test


def random_performance(test, n_c, lu_data_u, classes_one_hot):

    pred = []
    y = []

    for i in range(len(test)):
        pred.append(random.choice(list(range(0, n_c))))
        y.append(classes_one_hot[lu_data_u[test[i]]])

    micro = f1_score(pred, y, average='micro')
    macro = f1_score(pred, y, average='macro')
    weighted = f1_score(pred, y, average='weighted')

    print('F1 (micro): ' + str(round(micro * 100, 2)) + '% | F1 (macro): ' + str(
        round(macro * 100, 2)) + '% | F1 (weighted): ' + str(round(weighted * 100, 2)) + '%')

    exit()


def precompute_embs_lu(cd, keys, model):

    tok = Tokenizer(config.lm)

    all_embs = {}
    model.eval()

    with torch.no_grad():

        for i, element in enumerate(keys):

            ctx = cd.buildings_ctx[element]
            pois = torch.tensor(serialize_data([cd.entities[k] for k in ctx], tok, add_zero=True))
            pois_m = torch.tensor(np.where(pois != 0, 1, 0))
            emb = model.projection_p(
                torch.mean(torch.mean(model.lm(pois, attention_mask=pois_m)[0][:, :, :], 0), 0))
            image = torch.tensor(poly_to_raster(Polygon(cd.buildings_u[element]['points']))).unsqueeze(0).float()
            image_emb = model.projection_img(model.cv(image).squeeze())
            s = torch.tensor(norm_surface(cd.buildings_u[element]['area'])).unsqueeze(-1)
            s_emb = model.projection_s(s)
            all_embs[element] = {}
            all_embs[element]['ctx'] = emb.tolist()
            all_embs[element]['img'] = image_emb.tolist()
            all_embs[element]['s'] = s_emb.tolist()

            s = 'Precomputing building embeddings... (' + str(round((i + 1) / len(keys) * 100, 2)) + '%)'
            in_place_print(s)

    print(config.flush)
    save(all_embs, cd.city + '/Data/building_embeddings.json')


def land_use_prediction(city, cd, model, training=True, cm=False):

    if not os.path.isfile(city + '/Data/land_use.json'):
        print('Preprocessing ' + city + ' land use data...')
        land_use_to_json(city)

    with open(city + '/Data/land_use.json') as f:
        lu_data = json.load(f)

    lu_data_u = {}

    for key in lu_data.keys():
        if key in cd.buildings_u.keys():
            lu_data_u[key] = lu_data[key]

    k_list = list(lu_data_u.keys())

    if not os.path.isfile(city + '/Data/building_embeddings.json'):
        precompute_embs_lu(cd, list(lu_data_u.keys()), model)

    classes_d = {}

    for k in k_list:
        if lu_data_u[k] not in classes_d:
            classes_d[lu_data_u[k]] = [k]
        else:
            classes_d[lu_data_u[k]].append(k)

    classes_one_hot = {}
    o = 0
    for k in classes_d:
        classes_one_hot[k] = o
        o += 1

    train, valid, test = [], [], []

    for c in classes_d:

        random.shuffle(classes_d[c])
        for k in classes_d[c][:int(len(classes_d[c])/10*6)]:
            train.append(k)

        for k in classes_d[c][int(len(classes_d[c])/10*6):int(len(classes_d[c])/10*8)]:
            valid.append(k)

        for k in classes_d[c][int(len(classes_d[c])/10*8):]:
            test.append(k)

    print('Loading precomputed embeddings...')
    with open(str(city) + '/Data/building_embeddings.json', encoding='utf-8') as f:
        b_embs = json.load(f)

    ds_model = LULinear(len(list(set(lu_data_u.values()))), model.emb_size)
    train_land_use_prediction(cd, b_embs, ds_model, lu_data_u, classes_one_hot, train, valid, test, training=training,
                              cm=cm)


def freq_dic(data, classes_one_hot):

    freq = {}
    tot = len(data)

    for sid in data:

        c = classes_one_hot[data[sid]]

        if c not in freq:
            freq[c] = 1
        else:
            freq[c] += 1

    for k, v in freq.items():
        freq[k] = 1/(v/tot)

    return torch.tensor(np.array(list(freq.values()))).float()


def train_land_use_prediction(cd, b_embs, ds_model, lu_data_u, classes_one_hot, train, valid, test, training, cm=False):

    if not cm:
        print(config.sep_width * "-")
        print('Supervised Training')
        print('(Land use prediction)')

    opt = optim.Adam(params=ds_model.parameters(), lr=config.ds_lr)
    bs = config.ds_bs
    num_steps = (len(train) // bs) * config.ds_epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)
    criterion = nn.NLLLoss()

    best_f1 = 0

    for epoch in range(config.ds_epochs):

        if not training:
            break

        print_epoch(epoch + 1, config.ds_epochs)

        random.shuffle(train)
        opt.zero_grad()

        idx = 0
        step = 0

        while idx < len(train):

            ds_model.train()

            if idx > len(train) - bs:
                batch_l = train[idx:]
                batch_y = torch.tensor([classes_one_hot[lu_data_u[k]] for k in batch_l])

            else:
                batch_l = train[idx:idx + bs]
                batch_y = torch.tensor([classes_one_hot[lu_data_u[k]] for k in batch_l])

            batch_vis = []
            batch_ctx = []
            batch_pe = []

            for element in batch_l:

                pe = []
                if element in cd.buildings_u:
                    b_u = cd.buildings_u[element]
                    if 'lat' in b_u and 'lon' in b_u:
                        pe = PE((b_u['lat'], b_u['lon']), config.pe_size)

                if not len(pe):
                    pe = np.zeros(config.pe_size)

                if element in b_embs:

                    emb = torch.tensor(b_embs[element]['ctx'])
                    batch_ctx.append(emb)
                    image_emb = torch.tensor(b_embs[element]['img'])
                    s_emb = torch.tensor(b_embs[element]['s'])
                    batch_vis.append((image_emb + s_emb) / 2)
                    batch_pe.append(torch.tensor(pe).float())

            batch_vis = torch.stack(batch_vis)
            batch_ctx = torch.stack(batch_ctx)
            batch_pe = torch.stack(batch_pe)

            p = ds_model(batch_vis, batch_ctx, batch_pe)

            loss = criterion(p, batch_y)
            step = p_step(loss, step)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()

            idx += bs

        f1_ = validate_land_use_prediction(cd, b_embs, ds_model, None, valid,
                                           [classes_one_hot[lu_data_u[k]] for k in valid], epoch, step, bs,
                                           testing=False)

        if f1_ > best_f1:
            best_f1 = f1_
            pickle.dump(ds_model, open(str(cd.city) + '/Model/model_land_use.pkl', 'wb'))

            with open(config.log_file_name, 'a') as out_f:
                print('[SAVED]', file=out_f)

            print('[SAVED]')

        print(config.sep_width * "-")

    with open(str(cd.city) + '/Model/model_land_use.pkl', 'rb') as f:
        ds_model = pickle.load(f)

    _ = validate_land_use_prediction(cd, b_embs, ds_model, list(classes_one_hot.keys()), test,
                                     [classes_one_hot[lu_data_u[k]] for k in test], None, None, bs, testing=True, cm=cm)


def validate_land_use_prediction(cd, b_embs, ds_model, c_names, valid, y, epoch, step, bs, testing=False, cm=False):

    for _ in range(2):
        print(config.sep_width * "-")

    if not testing:
        print('Validation - Epoch: ' + str(epoch + 1) + ' Step: ' + str(step))

    else:
        print('Testing best model...')

    print('(Land use prediction)')

    print(config.sep_width * "-")

    ds_model.eval()

    idx = 0

    pred = []

    while idx < len(valid):

        ds_model.train()

        if idx > len(valid) - bs:
            batch_l = valid[idx:]

        else:
            batch_l = valid[idx:idx + bs]

        batch_vis = []
        batch_ctx = []
        batch_pe = []

        with torch.no_grad():

            for element in batch_l:

                pe = []
                if element in cd.buildings_u:
                    b_u = cd.buildings_u[element]
                    if 'lat' in b_u and 'lon' in b_u:
                        pe = PE((b_u['lat'], b_u['lon']), config.pe_size)

                if not len(pe):
                    pe = np.zeros(config.pe_size)

                if element in b_embs:
                    emb = torch.tensor(b_embs[element]['ctx'])
                    batch_ctx.append(emb)
                    image_emb = torch.tensor(b_embs[element]['img'])
                    s_emb = torch.tensor(b_embs[element]['s'])
                    batch_vis.append((image_emb + s_emb) / 2)
                    batch_pe.append(torch.tensor(pe).float())

            batch_vis = torch.stack(batch_vis)
            batch_ctx = torch.stack(batch_ctx)
            batch_pe = torch.stack(batch_pe)

            batch_p = ds_model(batch_vis, batch_ctx, batch_pe)

        for p in batch_p:
            pred.append(torch.argmax(p).item())

        idx += bs

    micro = f1_score(pred, y, average='micro')
    macro = f1_score(pred, y, average='macro')
    accuracy = accuracy_score(y, pred)
    weighted = f1_score(pred, y, average='weighted')

    # if testing:
    #    print(f1_score(pred, y, average=None))

    print('F1 (macro): ' + str(round(macro * 100, 2)) + '% | F1 (weighted): ' + str(
        round(weighted * 100, 2)) + '% | Accuracy: ' + str(round(accuracy * 100, 2)) + '%')

    with open(config.log_file_name, 'a') as out_f:
        print('F1 (macro): ' + str(round(macro * 100, 2)) + '% | F1 (weighted): ' + str(
            round(weighted * 100, 2)) + '% | Accuracy: ' + str(round(accuracy * 100, 2)) + '%', file=out_f)

    if cm:
        show_cm(y, pred, c_names)

    return macro


def show_cm(y, pred, c_names):

    cm = confusion_matrix(y, pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=c_names, yticklabels=c_names,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.xticks(rotation=45)
    plt.xticks(fontsize=6)

    fig.tight_layout()
    plt.show()


def avg_speed_prediction(city, cd, model, training=True):

    if not os.path.isfile(city + '/Data/uber_speed_data_ppc.csv'):

        in_place_print('Preprocessing uber speed data...')

        t0 = time.time()
        m_speed_df = pd.DataFrame()
        ids = []
        m_speed = []
        speed_df = pd.read_csv(str(city) + '/Data/uber_speed_data.csv')
        speed_df = speed_df[['osm_way_id', 'speed_mph_mean', 'speed_mph_stddev']]

        speed_df = speed_df.groupby('osm_way_id')  # .mean().reset_index()

        c = 0

        for g in speed_df:
            c += 1
            if str(g[0]) not in cd.data.keys():
                continue

            group = g[1]
            ids.append(g[0])
            tot_speed = 0
            tot_var = 0
            group_speeds = list(group['speed_mph_mean'])
            group_stds = list(group['speed_mph_stddev'])

            for i in range(len(group_speeds)):
                var = 1
                if config.inverse_variance_weights:
                    var = group_stds[i]**2

                tot_speed += group_speeds[i]*1/var
                tot_var += 1/var

            m_speed.append(tot_speed/tot_var)
            in_place_print('Preprocessing uber speed data... (' + str(round(c/len(speed_df)*100, 1)) + '%)')

        print(config.flush)

        del speed_df
        m_speed_df['osm_way_id'] = ids
        m_speed_df['speed_mph_avg'] = m_speed

        m_speed_df.to_csv(city + '/Data/uber_speed_data_ppc.csv', index=False)

        print('Completed! ('+str(round(time.time()-t0, 2))+'s)\n')

    else:
        m_speed_df = pd.read_csv(str(city) + '/Data/uber_speed_data_ppc.csv')

    train, valid, test = split_data(m_speed_df)

    ds_model = AVGSpeedLinear(config.polyline_size)
    train_avg_speed_prediction(cd, model, ds_model, train, valid, test, training)


def make_relation_voc(relations):

    way_voc = {}

    for key in relations:

        r = relations[key]

        for way in r:

            w = str(way)

            if w not in way_voc:
                way_voc[w] = 1
            else:
                way_voc[w] += 1

    max_value = max([way_voc[k] for k in way_voc])

    for way in way_voc:
        way_voc[way] /= max_value

    return way_voc


def train_avg_speed_prediction(cd, model, ds_model, train, valid, test, training=True):

    print(config.sep_width * "-")
    print('Supervised Training')
    print('(Avg speed prediction)')

    train_e = {}
    for i in range(train.shape[0]):
        way_id = train.iloc[i]['osm_way_id']
        s = train.iloc[i]['speed_mph_avg']
        train_e[str(int(way_id))] = s

    if not os.path.exists(cd.city + "/Ways/edge_list.json"):

        with open(cd.city + "/Ways/edge_list", "rb") as fp:
            edges_in = pickle.load(fp)

        t_e = list(train['osm_way_id'])

        edges = {}
        for e in edges_in:
            if e[0] in t_e or e[1] in t_e:
                if e[0] not in edges:
                    edges[e[0]] = [e[1]]
                else:
                    edges[e[0]].append(e[1])

                if e[1] not in edges:
                    edges[e[1]] = [e[0]]
                else:
                    edges[e[1]].append(e[0])

        with open(cd.city + "/Ways/edge_list.json", 'w', encoding='utf-8') as fp:
            json.dump(edges, fp, ensure_ascii=False, indent=4)

    with open(cd.city + "/Ways/edge_list.json", encoding='utf-8') as fp:
        edges = json.load(fp)

    model.eval()
    opt = optim.Adam(params=ds_model.parameters(), lr=config.ds_lr/10)
    bs = config.ds_bs
    num_steps = (train.shape[0] // bs) * config.ds_epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)
    criterion = nn.MSELoss()

    way_voc = make_relation_voc(cd.relations)

    best_rmse = math.inf

    for epoch in range(config.ds_epochs):

        print_epoch(epoch + 1, config.ds_epochs)

        train = train.sample(frac=1)
        opt.zero_grad()

        idx = 0
        step = 0

        while idx < train.shape[0]:

            ds_model.train()

            if idx > train.shape[0] - bs:
                batch_l = list(train.iloc[idx:, 0])
                batch_y = torch.tensor(np.array(train.iloc[idx:, 1])).float()

            else:
                batch_l = list(train.iloc[idx: idx + bs, 0])
                batch_y = torch.tensor(np.array(train.iloc[idx: idx + bs, 1])).float()

            batch_x = []
            batch_f = []
            batch_pe = []
            batch_rel = []

            for way in batch_l:

                batch_f.append(torch.tensor(polyline_features(cd.polylines[str(way)])))
                pe = PE(random.choice(cd.polylines[str(way)]['points']), config.pe_size//2)
                batch_pe.append(torch.tensor(pe).float())
                batch_x.append(torch.tensor(polyline_nodes(cd.polylines[str(way)], train_e, edges)).float())

                if str(way) in way_voc:
                    batch_rel.append(torch.tensor(way_voc[str(way)]*10))
                else:
                    batch_rel.append(torch.tensor(0))

            batch_x = torch.stack(batch_x)
            batch_f = torch.stack(batch_f)
            batch_pe = torch.stack(batch_pe)
            batch_rel = torch.stack(batch_rel)

            p = ds_model(batch_x, batch_f, batch_pe, batch_rel)

            loss = torch.sqrt(criterion(p, batch_y))
            step = p_step(loss, step)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()

            idx += bs

        rmse_ = validate_avg_speed_prediction(cd, model, ds_model, valid, epoch, step, bs, edges, train_e, way_voc, testing=False)
        if rmse_ < best_rmse:
            best_rmse = rmse_
            pickle.dump(ds_model, open(str(cd.city) + '/Model/model_avg_speed.pkl', 'wb'))
            print('[SAVED]')
        print(config.sep_width * "-")

    with open(str(cd.city) + '/Model/model_avg_speed.pkl', 'rb') as f:
        ds_model = pickle.load(f)

    _ = validate_avg_speed_prediction(cd, model, ds_model, test, None, None, bs, edges, train_e, way_voc, testing=True)


def validate_avg_speed_prediction(cd, model, ds_model, valid, epoch, step, bs, edges, train_e, way_voc, testing):

    for _ in range(2):
        print(config.sep_width * "-")

    if not testing:
        print('Validation - Epoch: ' + str(epoch+1) + ' Step: ' + str(step))

    else:
        print('Testing best model...')

    print('(Avg speed prediction)')

    print(config.sep_width * "-")

    model.eval()
    ds_model.eval()

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    idx = 0

    pred = []

    while idx < valid.shape[0]:

        if idx > valid.shape[0] - bs:
            batch_l = list(valid.iloc[idx:, 0])

        else:
            batch_l = list(valid.iloc[idx: idx + bs, 0])

        batch_x = []
        batch_f = []
        batch_pe = []
        batch_rel = []

        with torch.no_grad():

            for way in batch_l:

                batch_f.append(torch.tensor(polyline_features(cd.polylines[str(way)])))
                pe = PE(random.choice(cd.polylines[str(way)]['points']), config.pe_size//2)
                batch_pe.append(torch.tensor(pe).float())
                batch_x.append(torch.tensor(polyline_nodes(cd.polylines[str(way)], train_e, edges)).float())

                if str(way) in way_voc:
                    batch_rel.append(torch.tensor(way_voc[str(way)]))
                else:
                    batch_rel.append(torch.tensor(0))

            batch_x = torch.stack(batch_x)
            batch_f = torch.stack(batch_f)
            batch_pe = torch.stack(batch_pe)
            batch_rel = torch.stack(batch_rel)

            p = ds_model(batch_x, batch_f, batch_pe, batch_rel)
            pred.append(p)

        idx += bs

    pred = torch.cat(pred, 0)
    y = torch.tensor(np.array(valid.iloc[:, 1])).float()

    mse_ = mse(pred, y).item()
    rmse_ = sqrt(mse_)
    l1_ = l1(pred, y).item()
    r2 = r2_score(pred, y).item()
    mape_ = mape(y, pred)

    print('RMSE: ' + str(round(rmse_, 2)) + ' | MAE: ' + str(round(l1_, 2)) + ' | R2: ' + str(
        round(r2, 4)) + ' | MAPE: ' + str(round(mape_, 2)))

    print(config.sep_width * "-")

    return rmse_


downstream_functions = {'avg_speed': avg_speed_prediction, 'build_func': land_use_prediction,
                        'pop_density': population_density}
