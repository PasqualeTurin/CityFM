import config
from utils.pre_processing import *
from utils.errors import *
from utils.models import *


class CityData:

    def __init__(self, city, buildings_u=False):

        self.city = city

        with open(str(city) + '/Data/data.json', encoding='utf-8') as f:
            self.data = json.load(f)

        with open(str(city) + '/Data/entities.json', encoding='utf-8') as f:
            self.entities = json.load(f)

        with open(str(city) + '/Data/relations.json', encoding='utf-8') as f:
            self.relations = json.load(f)

        with open(str(city) + '/Data/polylines.json', encoding='utf-8') as f:
            self.polylines = json.load(f)

        if buildings_u:
            with open(str(city) + '/Data/buildings_untagged.json', encoding='utf-8') as f:
                self.buildings_u = json.load(f)

            with open(str(city) + '/Data/buildings_context.json', encoding='utf-8') as f:
                self.buildings_ctx = json.load(f)


class Tokenizer:

    def __init__(self, lm):

        if lm == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config.lm_names[lm])
        elif lm == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(config.lm_names[lm])
        elif lm == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(config.lm_names[lm])
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.lm_names[config.default_lm])

    def tokenize_batch(self, data, add_zero=False):

        if not data:
            data.append(config.no_context)

        elif add_zero:
            data.append(config.no_context)

        tok_data = []
        for d in data:

            tok_d = self.tokenizer.tokenize('[CLS] ' + d + ' [SEP]')
            tok_data.append(tok_d)

        tag_len = max([len(d) for d in tok_data])

        ids_data = []
        for tok_d in tok_data:

            if len(tok_d) < tag_len:
                tok_d += ['[PAD]'] * (tag_len - len(tok_d))
            else:
                tok_d = tok_d[:tag_len]

            ids_data.append(self.tokenizer.convert_tokens_to_ids(tok_d))

        return ids_data


def load_poly_dictionary(city):

    with open(str(city) + '/Data/polyline_to_id.json', encoding='utf-8') as f:
        pd = json.load(f)

    return pd


def serialize_data(data, tok, add_zero=False):

    entities_tags = []

    for d in data:

        tags = d['tags']
        rq_tags = {k: tags[k] for k in osm_strings.use_tags if k in tags}
        s_tags = json_string(rq_tags)
        entities_tags.append(s_tags)

    entities_tags = np.array(tok.tokenize_batch(entities_tags, add_zero=add_zero))
    return entities_tags


def serialize_cat(data, voc):

    entities_tags = []

    for d in data:
        tags = d['tags']
        rq_tags = {k: tags[k] for k in osm_strings.use_tags if k in tags}
        s_tags = json_string(rq_tags)
        entities_tags.append(voc[s_tags])

    if not data:
        entities_tags.append(voc[config.no_context])

    return entities_tags


def prepare_sequences(data, entities, relations):

    queries = np.array(list(relations.keys()))[:20]
    w = (check_window(config.rel_window) - 1) // 2
    tok = Tokenizer(config.lm)

    anchor_l = []
    pos_l = []

    for c, query in enumerate(queries):

        seq = relations[query]
        seq = [str(x) for x in seq]

        for i in range(len(seq) - 2 * w):

            for j in [x for x in range(-w, w + 1) if x != 0]:

                if seq[i + w] in data and seq[i + w + j] in data:

                    # if len(data[seq[i + w]]['context']) and len(data[seq[i + w + j]]['context']):
                    # if len(data[seq[i + w]]['context']) > 2:

                    # ctx = data[seq[i + w]]['context']
                    # random.shuffle(ctx)
                    # ctx.pop()

                    anchor_ctx = serialize_data([entities[k] for k in data[seq[i + w]]['context']], tok, add_zero=False)
                    pos_ctx = serialize_data([entities[k] for k in data[seq[i + w + j]]['context']], tok, add_zero=False)

                    anchor_l.append(anchor_ctx)
                    pos_l.append(pos_ctx)

        s = 'Serializing sequences... (' + str(round((c + 1) / len(queries) * 100, 2)) + '%)'
        in_place_print(s)

    print(config.flush)
    # print(len(anchor_l))

    return anchor_l, pos_l


def build_category_vocab(entities):

    c = 1
    vocab = {config.no_context: 0}

    for e in entities:

        tags = entities[e]['tags']
        rq_tags = {k: tags[k] for k in osm_strings.use_tags if k in tags}
        s_tags = json_string(rq_tags)

        if s_tags not in vocab:
            vocab[s_tags] = c
            c += 1

    return vocab


def write_step(city, step_loss):

    with open(str(city) + '/Model/loss.txt', 'w') as f:
        for s_l in step_loss:
            f.write(str(s_l)+'\n')


def save_model(city, model, e, task):
    pickle.dump(model, open(str(city) + '/Model/model.pkl', 'wb'))


def check_window(w):

    if w < 3:
        window_size_error()

    if not w % 2:
        return w-1

    return w


def print_step(loss, step_loss, step):

    step += 1

    if config.device == 'cuda':
        l_item = loss.cpu().detach().numpy()
    else:
        l_item = loss.item()

    step_loss.append(l_item)

    print('Step ' + str(step) + ' Loss: ' + str(round(l_item, 4)))

    return step


def p_step(loss, step):

    step += 1

    if config.device == 'cuda':
        l_item = loss.cpu().detach().numpy()
    else:
        l_item = loss.item()

    with open(config.log_file_name, 'a') as out_f:
        print('Step ' + str(step) + ' Loss: ' + str(round(l_item, 4)), file=out_f)

    print('Step ' + str(step) + ' Loss: ' + str(round(l_item, 4)))

    return step


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


def u_sampling(cd, poly_queries, idx):

    random.shuffle(poly_queries)
    used_c = []
    queries = []

    for p_q in poly_queries:

        tags = string_of_tags(cd.entities[p_q])
        if tags not in used_c:
            used_c.append(tags)
            queries.append(p_q)

            if len(queries) == idx:
                return queries

    i = len(poly_queries) - 1
    while len(queries) < idx:
        queries.append(poly_queries[i])
        i -= 1

    return queries


def train_SS_im(cd, model):

    print(config.sep_width * "-")
    print('Self-supervised Training')
    print('(Infomax)')

    queries = np.array(list(cd.data.keys()))
    # pd = load_poly_dictionary(cd.city)
    tok = Tokenizer(config.lm)
    opt = optim.Adam(params=model.parameters(), lr=config.lr)
    bs = config.bs
    num_steps = (len(queries) // bs)//5 # * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

    step_loss = []

    info_nce = InfoNCE(config.index)
    # poly_c = PolyContext(2)

    for epoch in range(config.n_epochs):

        print_epoch(epoch+1, config.n_epochs)

        step = 0

        # noinspection PyTypeChecker
        random.shuffle(queries)
        idx = 0
        loss = torch.tensor(0.)
        opt.zero_grad()

        while idx < len(queries):

            if idx > len(queries) - config.index:
                # query = queries[idx:]
                break

            else:
                query = queries[idx:idx + config.index]

            ctx = []
            query_p = []

            for q in query:
                context = cd.data[q]['context']
                if not context:
                    query_p.append(serialize_data([], tok))

                else:
                    r = random.randint(0, len(context))
                    if r == len(context):
                        query_p.append(serialize_data([], tok))

                    else:
                        if context[r] in cd.entities:
                            query_p.append(serialize_data([cd.entities[context[r]]], tok))
                        else:
                            query_p.append(serialize_data([], tok))

                ctx.append(serialize_data([cd.entities[k] for k in context if k in cd.entities], tok, add_zero=True))

            p_line_embs, ctx_embs = model(query_p, ctx, task='info_max')
            loss += info_nce(p_line_embs, ctx_embs)

            idx += config.index

            if idx % bs == 0:
                step = print_step(loss, step_loss, step)
                loss.backward()
                opt.step()
                scheduler.step()
                loss = torch.tensor(0.)
                opt.zero_grad()

            if idx > config.m_index:
                break

        write_step(cd.city, step_loss)
        save_model(cd.city, model, epoch+1, 'im')

    print('\n\n')


def train_SS_raster(cd, model):

    print(config.sep_width * "-")
    print('Self-supervised Training')
    print('(Polygon)')

    poly_queries = filter_polygons(cd)
    opt = optim.Adam(params=model.parameters(), lr=config.lr * 10)
    tok = Tokenizer(config.lm)
    bs = config.bs_raster

    step_loss = []

    info_nce = InfoNCE(config.index_raster)
    # raster_sim = RasterSim(config.index_raster, hidden=model.emb_size)

    for epoch in range(config.n_epochs):

        print_epoch(epoch+1, config.n_epochs)

        step = 0

        # noinspection PyTypeChecker
        # random.shuffle(poly_queries)
        idx = 0
        loss = torch.tensor(0.)
        opt.zero_grad()

        while idx < len(poly_queries):

            if idx > len(poly_queries) - config.index_raster:
                # query = poly_keys[idx:]
                break

            else:
                # query = poly_queries[idx:idx + config.index_raster]
                query = u_sampling(cd, poly_queries, config.index_raster)

            tags = []
            rasters = []
            surfaces = []

            for q in query:
                entity = cd.entities[q]
                tags.append(entity)
                rasters.append(poly_to_raster(Polygon(entity['points'])))
                surfaces.append(norm_surface(entity['area']))

            tags = np.array(serialize_data(tags, tok))
            rasters = np.array(rasters)
            t_embs, r_embs = model(tags, rasters, surfaces, task='raster')
            loss += info_nce(t_embs, r_embs)

            idx += config.index_raster

            if idx % bs == 0:
                step = print_step(loss, step_loss, step)
                loss.backward()
                opt.step()
                loss = torch.tensor(0.)
                opt.zero_grad()

            if idx > config.m_index:
                break

        write_step(cd.city, step_loss)
        save_model(cd.city, model, epoch + 1, 'raster')

    print('\n\n')


def train_SS_rel(cd, model):

    print(config.sep_width * "-")
    print('Self-supervised Training')
    print('(Relation Contrastive Learning)')

    queries = np.array(list(cd.data.keys()))
    # pd = load_poly_dictionary(cd.city)
    tok = Tokenizer(config.lm)
    opt = optim.Adam(params=model.parameters(), lr=config.lr/10)
    bs = config.bs_rel
    num_steps = (len(queries) // bs) // 5  # * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=num_steps)

    way_voc = make_relation_voc(cd.relations)

    step_loss = []
    # poly_c = PolyContext(2)

    for epoch in range(config.n_epochs):

        print_epoch(epoch + 1, config.n_epochs)

        step = 0

        # noinspection PyTypeChecker
        random.shuffle(queries)
        idx = 0
        loss = torch.tensor(0.)
        opt.zero_grad()

        while idx < len(queries):

            if idx > len(queries) - config.index_rel:
                # query = queries[idx:]
                break

            else:
                query = queries[idx:idx + config.index_rel]

            ctx = []
            ctx_hub = []

            qk_pairs = []
            qk_values = []

            for q in query:

                if q in way_voc and way_voc[q] not in qk_values:

                    for k in query:

                        if q != k and k in way_voc and not any(q in sublist for sublist in qk_pairs) and not any(
                                k in sublist for sublist in qk_pairs):

                            if abs(way_voc[q] - way_voc[k]) <= config.hub_threshold:

                                qk_pairs.append([q, k])

                                if way_voc[q] not in qk_values:
                                    qk_values.append(way_voc[q])

                                if way_voc[k] not in qk_values:
                                    qk_values.append(way_voc[k])

            if len(qk_pairs) < 2:
                continue

            info_nce = InfoNCE(len(qk_pairs))

            for q in qk_pairs:
                context = cd.data[q[0]]['context']
                ctx.append(serialize_data([cd.entities[k] for k in context if k in cd.entities], tok, add_zero=True))

                context = cd.data[q[1]]['context']
                ctx_hub.append(
                    serialize_data([cd.entities[k] for k in context if k in cd.entities], tok, add_zero=True))

            p_line_embs, ctx_embs = model(ctx, ctx_hub, task='info_max')
            loss += info_nce(p_line_embs, ctx_embs)

            idx += config.index_rel

            if idx % bs == 0:
                step = print_step(loss, step_loss, step)
                loss.backward()
                opt.step()
                scheduler.step()
                loss = torch.tensor(0.)
                opt.zero_grad()

        write_step(cd.city, step_loss)
        save_model(cd.city, model, epoch + 1, 'rel')

    print('\n\n')
