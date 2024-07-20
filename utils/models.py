import torch
from torch import nn, optim
import torch.nn.functional as F
# import timm
from transformers import DistilBertModel, RobertaModel
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

import config
from utils.pre_processing import *


def PE(pos, dim):

    if dim % 4:
        print("Error: 'dim' must be a multiple of 4")
        exit()

    p_enc = []

    for i in range(0, dim // 2, 2):

        for loc in pos:
            w_k = config.lambda_ / pow(10000, i / (dim // 2))
            p_enc.append(config.lambda_ * sin(loc * w_k))
            p_enc.append(config.lambda_ * cos(loc * w_k))

    return np.array(p_enc)


class HPRegressor(nn.Module):

    def __init__(self, emb_size):
        super().__init__()

        self.emb_size = emb_size

        self.regressor = nn.Sequential(
            nn.Linear(2*self.emb_size + config.pe_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, poly, n, p):

        return self.regressor(torch.cat([poly, n, p], dim=1)).squeeze()


class LULinear(nn.Module):

    def __init__(self, n_classes, poly_emb_size):
        super().__init__()

        self.poly_emb_size = poly_emb_size
        self.n_classes = n_classes

        self.polyline_sequential = nn.Sequential(
            nn.Linear(self.poly_emb_size * 2 + config.pe_size, 256),
            # nn.Linear(self.poly_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes)
        )

    def forward(self, x, ctx, pe):

        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        if len(ctx.shape) < 2:
            ctx = ctx.unsqueeze(0)

        if len(pe.shape) < 2:
            pe = pe.unsqueeze(0)

        x = self.polyline_sequential(torch.cat([x, ctx, pe], dim=1))
        # x = self.polyline_sequential(x)

        return F.log_softmax(x, dim=1)


class GeoVectorsClassifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        self.sequential = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes)
        )

    def forward(self, x):

        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        x = self.sequential(x)

        return F.log_softmax(x, dim=1)


class GeoVectorsRegressor(nn.Module):

    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        x = self.sequential(x)

        return x.squeeze()


class AVGSpeedLinear(nn.Module):

    def __init__(self, way_emb_size):
        super().__init__()

        self.way_emb_size = way_emb_size

        self.way_emb_linear = nn.Linear(self.way_emb_size, 128)
        # self.way_emb_linear = nn.Linear(128, 128)

        hw_emb = nn.Embedding(osm_strings.hw_emb, 128)
        lane_emb = nn.Embedding(osm_strings.lane_emb, 128)
        ms_emb = nn.Embedding(osm_strings.ms_emb, 128)

        self.rel_emb = nn.Linear(1, 128)
        self.emb_list = [hw_emb, lane_emb, ms_emb]

        self.polyline_sequential = nn.Sequential(
            nn.Linear(640 + config.pe_size//2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, way_emb, f_emb, pe, rel):

        rel_emb = self.rel_emb(rel.unsqueeze(-1))

        way_emb = self.way_emb_linear(way_emb).squeeze()

        emb = [way_emb, pe, rel_emb]
        # emb = []
        for i in range(len(self.emb_list)):
            emb.append(self.emb_list[i](f_emb[:, i]))

        # emb = [way_emb, rel_emb]
        polyline_emb = torch.cat(emb, 1)

        return self.polyline_sequential(polyline_emb).squeeze()


class SSModel(nn.Module):

    def __init__(self, city, lm, emb_size):
        super().__init__()

        self.city = city

        self.lm_name = lm
        if lm not in config.lm_names:
            self.lm_name = config.default_lm

        self.lm = None
        self.get_lm()
        self.emb_size = emb_size

        resnet18 = torch.hub.load(config.torch_vision, config.vision_model)
        self.cv = torch.nn.Sequential(*(list(resnet18.children())[:-1]))

        self.lm_hidden_size = config.lm_hidden_sizes[self.lm_name]
        self.vhs = config.vision_hidden_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.attention_linear = nn.Linear(self.lm_hidden_size, 1)

        self.attention_sequential = nn.Sequential(
            nn.Linear(self.lm_hidden_size, self.lm_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lm_hidden_size, 1)
        )

        self.projection_s = nn.Sequential(
            nn.Linear(1, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # self.context_proj = nn.Linear(self.lm_hidden_size, emb_size)
        self.projection_p = nn.Sequential(
            nn.Linear(self.lm_hidden_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        self.projection_img = nn.Linear(self.vhs, emb_size)

        self.fc = nn.Sequential(
            nn.Linear(self.lm_hidden_size, self.lm_hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.lm_hidden_size//2, 1)
        )

        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, queries, keys, surfaces=None, task='info_max'):

        query_emb = []
        key_emb = []

        if task == 'raster':

            self.lm.eval()

            query, query_m = q_tensor_mask(queries)
            keys = torch.tensor(keys).float()
            surfaces = torch.tensor(surfaces).unsqueeze(-1)

            with torch.no_grad():
                query_emb = self.projection_p(torch.mean(self.lm(query, attention_mask=query_m)[0][:, :, :], 1))
            key_emb = self.projection_img(self.cv(keys).squeeze())
            s_emb = self.projection_s(surfaces)
            key_emb = (key_emb + s_emb)/2

        else:

            for i in range(len(queries)):

                query = queries[i]
                key = keys[i]

                query, query_m, key, key_m = qk_tensor_mask(query, key)

                if task == 'info_max':

                    q_enc = self.lm(query, attention_mask=query_m)[0][:, :, :]
                    q_emb = self.projection_p(torch.mean(torch.mean(q_enc, 0), 0))
                    query_emb.append(q_emb)

                    k_enc = self.lm(key, attention_mask=key_m)[0][:, :, :]
                    k_emb = self.projection_p(torch.mean(torch.mean(k_enc, 0), 0))
                    key_emb.append(k_emb)

                elif task == 'relation_seq':

                    self.lm.eval()

                    with torch.no_grad():
                        q_emb = self.projection_p(torch.mean(self.lm(query, attention_mask=query_m)[0][:, :, :], 1))
                        k_emb = self.projection_p(torch.mean(self.lm(key, attention_mask=key_m)[0][:, :, :], 1))

                    q_attn = F.softmax(self.attention_sequential(q_emb), 0)
                    q_emb = torch.sum(q_emb * q_attn, 0)
                    # q_emb = torch.mean(q_emb, 0)

                    k_attn = F.softmax(self.attention_sequential(k_emb), 0)
                    k_emb = torch.sum(k_emb * k_attn, 0)
                    # k_emb = torch.mean(k_emb, 0)

                    query_emb.append(q_emb)
                    key_emb.append(k_emb)

            query_emb = torch.stack(query_emb)
            key_emb = torch.stack(key_emb)

        return query_emb, key_emb

    def get_lm(self):

        if self.lm_name == 'bert':
            self.lm = BertModel.from_pretrained(config.lm_names[self.lm_name])
        elif self.lm_name == 'roberta':
            self.lm = RobertaModel.from_pretrained(config.lm_names[self.lm_name])
        elif self.lm_name == 'distilbert':
            self.lm = DistilBertModel.from_pretrained(config.lm_names[self.lm_name])
        else:
            self.lm = BertModel.from_pretrained(config.lm_names[self.lm_name])

    def predict_context(self, query_emb, key_emb):

        p0 = torch.abs(torch.sub(query_emb[0], key_emb[0]))
        p1 = torch.abs(torch.sub(query_emb[1], key_emb[1]))

        n0 = torch.abs(torch.sub(query_emb[0], key_emb[1]))
        n1 = torch.abs(torch.sub(query_emb[1], key_emb[0]))

        p0 = self.sigmoid(self.fc(p0))
        p1 = self.sigmoid(self.fc(p1))
        n0 = self.sigmoid(self.fc(n0))
        n1 = self.sigmoid(self.fc(n1))

        return p0, p1, n0, n1


def device_as(t1, t2):

    return t1.to(t2.device)


def q_tensor_mask(query):

    query = torch.tensor(query)

    query_m = np.where(query != 0, 1, 0)
    query_m = torch.tensor(query_m)

    return query, query_m


def qk_tensor_mask(query, key):

    query = torch.tensor(query)
    key = torch.tensor(key)

    query_m = np.where(query != 0, 1, 0)
    query_m = torch.tensor(query_m)

    key_m = np.where(key != 0, 1, 0)
    key_m = torch.tensor(key_m)

    return query, query_m, key, key_m


class PolyContext(nn.Module):

    def __init__(self, batch_size):

        super().__init__()
        self.batch_size = batch_size

    @staticmethod
    def forward(p0, p1, n0, n1):

        y = torch.tensor([1., 1., 0., 0.])
        p = torch.cat([p0, p1, n0, n1])

        loss = nn.BCELoss()
        return loss(p, y)


class InfoNCE(nn.Module):

    def __init__(self, batch_size, temperature=0.5):

        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    @staticmethod
    def compute_batch_sim(a, b):

        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, polyline_embs, c_embs):

        b_size = polyline_embs.shape[0]
        polyline_norm = F.normalize(polyline_embs, p=2, dim=1)
        c_norm = F.normalize(c_embs, p=2, dim=1)

        similarity_matrix = self.compute_batch_sim(polyline_norm, c_norm)

        sim_ij = torch.diag(similarity_matrix, b_size)
        sim_ji = torch.diag(similarity_matrix, -b_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)

        return loss


class RasterSim(nn.Module):

    def __init__(self, batch_size, hidden, temperature=0.5):

        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.hidden = hidden

    @staticmethod
    def compute_batch_sim(a, b):

        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, t_embs, r_embs):

        loss = 0
        for i in range(self.batch_size):
            loss += torch.sum(torch.abs(t_embs[i] - r_embs[i]))

        loss = loss/self.batch_size/sqrt(self.hidden)

        return loss
