import pickle
import torch
import numpy as np
from utils.pre_processing import *
from utils.models import SSModel, PE
from utils.training import Tokenizer, serialize_data



def encode_nodes(city, nodes):
    
    with open(city+'/Model/model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    tok = Tokenizer(model.lm_name)
    
    model.eval()
    
    textual_embs = []
    positional_embs = []
    
    for node in nodes:
        
        loc = (node['lat'], node['lon'])
        pe = torch.tensor(PE(loc, 256))
        poi = torch.tensor(serialize_data([node], tok))
        poi_m = torch.tensor(np.where(poi != 0, 1, 0))
        emb = model.projection_p(torch.mean(model.lm(poi, attention_mask=poi_m)[0][:, :, :].squeeze(), 0))
        
        positional_embs.append(pe)
        textual_embs.append(emb)
        
    return torch.stack(positional_embs), torch.stack(textual_embs)



def encode_polygons(city, polygons):

    with open(city+'/Model/model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    tok = Tokenizer(model.lm_name)
    
    model.eval()
    
    textual_embs = []
    visual_embs = []
    positional_embs = []
    for polygon in polygons:
    
        loc = (polygon['lat'], polygon['lon'])
        pe = torch.tensor(PE(loc, 256))
    
        image = torch.tensor(poly_to_raster(Polygon(polygon['points']))).unsqueeze(0).float()
        tags = torch.tensor(serialize_data([polygon], tok))
        tags_m = torch.tensor(np.where(tags != 0, 1, 0))
        s = torch.tensor(norm_surface(polygon['area'])).unsqueeze(-1)

        text_emb = model.projection_p(torch.mean(model.lm(tags, attention_mask=tags_m)[0][:, :, :].squeeze(), 0))

        image_emb = model.projection_img(model.cv(image).squeeze())
        s_emb = model.projection_s(s)
        visual_emb = (image_emb + s_emb) / 2
        
        positional_embs.append(pe)
        textual_embs.append(text_emb)
        visual_embs.append(visual_emb)
        
    
    return torch.stack(positional_embs), torch.stack(textual_embs), torch.stack(visual_embs)


def encode_polylines(polylines):

    features_embs = []
    positional_embs = []
    
    for polyline in polylines:
        
        p_feat = polyline_features(polyline)
        features_embs.append(torch.cat([torch.tensor([p_feat[0]] * 64), torch.tensor([p_feat[1]] * 64), torch.tensor([p_feat[2]] * 64)]))
        
        positional_embs.append(torch.tensor(PE(random.choice(polyline['points']), 256)))
    
    return torch.stack(positional_embs), torch.stack(features_embs)
