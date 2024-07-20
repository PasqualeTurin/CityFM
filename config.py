# from utils.prediction import *

log_file_name = 'log_file.txt'
flush = ''
filter_out_d = 50000
context_d = 100
context_b = 100
context_out = 500
img_h = 224
img_w = 224
rel_window = 7
lm = 'bert'
lm_names = {'bert': 'bert-base-uncased', 'distilbert': 'distilbert-base-uncased', 'roberta': 'roberta-base'}
lm_hidden_sizes = {'bert': 768, 'distilbert': 768, 'roberta': 768}
default_lm = 'bert'
no_context = 'context: none'
torch_vision = 'pytorch/vision:v0.10.0'
vision_model = 'resnet18'
vision_model_weights = 'ResNet18_Weights.DEFAULT'
vision_hidden_size = 512
inverse_variance_weights = False
# info_max
bs = 256
index = 64
# raster to tags
bs_raster = 256
index_raster = 64
# relation_seq
bs_rel = 1024
index_rel = 1024
lr = 1e-4
# downstream
ds_bs = 128
ds_lr = 5e-3
ds_epochs = 20
validate_every = 50

device = 'cpu'
n_epochs = 5
lambda_ = 100
hub_threshold = 0.02
pd_limu = 3
pe_size = 256
polyline_size = 256
m_index = 64*1e2
max_surface = 5000
downstream_tasks = ['avg_speed', 'build_func', 'pop_density']
sep_width = 50
