import torch


class CFG:
    # device =torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")  # Use GPU is avaliable 
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') 
    if(device.type == "cuda" or device.type == "mps"):
        torch.cuda.empty_cache()
        cuda = True
    else:
        cuda = False
    debug = False
    if debug:    
        data_path = './data/casp12_data_30/'
    else: 
        data_path = './data/casp12_data_100/' #'./data/data2'
    inference_path = './data/inference_data'
    results_path = './res/results-emb/'
    seed = 42
    # Train data parameters
    homothresh = 0.9
    constraint = True
    split_train_size = 0.8
    debug_size  = 10
    sigma = 0.1 # for score matching loss
    # Model parameters
    h = 0.1
    coords_emb_size = 48
    embedding_size = 20
    filters = 64
    num_layers = 3
    dropout_rate = 0.2
    model_path = "./res/trianed_models-newDecoys/"
    
    gaussian_coef = -0.008*1e1
    #training parameters
    lr = 0.0001
    wd = 0.00001
    batch_size = 1
    num_workers = 2
    N = 10
    num_epochs = 50
    seq_len = 600
    SM = False # score matching loss
    gradient_penalty = True
    decoy_threshold = 20
    max_grad_norm = 10.0
    clip_grad_norm = False
    reg_alpha = 0.1
    # defalut parameters
    torch_default_dtype = torch.float32
    precision = torch.float32