import os
import allel
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn

def load_vcf(vcf_path, tabix_path, extract_position=True, unique_sample=True):
    genotype_arr = []
    callset = allel.read_vcf(vcf_path, tabix=tabix_path, alt_number=2)
    genotype_arr = callset['calldata/GT']
    #PL_highpass = callset['calldata/PL']
    sample_ids = callset['samples']
    genotype_arr = allel.GenotypeArray(genotype_arr,dtype='i1')
    allele_count = genotype_arr.count_alleles()
    genotype_arr = genotype_arr.to_n_alt(fill=-1)
    print("my_GT SHAPE:",genotype_arr.shape)
    genotype_arr = genotype_arr.T
    data = np.reshape(genotype_arr, (genotype_arr.shape[0], genotype_arr.shape[1], 1))
    #print("data shape: ", data.shape)
    maf = allele_count.to_frequencies()
    a1 = np.round(maf[:,0],6)
    a2 = np.round(maf[:,1],6)
    maf = np.min(np.stack([a1,a2]), axis=0)
    if unique_sample==True:
        _, unique_indexes = np.unique(genotype_arr, axis=0, return_index=True)
        #print(len(unique_indexes))
        data = data[unique_indexes]
        sample_ids = sample_ids[unique_indexes]
        #PL_highpass = PL_highpass[unique_indexes]
        # variant_pos = callset['variants/POS'][unique_indexes]
    if extract_position:
        return sample_ids, data, maf, callset['variants/POS']    
    print("initial input dim: ", data.shape)
    return sample_ids, data, maf

# def convert2prob(log_gl):
#     return np.power(10,-log_gl/10)

# def normalize_pl(pl_array):
#     normalized_arr = pl_array / pl_array.sum(axis=2)[:,:, np.newaxis]
#     return normalized_arr

def get_allele_p(high_pass):
    Pr_highpass = high_pass[:,:,0] + high_pass[:,:,1] 
    Pa_highpass = high_pass[:,:,2] + high_pass[:,:,1] 
    return Pr_highpass, Pa_highpass

def filter_by_MAF(data, MAF):
    original_shape = data.shape
    
    ys_values = data.copy().transpose((1,0,2))
    
    indexes = list(range(len(MAF)))
    ys_dict = dict(zip(indexes,ys_values))
    
    MAF_a = np.array(MAF)
    filtered_mask = MAF_a >= min_MAF
    indexes = np.array(indexes)
    filtered_keys = indexes[filtered_mask]
    
    subdict = {x: ys_dict[x] for x in filtered_keys}
    sorted_subdict = dict(sorted(subdict.items(), key=lambda item: item[0]))
    mapped_ys = np.array(list(sorted_subdict.values())).transpose((1,0,2))
    
    print(mapped_ys.shape)
    print("UNIQUE keys", len(np.unique(filtered_keys)))
    mapped_ys = np.reshape(mapped_ys, [original_shape[0],len(filtered_keys), original_shape[2]])
   
    return mapped_ys

def convert_dosage_to_binomial(GT_input):
    GT = np.reshape(GT_input,[GT_input.shape[0],GT_input.shape[1]])
    ds_to_allel_presence_prob = {0:[1,0], 1:[1,1], 2:[0,1], -1: [0,0]}
    results = np.zeros([GT.shape[0],GT.shape[1],2])
    for old, new in ds_to_allel_presence_prob.items():
        results[GT == old] = new
    return results

def flatten_data(x):
    x = np.reshape(x, (x.shape[0],-1))
    return x 

def calculate_alpha(y_true,flip=False):
    ys = convert_dosage_to_binomial(y_true)
    ys = np.reshape(ys, [ys.shape[0], ys.shape[1]*ys.shape[2]])
    #w0=np.mean(1-ys, axis=0)
    w1=np.mean(ys, axis=0)
    #alpha = 1/0.0001+np.min(np.stack([w0,w1]), axis=0)
    alpha1 = np.expand_dims(w1, axis=0)
    if(flip==True):
        alpha1 = 1-alpha1
    #alpha = np.nan_to_num(alpha, nan=0.0001, posinf=0.0001, neginf=0.0001)
    alpha1 = Variable(torch.from_numpy(alpha1).float()).cuda()
    #for float64 precision
    #alpha1 = Variable(torch.from_numpy(alpha1).double()).cuda()
    return alpha1

def get_optimizer(parameters, learning_rate, L2, optimizer_type='adam'):
    if optimizer_type == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=L2)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=L2)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=L2)
    elif optimizer_type == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=learning_rate, weight_decay=L2)
    elif optimizer_type == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=learning_rate, weight_decay=L2)
    else:
        print("oprimizer not supported:", optimizer_type, "setting adam as default")
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=L2)
    
def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to('cuda:0')
    return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

# define the sparse loss function
def sparse_loss(rho, data, model_children):
    encoded_values = model_children[0](data)
    loss = kl_divergence(rho, encoded_values)
    return loss    

# define L1_loss
def l1_loss(model):
    l1_regularization = 0.
    l1_loss = nn.L1Loss()
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_regularization = l1_regularization + l1_loss(param, target=torch.zeros_like(param))
    return l1_regularization

def focal_loss(y_pred, y_true, gamma=3, alpha=None):
    loss=nn.BCELoss(reduction='none')
    cross_entropy_loss = loss(y_pred, y_true)
    p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))

    modulating_factor = 1.0
    if gamma:
        modulating_factor = torch.pow(1.0 - p_t, gamma)

    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))
    
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)
    
    return focal_cross_entropy_loss.mean()
    