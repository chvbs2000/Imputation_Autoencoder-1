import os
import allel
import numpy as np
import pandas as pd
import time
import sys
import datetime
import argparse
import importlib.util
from simulation import simulator
import collections

#parallel processing libraries
# import multiprocessing as mp
# from functools import partial # pool.map with multiple args
# import subprocess as sp
from util import *
from model import *

#DL libraries
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

print("Pytorch version is:", torch.__version__)
print("Cuda version is:", torch.version.cuda)
print("cuDNN version is :", torch.backends.cudnn.version())
print("Arch version is :", torch._C._cuda_getArchFlags())

#test torch.float64 later (impact in accuracy and runtime)
torch.set_default_dtype(torch.float32)

def main(args):
    # set params
    learning_rate = args.learn_rate
    beta = args.beta
    rho = args.rho
    L1 = args.l1
    L2 = args.l2
    gamma = args.gamma
    loss_type = args.loss_type
    disable_alpha = args.disable_alpha
    decay_rate = args.decay_rate
    size_ratio = args.size_ratio
    optimizer_type = args.optimizer
    act = args.activation
    n_layers = args.n_layers
    batch_size = args.batch_size
    start = 0 
    avg_loss = np.inf 
    tmp_loss = 0
    max_epochs=args.max_epochs
    min_MAF = args.min_MAF
    n_cycle = 500
    pretrain_pth = args.pretrain_pth
    pretrain_params = args.pretrain_params
    patience = args.patience
    pretrain_model_dir = args.pretrain_model_dir
    
    # load data
    vcf_path = "/home/kchen/low_pass/autoencoder_imputation/HRC.r1-1.EGA.GRCh37.chr22.haplotypes.35863936-35891148.vcf.VMV1.gz"
    tabix_path = "/home/kchen/bin/tabix"
    #vcf_data = allel.read_vcf(vcf_path, tabix=tabix_path, alt_number=2)
    sample_ids, data, maf, variant_pos = load_vcf(vcf_path, tabix_path)
    
    # filter variants based on minimum MAF if any
    if min_MAF > 0:
        filtered_data = filter_by_MAF(data, maf)
    else:
        filtered_data = data
    
    train_x = data
    train_y = filtered_data
    
    ni = train_x.shape[1]*2
    no = train_y.shape[1]*2
    
    
    n = len(data)
    print("n_input: ", ni)
    print("n_output: ", no)
    
    # define model 
    # pretrain_model_dir = "/home/kchen/low_pass/autoencoder_imputation/IMPUTATOR_HRC.r1-1.EGA.GRCh37.chr22.haplotypes.35863936-35891148.vcf.VMV1"
    model_pth = pretrain_model_dir + '/' + pretrain_pth
    model_params = pretrain_model_dir + '/' + pretrain_params
    pretrain_path = (model_pth, model_params)
    
    model = FFNN_Autoencoder(input_dim=ni, output_dim=no, n_layers=n_layers, size_ratio=size_ratio, activation=act, pretrain_path=pretrain_path).cuda()
    
    # model = FFNN_Autoencoder(input_dim=ni, output_dim=no, n_layers=n_layers, size_ratio=size_ratio, activation=act).cuda()
    #print(model)
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model.parameters(), learning_rate, L2, optimizer_type=optimizer_type)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
    current_lr =  my_lr_scheduler.get_last_lr()
    print("Current learning rate: {}".format(str(current_lr[0])))
    
    # get the layers as a list to apply sparsity penalty later
    model_children = list(model.children()) 
    print(model_children)
    if(args.model_dir == 'auto'):
        model_dir="./IMPUTATOR" + "_" + os.path.basename(args.input)
    else:
        model_dir=args.model_dir

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + '/' + args.model_id + '.pth'
    print("Model save path:", model_path)

    hp_path = model_dir + '/' + args.model_id + '_param.py'

    # TRAINING RESUME FEATURE ADDED: 
    # loads weights from previously trained model. 
    # Note that the current model must have the same architecture as the previous one.
    write_mode = 'w'
    if(args.resume and (not os.path.exists(model_path) or not os.path.exists(hp_path))):
        print("WARNING: model path doesn't exist:", model_path + " (and/or its respective *_param.py)", "\nYou set --resume=True but there is no model to resume from. Make sure you provided the correct path. The model will be trained from scratch.")
    if(args.resume and os.path.exists(model_path) and os.path.exists(hp_path)):
        print("Resume mode activated (--resume) and found pre-existing model weights at", model_path, "\nLoading weights")
        model.load_state_dict(torch.load(model_path))
        spec = importlib.util.spec_from_file_location(args.model_id+'_param', hp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if "last_epoch" in dir(module):
            start = module.last_epoch
        else:
            print("Previous epoch value not found... Keeping 0 value.")
        if "avg_loss" in dir(module):
            avg_loss = module.avg_loss
        else:
            print("Previous losss value not found... Keeping Inf value.")
        if 'early_stop' in dir(module):
            early_stop = module.early_stop
        else:
            early_stop = 0
            
        write_mode ='a'
        if(early_stop > 0):
            print("Cancelling training because early stop was already reached at epoch", early_stop)
            sys.exit()
        elif(start == max_epochs):
            print("Cancelling training because max_epochs was already reached at previous run", max_epochs)            
            sys.exit()            
        else:
            print("Resuming training from epoch", start)
        if(decay_rate > 0):
            current_lr = learning_rate
            for i in range(int(start / n_cycle)):
                current_lr = current_lr*decay_rate
            optimizer = get_optimizer(model.parameters(), current_lr, L2, optimizer_type=optimizer_type)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
            current_lr =  my_lr_scheduler.get_last_lr()
            print("Resuming training from learning rate:", current_lr[0])

    with open(hp_path, write_mode) as param_file:
        param_file.write("n_layers = " + str(n_layers) + "\n")
        param_file.write("size_ratio = " + str(size_ratio) + "\n")
        param_file.write("activation = \'" + act + "\'" + "\n")
        if write_mode == 'w':
            param_file.write("early_stop = 0\n")
    print("New inference parameters saved at:", hp_path)
          
#     print("shape train (input):", train_x.shape)
#     print("shape train (output):", train_y.shape)
    
    total_batch = int((n/batch_size)-0.5)
    #train_y = np.transpose(train_y, (1,0,2))
    train_y = convert_dosage_to_binomial(train_y.copy())
    
    # for focal loss
    if disable_alpha:
        alphas = None
    else:
        alphas = calculate_alpha(filtered_data)
    print("length of variant postions: ", len(variant_pos))

    ys = flatten_data(train_y.copy())
    pos_gt_dict={}
    
    #####
    # pos_gt_dict is used to store all sample dosage per variant. 
    # This will be used as high pass dosage label data for low pass allele presence likelihood simulation
    # example for pos_gt_dict:
    #
    #    {'variant position 1':[sample_1 dosage, sample_2 dosage, ...., sample_n dosage], 
    #     'variant position 2':[sample_1 dosage, sample_2 dosage, ...., sample_n dosage], ...}
    #
    ###
    
    for i,e in enumerate(variant_pos):
        pos_gt_dict[e] = train_x[:,i,:].flatten()
        #pos_gt_dict[e] = train_x[i,:,:].flatten() #before fixed on 01/04/2022
        
    #np.save("var_genotype_dict.npy", pos_gt_dict)
#     from pprint import pprint
#     pprint(pos_gt_dict)
    
    # load simulation model
    highpass_arr = np.array(list(pos_gt_dict.values()))
    lowpass_simulator = simulator()
    
    startTime = time.time()
    i=0   
    trigger_times = 0
    
    # start training
    for epoch in range(start,max_epochs):
        epochStart = time.time()
        epoch_loss=0
        cpuStop = time.time()
        cpuTime = (cpuStop-epochStart)
        comTime = 0        
        gpuTime = 0
        # simulation
        simulated_data = lowpass_simulator.simulate_allele(highpass_arr)
#         print(simulated_data.shape)
        xs = np.transpose(simulated_data, (1, 0, 2))
        xs = flatten_data(xs)
#         print("xs shape: ", xs.shape)
#         print("ys shape: ", ys.shape)
#         print(xs)
#         print(ys)

        for batch_i in range(total_batch):
            
            comStart = time.time()
            
            #prepare data
            batch_data = [xs[batch_i*batch_size:(batch_i+1)*batch_size], 
                              ys[batch_i*batch_size:(batch_i+1)*batch_size]]
            
            #for float32
            masked_data = Variable(torch.from_numpy(batch_data[0]).float()).cuda()
            true_data = Variable(torch.from_numpy(batch_data[1]).float()).cuda()

            gpuStart = time.time()
            comTime += (gpuStart-comStart)

            #forward propagation
            reconstructed = model(masked_data)
#             print(reconstructed)
#             print(true_data)
           
            #focal loss
            if loss_type=='FL':
                loss = focal_loss(reconstructed, true_data, gamma=gamma, alpha=alphas)
            else:
                #CE (log loss)
                loss = criterion(reconstructed, true_data)
            
            #if applying L1 regularizaton
            if L1 > 0:
                l1_sparsity = l1_loss(model)
                loss = loss + l1_sparsity
            
            #if applying KL divergence regularization
            if beta > 0:
                kl_sparsity = sparse_loss(rho, true_data, model_children)
                loss = loss + kl_sparsity
                
            #backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            gpuTime += (time.time()-gpuStart)
            epoch_loss+=loss.data
       
        tmp_loss += epoch_loss/total_batch
        epochTime = (time.time()-epochStart)
                
        print('epoch [{}/{}], epoch time:{:.4f}, CPU-task time:{:.4f}, GPU-task time:{:.4f}, CPU-GPU-communication time:{:.4f}, loss:{:.4f}'.format(epoch + 1, max_epochs,epochTime, cpuTime, gpuTime, comTime, epoch_loss/total_batch), flush=True)
                
        if avg_loss > tmp_loss:
            print("Loss improved from", avg_loss, "to", tmp_loss)
            avg_loss = tmp_loss
            tmp_loss = 0
            print("Saving model to", model_path)
            torch.save(model.state_dict(), model_path)
            with open(hp_path, 'a') as param_file:
                param_file.write("last_epoch = "+str(epoch+1)+"\n")
                param_file.write("avg_loss = "+str(avg_loss)+"\n")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stoping, no improvement observed. Previous loss:", avg_loss, "Currentloss:", tmp_loss)
                print("Best model at", model_path)
                with open(hp_path, 'a') as param_file:
                    param_file.write("early_stop = "+str(epoch+1)+"\n")
                break
        #exponentially decrease learning rate in each checkpoint
        if(decay_rate>0):
            my_lr_scheduler.step()
            current_lr =  my_lr_scheduler.get_last_lr()
            print("Current learning rate:",current_lr[0])

    executionTime = (time.time() - startTime)

    print('Execution time in seconds: ' + str(executionTime))
    print('Run time per epoch: ' + str(executionTime/epoch))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input", type=str, help="[str] Input file (ground truth) in VCF format")
    parser.add_argument("-L", "--l1", type=float, help="[float] L1 regularization scaling factor", default=1e-9)
    parser.add_argument("-W", "--l2", type=float, help="[float] L2 regularization scaling factor (a.k.a. weight decay)", default=1e-5)
    parser.add_argument("-B", "--beta", type=float, help="[float] Beta scaling factor for sparsity loss (KL divergence)", default=0.)
    parser.add_argument("-R", "--rho", type=float, help="[float] Rho desired mean activation for sparsity loss (KL divergence)", default=0.05)
    parser.add_argument("-G", "--gamma", type=float, help="[float] gamma modulating factor for focal loss", default=0.)
    parser.add_argument("-A", "--disable_alpha", type=int, help="[0 or 1]=[false or true] whether disable alpha scaling factor for focal loss", default=0)
    parser.add_argument("-C", "--learn_rate", type=float, help="[float] learning rate", default=0.001)
    parser.add_argument("-F", "--activation", type=str, help="[relu, leakyrelu, tanh, sigmoid] activation function type", default='relu')
    parser.add_argument("-O", "--optimizer", type=str, help="[adam, sgd, adadelta, adagrad] optimizer type", default='adam')
    parser.add_argument("-T", "--loss_type", type=str, help="[CE or FL] whether use CE for binary cross entropy or FL for focal loss", default='CE')
    parser.add_argument("-D", "--n_layers", type=int, help="[int, even number] total number of hidden layers", default=4)
    parser.add_argument("-S", "--size_ratio", type=float, help="[float(0-1]] size ratio for successive layer shrink (current layer size = previous layer size * size_ratio)", default=0.7)
    parser.add_argument("-E", "--decay_rate", type=float, help="[float[0-1]] learning rate decay ratio (0 = decay deabled)", default=0.)
    parser.add_argument("-H", "--model_id", type=str, help="[int/str] model id or name to use for saving the model", default='best_model')
    parser.add_argument("-J", "--model_dir", type=str, help="[str] path/directory to save the model", default='auto')
    parser.add_argument("-Z", "--batch_size", type=int, help="[int] batch size", default=256)
    parser.add_argument("-X", "--max_epochs", type=int, help="[int] maximum number of epochs if early stop criterion is not reached", default=20000)
    parser.add_argument("-U", "--resume", type=int, help="[0 or 1]=[false or true] whether enable resume mode: recover saved model (<model_id>.pth file) in the model folder and resume training from its saved weights.", default=0)
    parser.add_argument("-K", "--min_MAF", type=float, help="set minimum allele frequency filter criteria (default = 0.0)", default=0.0)
    parser.add_argument("-P", "--pretrain_pth", type=str, help="pretrain model pth", default="model_346665_F.pth")
    parser.add_argument("-P2", "--pretrain_params", type=str, help="pretrain model params", default="model_346665_F_param.py")
    parser.add_argument("-p", "--patience", type=int, help="[int] epoch to wait before early stopping", default=10)
    parser.add_argument("-PM", "--pretrain_model_dir", type=str, help="[str] path/directory to the pretained models", default='./')
    
    args = parser.parse_args()
    main(args)