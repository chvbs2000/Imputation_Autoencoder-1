import numpy as np
import torch
from torch import nn
import importlib.util


# class pretrained_Autoencoder(nn.Module):
#     """
#     nn.Linear Parameters
#     input_dim – size of each input sample
#     output_dim – size of each output sample
#     n_layers - total number of layers
#     size_ratio=1.0 keep full layer size for encoding hidden layers 
#     """
    
#     def __init__(self,input_dim, output_dim, n_layers=4, size_ratio=0.7, activation='relu'):
#         super(pretrained_Autoencoder, self).__init__()

#         def get_activation(activation):

#             if(activation=='relu'):
#                 return nn.ReLU(True)
#             elif(activation=='tanh'):
#                 return nn.Tanh()
#             elif(activation=='sigmoid'):
#                 return nn.Sigmoid()
#             elif(activation=='leakyrelu'):
#                 return torch.nn.LeakyReLU()

#         encoder_layers = []        
#         in_size_list = [input_dim]
#         out_size_list = [output_dim]
                
#         for i in range(int(n_layers/2)):
#             out_size_list += [int(out_size_list[i]*size_ratio)]
#             encoder_layers += [nn.Linear(in_size_list[i], out_size_list[i+1])]
#             encoder_layers += [get_activation(activation)]
#             in_size_list += [out_size_list[i+1]]
            
#         decoder_layers = []
#         out_size_list.reverse()
        
#         for i in range(int(n_layers/2)-1):
#             decoder_layers += [nn.Linear(out_size_list[i], out_size_list[i+1])]
#             decoder_layers += [get_activation(activation)]
            
#         decoder_layers += [nn.Linear(out_size_list[-2], output_dim)]  
#         decoder_layers += [get_activation('sigmoid')]
        
#         self.encoder = nn.Sequential(*encoder_layers)
#         self.decoder = nn.Sequential(*decoder_layers)
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#        return x  

class FFNN_Autoencoder(nn.Module):
    """
    nn.Linear Parameters
    input_dim – size of each input sample
    output_dim – size of each output sample
    n_layers - total number of layers
    size_ratio=1.0 keep full layer size for encoding hidden layers 
    
    model architecture:
    
        [Sequential(
      (0): Linear(in_features=419, out_features=586, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=586, out_features=410, bias=True)
      (3): LeakyReLU(negative_slope=0.01)
      (4): Linear(in_features=410, out_features=287, bias=True)
      (5): LeakyReLU(negative_slope=0.01)
      (6): Linear(in_features=287, out_features=200, bias=True)
      (7): LeakyReLU(negative_slope=0.01)
    ), Sequential(
      (0): Linear(in_features=200, out_features=287, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=287, out_features=410, bias=True)
      (3): LeakyReLU(negative_slope=0.01)
      (4): Linear(in_features=410, out_features=586, bias=True)
      (5): LeakyReLU(negative_slope=0.01)
      (6): Linear(in_features=586, out_features=838, bias=True)
      (7): Sigmoid()
    )]
    """
    
    def __init__(self,input_dim, output_dim, n_layers=8, size_ratio=0.7, activation='relu', pretrain_path=('./','./')):
        super(FFNN_Autoencoder, self).__init__()

        def get_activation(activation):

            if(activation=='relu'):
                return nn.ReLU(True)
            elif(activation=='tanh'):
                return nn.Tanh()
            elif(activation=='sigmoid'):
                return nn.Sigmoid()
            elif(activation=='leakyrelu'):
                return torch.nn.LeakyReLU()

        encoder_layers = []        
        in_size_list = [input_dim]
        out_size_list = [output_dim]
        
        # encoder 
        for i in range(int(n_layers/2)):
            out_size_list += [int(out_size_list[i]*size_ratio)]
            encoder_layers += [nn.Linear(in_size_list[i], out_size_list[i+1])]
            encoder_layers += [get_activation(activation)]
            in_size_list += [out_size_list[i+1]]
            
        decoder_layers = []
        out_size_list.reverse()
        
        # decoder 
#         for i in range(int(n_layers/2)-1):
#             decoder_layers += [nn.Linear(out_size_list[i], out_size_list[i+1])]
#             decoder_layers += [get_activation(activation)]
            
#         decoder_layers += [nn.Linear(out_size_list[-2], output_dim)]  
#         decoder_layers += [get_activation('sigmoid')]
        
        ##### load pre-trained model #####
        meta_path , param_path = pretrain_path
        spec = importlib.util.spec_from_file_location("my_model_name" + "_param", param_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded_model = pretrained_Autoencoder(input_dim=input_dim, output_dim=input_dim, n_layers=8, size_ratio=0.7, activation='relu').cuda()
        loaded_model.load_state_dict(torch.load(meta_path))
        decoder_layers = list(loaded_model.children())[1]
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = decoder_layers
        
        # freeze decoder weight 
        params = self.decoder.parameters()
        for param in params:
            param.requires_grad = False
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    