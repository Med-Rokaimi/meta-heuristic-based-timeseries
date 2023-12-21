import argparse
import torch


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


parser = argparse.ArgumentParser(description=' Time Series Forecasting')


# data loader

parser.add_argument('--results', type=str, default='./results/', help='location of model results')

# forecasting task
parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# model define
parser.add_argument('--n_hidden', type=int, default=4, help='numbers of hidden units')
parser.add_argument('--epoch', type=int, default=1000, help='epoch ')
parser.add_argument('--num_layers', type=int, default=512, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

# optimization
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')#bathsize
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')#lr


parser.add_argument('--features', type=list, default='Energy', help='list of external variables')
parser.add_argument('--model', type=str, default='LSTM', help='NN model')


def args_config():
    args = dotdict()

    return args


Algorithms =['GWO', 'L_SHADE']
Features ={0:'Price',
           10:'sentimen', 20:'Energy', 30:'USD',
           120:['sentimen', 'Energy'], 130: ['sentimen', 'USD'],
           230:['energy', 'USD'],
           1230: ['sentimen', 'Energy', 'USD'],
           }
Models ={1:'LSTM', 2:'Bi-LSTM', 3:['CNN-LSTM'], 4:['CNN-LSTM-att'],
         5:{'encoder-decoder-LSTM'}, 6:['GRU'], 7:['Bi-GRU']}

def set_args():
    args = dotdict()

    #expermint setting:
    args.features= ['Price', 'sentimen']
    args.pred_len = 3
    args.itr = 0


    args.epoch = 2000
    args.batch_size = 32
    args.patience = 30
    args.results = "/results/"
    args.num_layers = 1
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args



