
from keras.metrics import *
from keras.callbacks import *



# evaluate one or more weekly forecasts against expected values
import matplotlib.pyplot as plt
import numpy as np

def model_evaluation(trues, preds):
    preds= np.round(preds,2)
    trues = np.round(trues, 2)
    mae,mse,rmse = metric(trues[:,-1], preds[:,-1])
    return mae,mse,rmse


def plot_loss(history):
     print("plotting")
     plt.plot(history.history['loss'])
     plt.plot(history.history['val_loss'])
     plt.title('model loss')
     plt.ylabel('loss')
     plt.xlabel('epoch')
     plt.legend(['loss', 'val_loss'], loc='upper left')
     plt.show()


def plot_trues_preds(trues, preds, path):
    plt.plot(trues[:,-1])
    plt.plot(preds[:,-1])
    plt.title('tures vs plots')
    plt.legend(['trues', 'preds'], loc='upper left')
    plt.savefig(path, bbox_inches='tight')
    #plt.show()

def MAE(pred, true):
    return np.mean(np.abs(pred - true))
def MSE(pred, true):
    return np.mean((pred - true) ** 2)
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    print(" MAE: {:.4f} MSE , {:.4f} : RMSE , {:.4f}".format(mae, mse ,rmse))
    return mae,mse,rmse



