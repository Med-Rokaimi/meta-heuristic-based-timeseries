# This is a sample Python script.
import sys
import time
import warnings

from utils.helper import print_params, generate_loss_value, start_job, show_exp_summary, save_model_history_plotting

warnings.filterwarnings("ignore")
from datetime import datetime as dt
from config.args import  set_args
from mealpy.swarm_based import GWO

from data.data import prepare_datat
from sklearn.preprocessing import LabelEncoder
from utils.evaluation import model_evaluation, plot_trues_preds
from fitness.fitness import decode_solution,  save_to_file, save_to_best_file, save_results
from mealpy.utils import io
#parameters

import torch


global_best = {'mse': 1000}
dataset= {}

fitness_list, best_scores = [], []
keras_models = ['keras-cnn-lstm', 'CNN-LSTM-att', 'encoder-decoder-LSTM',
                'CNN-GRU', 'CNN-GRU-att', 'encoder-decoder-GRU']
torch_models = ['LSTM', 'Bi-LSTM', 'torch-CNN-LSTM', 'Bi-GRU', 'GRU']

#setp expermint run parameters

def fitness_function(solution, i=[0]):
    itr = i[0] = i[0]+ 1
    structure = decode_solution(solution, encod_data)
    data = prepare_datat(structure['seq_len'], args)
    args.seq_len=structure['seq_len']
    problem["dataset"] = data
    #train the model and return the loss value
    fitness, trues , preds = generate_loss_value(structure, data , args , keras_models, torch_models)
    #fit_dic = register_current_result(fitness, structure)
    #fitness_list.append(fit_dic)
    if fitness < global_best['mse']:
        global_best['mse']= fitness
        current_time=  time.time() - start_time
        save_to_best_file(EXCEL_RESULT_PATH, structure, args, fitness, running_info, current_time, job_id, itr, f=False)
        #save_results(trues, preds, result_path)
        #plot_trues_preds(trues, preds, result_path + str(fitness) + ".jpg")
        print_params(structure, itr, args, fitness)
    print('Global best score :' ,global_best['mse'])
    return fitness

if __name__ == '__main__':
    args = set_args()
    if len(sys.argv) <= 1:
        print("error: No model name found, you should pass the model name to the main function"
              " i.e python main.py CNN_LSTM ")
        sys.exit()

    args.model = sys.argv[1]
    iteration, pop_size = 30, 10

    model = GWO.OriginalGWO(epoch=iteration, pop_size=pop_size)
    EXCEL_RESULT_PATH = "./results/best_scores.xlsx"
    job_id, running_info, result_path = start_job(EXCEL_RESULT_PATH, model.__str__(), args.model)


    print("========================================")
    print("start run: ", args.run)
    print(f"algorithm: {model.__str__()} , Model: {args.model} , "
          f" prediction horizon: {args.pred_len} , Features: {args.features}")
    print("=======================================\n")



    itr=0
    encod_data = {}
    # LABEL ENCODER
    OPT_ENCODER = LabelEncoder()

    OPT_ENCODER.fit(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'Adam', 'Adamax'])
    encod_data["OPT_ENCODER"] = OPT_ENCODER

    LB = [0,    0.0001, 1,    0.2, 3,     0, 2]
    UB = [6.99, 0.01,   6.99, 0.5, 29.99, 0.1,     127.99]

    problem = {
        "fit_func": fitness_function,
        "lb": LB,
        "ub": UB,
        "minmax": "min",
        "verbose": True,
        "save_population": False,
        "log_to": "console",
        "dataset": {}
    }


    start_time = time.time()
    model.solve(problem) #

    end_time = time.time() - start_time

    print("run time:", end_time)
    print(f"Best solution: {model.solution[0]}")
    sol = decode_solution(model.solution[0], encod_data)

    show_exp_summary(sol, args, model)
    #add the history
    #save_to_file(fitness_list, result_path, args)
    #best_scores.append(sol)
    ## Save best model to file


    # save model history
    io.save_model(model, result_path + "checkpoints")
    save_model_history_plotting(model, result_path)



    #plot the best results
    '''
    _, mse, _ = model_evaluation(global_best['trues'], global_best['preds'])
    print("MSE of the best results obtained :", mse)
    print("Trues :", global_best['trues'])
    print("Preds :", global_best['preds'])
    plot_trues_preds(global_best['trues'], global_best['preds'], result_path + str(mse) +".jpg")
    save_results(global_best['trues'],global_best['preds'], result_path)
    # load the model from disk
    '''
    save_to_best_file(EXCEL_RESULT_PATH,sol, args, model.solution[1][0], running_info, end_time, job_id, itr, f= False)






