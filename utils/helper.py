from _keras.trainer import run_keras_trian_prediction
from data.data import denormolize_data
from pytorch.trainer import run_pytorch_trian_prediction
from utils.evaluation import model_evaluation
import pandas as pd
from datetime import datetime as dt

def generate_loss_value(structure, data , args,  keras_models, torch_models):

    if args.model in keras_models:
        trues, preds = run_keras_trian_prediction(data, structure, args)
    elif args.model in torch_models:
        trues, preds= run_pytorch_trian_prediction(data, structure, args)
    else:
        print('error: incorrect model name. Available models:')
        print(keras_models)
        print(torch_models)
        exit()
    #3- evaluation models with denormoloized values
    trues, preds = denormolize_data(trues, preds)
    #save_results(trues, preds, result_path)
    _, mse, _ = model_evaluation(trues, preds)
    return mse, trues, preds


def start_job(EXCEL_RESULT_PATH, algorithm, model):

    df_excel = pd.read_excel(EXCEL_RESULT_PATH)
    last_job = df_excel.iloc[-1, 0]
    new_job= last_job +1

    print(f'job {new_job} : has been started')
    opt_info = str(new_job) + '_' + algorithm + '_' + model

    now = dt.now()
    timestr = now.strftime("%Y_%m_%d__%H_%M_%S")
    result_path = "./results/" + str(new_job) + "_"  + algorithm + timestr + '_' + model + "/"

    return  new_job, opt_info , result_path

def show_exp_summary(sol, args, model):
    print(f"Opt: {sol['opt']},"
          f"Network: {args.model} ,"
          f"Learning-rate: {sol['learning_rate']}, "
          f"dropout: {sol['dropout']}, "
          f"timesteps: {sol['seq_len']}, "
          f"n-hidden: {sol['n_hidden_units']} ,"
          f"n-h2: {sol['h2']} ,"
          f"weight_decay: {sol['weight_decay']}")

    print("get_parameters")
    print(model.get_parameters())
    print(model.get_name())
    print(model.problem.get_name())
    print(model.get_attributes()["solution"])

def save_model_history_plotting(model, result_path):
    print (model)
    ## You can access them all via object "history" like this:
    model.history.save_global_objectives_chart(filename=result_path + "global_objectives_chart")
    model.history.save_local_objectives_chart(filename=result_path + "local_objectives_chart/loc")
    model.history.save_global_best_fitness_chart(filename=result_path + "global_best_fitness_chart/gbfc")
    model.history.save_runtime_chart(filename=result_path + "runtime_chart/rtc")
    model.history.save_exploration_exploitation_chart(filename=result_path + "xploration_exploitation_chart/eec")
    model.history.save_diversity_chart(filename=result_path + "diversity_chart/dc")

#print the papramaters of the current solution
def print_params(structure, i, args, mse):
    print(f"best score updated: MSE: ({mse})")
    print(f"itr {i} ; MSE: ({mse}), Paramaters: timesteps: {structure['seq_len']}, preddiction :{args.pred_len}, model:{args.model},"
          f" optimiser:{structure['opt']}, h:{structure['n_hidden_units']}, lr{structure['learning_rate']},"
          f"droupout: {structure['dropout']}, weight_decay: {structure['weight_decay']} , h2: {structure['h2']}")
    print("----\n")


