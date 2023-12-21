import os
import pandas as pd
import numpy as np
from datetime import datetime as dt

def decode_solution(solution, encod_data):
    opt_integer = int(solution[0])
    opt = encod_data["OPT_ENCODER"].inverse_transform([opt_integer])[0]
    learning_rate = solution[1]
    n_hidden_units = 2 ** int(solution[2])
    dropout = solution[3]
    seq_len = int(solution[4])
    weight_decay = int(solution[5])
    h2 = int(solution[6])
    return {
        "opt": opt,
        "learning_rate": learning_rate,
        "n_hidden_units": n_hidden_units,
        "dropout": dropout,
        "seq_len": seq_len,
        "weight_decay": weight_decay,
        "h2": h2,
    }
def save_results(trues, preds, PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    np.save(PATH + 'preds.npy', preds)
    np.save(PATH + 'vals.npy', trues)

def save_to_file(records, PATH , args):
        df = pd.DataFrame(records)
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        PATH= PATH + args.model+".csv"
        print(PATH)
        df.to_csv(PATH, index=False)

def save_to_best_file(excel_path, sol, args, score, opt_info, end_time, job_id, itr, f):
        now = dt.now()
        timestamp = now.strftime("%Y_%m_%d__%H_%M_%S")
        df_excel = pd.read_excel(excel_path)
        epoch=0
        result= [job_id, epoch, opt_info, ""+str(itr), f,
                 " ",
                 args.features,
                 score,
                 sol['opt'],
                 sol['n_hidden_units'], sol['learning_rate'], sol['dropout'],
                 args.pred_len, sol['seq_len'], sol['weight_decay'],
                 sol['h2'], args.model,  end_time ,  timestamp]
        #df_excel.loc[job_id] = result
        df_excel = df_excel.drop(df_excel[df_excel['jobID'] == job_id].index)
        df_excel.loc[len(df_excel)] = result
        df_excel.to_excel(excel_path, index=False)


def register_current_result(score, structure):
    fit_dic ={}

    print("fitness = generate_loss_value(structure, data)")
    fit_dic['score'] = score
    fit_dic['opt'] = structure['opt']
    fit_dic['dropout'] = structure['dropout']
    fit_dic['learning_rate'] = structure['learning_rate']
    fit_dic['n_hidden_units'] = structure['n_hidden_units']
    fit_dic['weight_decay'] = structure['weight_decay']
    fit_dic['h2'] = structure['h2']
    return fit_dic



