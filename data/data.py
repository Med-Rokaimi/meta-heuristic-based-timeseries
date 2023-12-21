import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
ss = StandardScaler()


def getData():
    DATAPATH = 'dataset/oil.xlsx'
    df = pd.read_excel(open(DATAPATH, 'rb'), index_col=0, parse_dates=True)
    #drop the first 7 rows and obtain only 2380 to facilitate the split rate calculation
    df = df[6:]
    #print(f"dataset shape: ({df.shape[0]} * {df.shape[1]}), samples * features")
    return df
def prepare_datat(seq_len, args):
    # read the data from the excel file
    df = getData()
    # select only two features
    df = df[args.features]
    args.feature_no = len(df.columns)
    # 4- X and y
    X, y = df, df.Price.values
    # 5- Normalize
    X_trans, y_trans = normalize__my_data(X, y)

    # 7- Build the sequence
    X_ss, y_mm = split_sequences(X_trans, y_trans, seq_len, args.pred_len)

    # num_of_samples = 2386
    train_size = 2070
    train_test_cutoff = train_size
    vald_size = 250
    test_size = 60
    # understand_data_values_for_split(num_of_samples, X, y)
    data = split_train_test_pred(X_ss, y_mm, train_test_cutoff, vald_size, test_size)
    return data
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def split_train_test_pred (X_ss, y_mm , train_test_cutoff, vald_size, predict_size):
    X_train = X_ss[:train_test_cutoff]
    X_valid = X_ss[train_test_cutoff: train_test_cutoff + vald_size]

    y_train = y_mm[:train_test_cutoff]
    y_valid = y_mm[train_test_cutoff: train_test_cutoff + vald_size]

    X_test = X_ss[-predict_size:]
    y_test = y_mm[-predict_size:]

    #print("Training Shape:", X_train.shape, y_train.shape)
    #print("Validation  Shape:", X_valid.shape, y_valid.shape)
    #print("Test Shape:", X_test.shape, y_test.shape)
    data = {"X_train": X_train, "y_train": y_train, "X_valid": X_valid, "y_valid": y_valid, "X_test": X_test,
            "y_test": y_test}
    #print("Data - X_train", X_train.shape)
    return data

def normalize__my_data(X, y):
  X_trans = ss.fit_transform(X)
  #X_trans = x_scaler.fit_transform(X)
  y_trans = mm.fit_transform(y.reshape(-1, 1))
  return X_trans, y_trans

def denormolize_data(trues, preds):
    return mm.inverse_transform(trues), mm.inverse_transform(preds)
def understand_data_values_for_split(num_of_samples, X_, y_ ):

  #print(X_.shape, y_.shape)
  #Training set
  #print("Training set")
  X_train_=X_.loc[:'01-01-2020']
  y_train_ = y_.loc[:'01-01-2020']
  #print(X_train_.shape , X_train_.index.min(), X_train_.index.max())
  #print(y_train_.shape , y_train_.index.min(), y_train_.index.max())
  #Validation set
  #print("Validation set")
  X_valid_=X_.loc['01-01-2020':'12-31-2020']
  y_valid_=y_.loc['01-01-2020':'12-31-2020']
  #print(X_valid_.shape , X_valid_.index.min(), X_valid_.index.max())
  #print(y_valid_.shape , y_valid_.index.min(), y_valid_.index.max())

  #Predic set
  #print("Prediction set")
  X_test_=X_.loc['12-31-2020':]
  y_test_=y_.loc['12-31-2020':]
  #print(X_test_.shape , X_test_.index.min(), X_test_.index.max())
  #print(y_test_.shape , y_test_.index.min(), y_test_.index.max())
  darw_splitted_date(num_of_samples , X_train_ , X_valid_, X_test_ )
  return X_train_, X_valid_, X_test_ ,y_test_, y_train_ , y_valid_ , y_valid_

def darw_splitted_date(num_of_samples , X_train_ , X_valid_, X_test_):
  plt.figure(figsize=(6,3))
  figure, axes = plt.subplots()
  axes.plot(X_train_["Price"][-num_of_samples:], color="blue")
  axes.plot(
      X_valid_["Price"][-num_of_samples - 259 : ],
      color="red",
      alpha=0.5,
  )
  axes.plot(
      X_test_["Price"][-num_of_samples - 259  : ],
      color="green",
      alpha=0.5,
  )
  plt.show()

def add_price_change(df):
  price_change_column="Price_changes"
  cols= df.columns
  if price_change_column in cols:
    print("already added")
  else:
    df_chng= df['Price'].pct_change()
    df['Price_changes'] = df_chng
    print(df['Price_changes'].isna().sum())
    df['Price_changes'] = df['Price_changes'].replace(np.nan, 0)
  return df


from logging import raiseExceptions
# add the time/date feature to the dataset
import datetime
def add_date_features(df , date_column):
  cols=df.columns
  if date_column not in cols:
    print("the date column is unkown")
    return df
  else:
    #df_enc['dayofmonth'] = df_enc['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    #df_enc['week'] = df_enc['date'].dt.isocalendar().week
    df['season'] = (df['date'].dt.month -1)//3
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df[0:2]
    # Perform one-hot encoding of categorical date features
    encoded_data = pd.get_dummies(df, columns=['dayofweek', 'season', 'month', 'year'])
    # Display the encoded data
    return encoded_data
def variates_selection(df, date_feature_enabled, change_price):
  data__=[]
  data__=df.copy()
  if(change_price):
    data__ = add_price_change(data__)

  if(date_feature_enabled):
    data__= data__.reset_index()
    data__= add_date_features(data__, 'date')
    data__=data__.set_index('date')
  else:
    data__=data__.copy()
  return data__
