from keras import optimizers
from _keras.models.keras_lstm import *
from keras.callbacks import EarlyStopping, TerminateOnNaN
tf.random.set_seed(2012)
def run_keras_trian_prediction(data, structure, args ):

    model = initial_keras_model(structure, args)
    optimizer = getattr(optimizers, structure["opt"])(learning_rate=structure["learning_rate"])
    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=args.patience)
    model.fit(data['X_train'], data['y_train'], epochs=args.epoch, batch_size=args.batch_size, verbose=0,
                        validation_data=(data['X_valid'], data['y_valid']), shuffle=False
                        , callbacks=[TerminateOnNaN(), es])
    # plot_loss(history)
    preds = model.predict(data['X_test'], verbose=0)
    preds = preds.reshape(-1, preds.shape[1])
    trues = data['y_test']

    #print(trues.shape, preds.shape)
    return trues, preds

def initial_keras_model(structure , args):
    if args.model=='keras-cnn-lstm':
        model = keras_cnn_lstm( structure , args)
    elif args.model=='CNN-LSTM-att':
        model = CNN_LSTM_att(structure, args)
    elif args.model == 'encoder-decoder-LSTM':
        model = encoder_decoder_LSTM(structure , args)
    elif args.model == 'encoder-decoder-GRU':
        model = encoder_decoder_GRU(structure, args)
    else:
        print("incorrect model name")
        return None

    return model

