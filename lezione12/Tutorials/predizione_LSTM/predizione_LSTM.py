#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : LSTM                                                         |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 


# importo librerie standard
import numpy as np 
import pandas as pd

# importo librerie di ploting
import matplotlib.pyplot as plt
import seaborn as sns # solo per avere lo sfondo nero sui grafici

import torch
from sklearn.preprocessing import MinMaxScaler # per normalizzare
import math, time
from sklearn.metrics import mean_squared_error

# importiamo il modello che abbiamo definito
from MauroLSTM import *


def split_data(stock, lookback):
    """
    Funzione di data split per rispettare train e test 
    ma anche il parametro lookback che definisce il numero di passi indietro
    """
    data_raw = stock.to_numpy() # converte  in un numpy array
    data = []
    
    # Crea tutte le possibili sequenze di dati della lunghezza pari a "lookback"
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    # di fatto in uscita da questo ciclo abbiamo tutte le sequenze 
    # di lunghezza pari a lookback, data è una matrice che ha numero di 
    # righe uguali al numero di campioni da analizzare -lookback e 
    # e numero di colonne pari a lookback (c'è ridondanza di informazione)

    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]


def plot_training_result (_original_train, _predict_train, _original_test, _predict_test, _vettore_loss):

    plt.style.use('dark_background')    
    #fig = plt.figure()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.subplot(2, 2, 1)
    ax = sns.lineplot(data=_vettore_loss, color='royalblue')
    ax.set_xlabel("Epoch", size = 12)
    ax.set_ylabel("Loss", size = 12)
    ax.set_title("Training Loss", size = 12, fontweight='bold')
    
    full_data = np.squeeze(np.append(_original_train, _original_test, axis=0),axis=1)
    full_data_prediction = np.squeeze(np.append(_predict_train, _predict_test, axis=0), axis=1 )
    plt.subplot(2, 2, 2)
    ax = sns.lineplot( data=full_data, label="Data", color='royalblue')
    ax = sns.lineplot( data=full_data_prediction, label="Training (LSTM)", color='tomato')
    ax.set_title('Prezzo azione', size = 12, fontweight='bold')
    ax.set_xlabel("Giorni", size = 12)
    ax.set_ylabel("Costo (USD)", size = 12)
    ax.set_xticklabels('', size=10)

    plt.subplot(2, 2, 3)
    ax = sns.lineplot(x = _original_train.index, y = _original_train[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = _predict_train.index, y = _predict_train[0], label="Training (LSTM)", color='tomato')
    ax.set_title('Prezzo azione train', size = 12, fontweight='bold')
    ax.set_xlabel("Giorni", size = 12)
    ax.set_ylabel("Costo (USD)", size = 12)
    ax.set_xticklabels('', size=10)
    
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.subplot(2, 2, 4)
    ax = sns.lineplot(x = _original_test.index, y = _original_test[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = _predict_test.index, y = _predict_test[0], label="Test (LSTM)", color='tomato')
    ax.set_title('Prezzo azione test', size = 12, fontweight='bold')
    ax.set_xlabel("Giorni", size = 12)
    ax.set_ylabel("Costo (USD)", size = 12)
    ax.set_xticklabels('', size=10)
    plt.show()
 
# entry point
if __name__ == '__main__':
    # parametri di training
    input_dim = 1
    hidden_dim = 64  # best 64
    num_layers = 2
    output_dim = 1
    max_num_epoche = 50
    # scegliamo la lunghezza della sequenza dei dati da analizzare per ogni predizione, 
    # questo rappresenta il numero di passi indietro 
    lookback = 40 

    # carico dei dati
    filepath = './TSLA-2014-12-27-2019-12-27.csv'
    data = pd.read_csv(filepath, delimiter=',',usecols=['Date','Open','High','Low','Close'])
    #data = data.sort_values('Date',ascending=True)
    # stampiamo qualcosa dei dati a schermo
    print(data.head())

    price = data[['Close']]
    print(price.info())

    # normalizziamo i dati
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

    x_train, y_train, x_test, y_test = split_data(price, lookback)

    # trasformiamo tutti i dati in tensori di pytorch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

    # istanziamo il modello 
    model = MauroLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.007)

    vettore_loss = np.zeros(max_num_epoche)
    start_time = time.time()
    lstm = []

    # ciclo di training
    for t in range(max_num_epoche):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoca ", t, "MSE: ", loss.item())
        vettore_loss[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    training_time = time.time()-start_time
    print("Durata del training: {}".format(training_time))


    # Inferenza 
    y_test_pred = model(x_test)

    # Invertiamo le predizioni per la normalizzazione dello scaler 
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())


    # Calcoliamo l'errore quadratico medio 
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)

    # per plottare tramite dataframe
    original_train = pd.DataFrame(y_train)
    predict_train = pd.DataFrame(y_train_pred)
    original_test = pd.DataFrame(y_test)
    predict_test = pd.DataFrame(y_test_pred)

    plot_training_result(original_train, predict_train, original_test, predict_test, vettore_loss)
