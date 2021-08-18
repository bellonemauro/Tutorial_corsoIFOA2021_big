#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial 4: Visualizzazione dati                                         |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

'''
# Dashboard Streamlit

Dipendenze: 
 - Python
 - Streamlit
 - yfinance

'''
# Importo le librerie
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go 
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from scipy.stats import pearsonr

# Scarica i dati da Yahoo Finance usando uno specifico simbolo
# By default scarica tutti i dati disponbili fino alla data odierna
# per farlo usiamo una libreria python yfinance https://github.com/ranaroussi/yfinance
#
# Infine, salva i dati contenuti in un file indicato come filename
# Attenzione: questo è un esempio, non ha controlli di esistenza dei file, 
#             quindi file con lo stesso nome saranno sovrascritti.
#             Di default salva nella stessa cartella dove lo scritp è eseguito.
@st.cache
def load_data( tickerSymbol = 'MFST'):
    # data odierna
    oggi = date.today()

    # scarica i dati
    tickerData = yf.download(tickerSymbol)
    
    # in alternativa possiamo scaricare lo storico tra due date definite
    #tickerData = tickerData.history(period='1d', start='2020-1-1', end=today)

    # impostiamo un nome per il fle da salvare in un CSV dedicato
    filename = './' + tickerSymbol + '_' + str(oggi)
	
    #salviamo i dati
    tickerData.to_csv('./'+filename+'.csv')
    return tickerData
	
# Semplice grafico a linee per la visualizzazione dei prezzi 
#
def plot_lines (dataframe) -> px.line:
    

    fig = px.line(dataframe, y=dataframe.columns[0:4])
 
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    fig.update_layout(
        height=550, 
        xaxis_title="Giorni ",
        yaxis_title="Prezzo azione ($) ",
    )
    return fig



def visualize_box_plot(data) -> go.Figure:

    weeks = 10
    exclude_days = 6
    from_index = len(data) - (weeks * 7) - exclude_days
    to_index = len(data) - exclude_days

    box_plot_data = data[:][from_index:to_index]

    #plt.rcParams['figure.figsize'] = [8, 6]
    fig = go.Figure()
    fig.add_trace(go.Box(y=box_plot_data['Close'], name='Close',
                marker_color = 'indianred'))
    fig.add_trace(go.Box(y=box_plot_data['High'], name = 'High',
                marker_color = 'lightseagreen'))
    
    return fig

def plot_candele(dataframe) -> go.Figure:

    weeks = 10
    exclude_days = 6
    from_index = len(dataframe) - (weeks * 7) - exclude_days
    to_index = len(dataframe) - exclude_days

    dataframe = dataframe[:][from_index:to_index]
    
    fig = go.Figure(data=[go.Candlestick(
                open=dataframe['Open'],
                high=dataframe['High'],
                low=dataframe['Low'],
                close=dataframe['Close'])])
    return fig


def calcola_correlazione_pearson(data_1, data_2):

    # Calcoliamo la correlazione sulle ultime 100 settimane (2 anni), 
    # e fino a 6 giorni precedenti alla fine dei dati
    # 
    # Attenzione, questa funzione non ha controlli di coerenza dei dati,
    # quindi è compito dell'utente assicurarsi che abbiano le stesse lunghezze 
    # e siano acquisiti nello stess giorno
    setimane = 100
    esclusione_giorni = 6
    
    # calcolo l'indice di partenza da estrarre
    from_index = len(data_1) - (setimane * 7) - esclusione_giorni
    # calcolo l'indice di fine da estrarre
    to_index = len(data_1) - esclusione_giorni
    estratto_di_data1 = data_frame_1[:][from_index:to_index]
    
    # calcolo l'indice di partenza da estrarre
    from_index = len(data_2) - (setimane * 7) - esclusione_giorni
    # calcolo l'indice di fine da estrarre
    to_index = len(data_2) - esclusione_giorni
    estratto_di_data2  = data_2[:][from_index:to_index]

    # rimuovo nan se presenti
    nan_array = np.isnan(estratto_di_data1['Close'])
    not_nan_array = ~ nan_array
    estratto_di_data1 = estratto_di_data1['Close'][not_nan_array]
    
    # rimuovo nan se presenti
    nan_array = np.isnan(estratto_di_data2['Close'])
    not_nan_array = ~ nan_array
    estratto_di_data2 = estratto_di_data2['Close'][not_nan_array]

    correlazione = pearsonr(estratto_di_data1, estratto_di_data2)
    return correlazione



# entry point
if __name__ == "__main__": 

    # Titolo
    st.title('Dashboard di visualizzazione dei prezzi delle azioni nel mercato')
    st.markdown(
    """
    # Guida

    Questa applicazione permette di visualizzare in una dashboard dinamica il prezzo di una azione 
    prendendo i dati da finance.yahoo.com usando una API python, yfinance. 

    Inserisci a destra i nomi di due simboli da scarcare e confrontare, guarda i grafici. 

    """
    )

    # Slider sidebar
    st.sidebar.write('Simbolo da caricare')

    # Legge il primo simbolo da scaricare e carica i dati nel dataframe
    simbolo_dati_1 = st.sidebar.text_input("Nome del simbolo", value="NVDA")
    data_frame_1 = load_data(simbolo_dati_1) # load data

    simbolo_dati_2 = st.sidebar.text_input("Nome del simbolo", value="AAPL")
    data_frame_2 = load_data(simbolo_dati_2)

    # i data_frame sono delle strutture dati del seguente tipo: 
    #
    # Date | Open | High | Low | Close |
    # 1922 | prezzo| prezzo| prezzo| prezzo| 
    # 
    # questa struttura dati è in realtà riduttiva ma funziona per il nostro scopo
    # dettagli sul sito ufficale yfinance
    
    # Stampo una descrizione dataframe
    st.write('Descrizione dataframe:')
    st.write(data_frame_1.describe(include='all'))
    #st.write(data_frame_1) # stampa tutti i dati, inutile e time consuming

    st.write('Descrizione dataframe:')
    st.write(data_frame_2.describe(include='all'))
    #st.write(data_frame_2) # stampa tutti i dati, inutile e time consuming

    # Plot a linee
    fig = plot_lines (data_frame_1)
    st.plotly_chart(fig)
    
    # Plot a candele
    fig2 = plot_candele(data_frame_1)
    st.plotly_chart(fig2)

    # Visualizza un box plot
    fig3 = visualize_box_plot(data_frame_1)
    st.plotly_chart(fig3)

    # Plot a linee
    fig4 = plot_lines (data_frame_2)
    st.plotly_chart(fig4)

    # Plot a candele
    fig5 = plot_candele(data_frame_2)
    st.plotly_chart(fig5)
    
    # Visualizza un box plot
    fig6 = visualize_box_plot(data_frame_2)
    st.plotly_chart(fig6)

    

    correlazione = calcola_correlazione_pearson(data_frame_1, data_frame_2)
    st.write('Il coefficiente di correlazione di Pearson è: ', np.round(correlazione,2))



    st.subheader("""
    ---------------------------------------------------------------------
    Software rilasciato su licenza BDS.

    Autore: Mauro Bellone, http://www.maurobellone.com""")
