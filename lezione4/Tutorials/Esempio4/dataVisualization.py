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

# Create a data folder in your current dir.
def SaveData(df, filename):
	df.to_csv('./'+filename+'.csv')

@st.cache
def load_data( tickerSymbol = 'MFST'):
    today = date.today()
    #get data on this ticker
    tickerData = yf.download(tickerSymbol)
    #data = pd.read_csv(path)
    #tickerDf = tickerData.history(period='1d', start='2020-1-1', end=today)
    dataname= './'+tickerSymbol+'_'+str(today)
	#files.append(dataname)
    SaveData(tickerData, dataname)
    return tickerData
	
  


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

if __name__ == "__main__": 
    # Titolo
    st.title('Dashboard di visualizzazione dei prezzi delle azioni nel mercato')

    # Slider sidebar
    st.sidebar.write('Simbolo da caricare')

    data_symbol = st.sidebar.text_input("Nome del simbolo", value="NVDA")
    myData = load_data(data_symbol) # load data

    # Descrizione dataframe
    st.write('Descrizione dataframe:')
    st.write(myData.describe(include='all'))
    st.write(myData)

    # Plot a linee
    fig = plot_lines (myData)
    st.plotly_chart(fig)

    fig2 = plot_candele(myData)
    st.plotly_chart(fig2)
    
    fig3 = visualize_box_plot(myData)
    st.plotly_chart(fig3)

    st.markdown(
    """

    # Guida

    Questa applicazione permette di visualizzare in una dashboard dinamica il prezzo di una azione 
    prendendo i dati da finance.yahoo.com. 

    """
    )

    st.subheader("""
    ---------------------------------------------------------------------
    Software rilasciato su licenza BDS.

    Autore: Mauro Bellone, http://www.maurobellone.com""")
