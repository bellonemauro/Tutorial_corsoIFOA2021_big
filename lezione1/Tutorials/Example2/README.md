# Descrizione Esempio 2

Questo tutorial è pensato dimostrare come più thread possono generare un conflitto sulla scrittura della memoria. 
Il codice è scritto in C++11.  
L'obiettivo di apprendimento è di capire come e quando alcune operazioni parallelizzabili possono generare un conflitto in memoria e il concetto di atomicità che rende la parallelizzazione conflict-free. 
Tuttavia, tutto ciò ha un costo, la parallelizzazione in questo caso comporta un tempo computazionale più alto. 

La procedura da seguire è la seguente: 
1. Aprire 1 terminal  
2. Collegarsi alla macchina remota tramite ssh utente_remoto@ip_utente_remoto e copiare il codice tramite scp -r file_da_copiare cartella_di_destinazione
3. Entrare nella cartella di destinazione ed eseguire ``` mkdir ./build && cmake ./build/  ``` 
4. Entrare nella cartella di build ``` cd build ```
5. Compilare il codice ``` make ```
6. Eseguire più volte il codice ``` ./esempioThread ```


<img src="https://github.com/bellonemauro/Tutorial_corsoIFOA2021_big/blob/main/lezione1/Tutorials/Example2/screen_result.png" />
