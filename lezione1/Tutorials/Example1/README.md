# Descrizione Esempio 1

Questo tutorial è pensato semplicemente per produrre un carico computazionale da visualizzare tramite il tool <a href="https://htop.dev/">htop</a>.

ATTENZIONE: il codice è pensato intenzionalmente per saturare la CPU e la memoria quindi può far crashare l'intera macchina, da usare solo nell'ambito del tutorial per motivi di apprendimento. Mai scrivere un codice come quello di questo tutorial nel mondo reale :-).

La procedura da seguire è la seguente: 
1. Aprire 2 terminal diversi 
2. Collegarsi alla macchina remota tramite ssh utente_remoto@ip_utente_remoto e copiare il codice tramite scp -r file_da_copiare cartella_di_destinazione
3. Sul primo terminal eseguire htop 
4. sul secondo terminal eseguire il codice python coreOverload.py
5. monitorare l'utilizzo delle risorse su htop
6. interrompere il codice prima che il pc remoto si saturi


<img src="https://github.com/bellonemauro/Tutorial_corsoIFOA2021_big/blob/main/lezione1/Tutorials/Example1/screen_result.png"  width="1024" height="600" />
