# Tutorial word_count commentato in italiano

Per eseguire questo tutorial hadoop deve essere correttamente installato e configurato sul sistema, 

## Procedura:

Eseguiamo il file system di Hadoop e il relative resource manager

 	./sbin/start_dfs.sh
  	./sbin/start_yarn.sh

Compiliamo l’esempio di word count in codice 

	hadoop com.sun.tools.javac.Main WordCount.java
	jar cf wc.jar ./WordCount*.class
	hadoop jar wc.jar WordCount --/PERCORSO_AI_MIEI_FILE_input/ --/PERCORSO_PER_SALVARE_IL_RISULTATO_output

Esaminiamo l’output

  	hdfs dfs -get output output
  	cat output/*

oppure 

  	hdfs dfs -cat output/*
