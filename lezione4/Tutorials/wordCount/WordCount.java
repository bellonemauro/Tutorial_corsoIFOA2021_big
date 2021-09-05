/*  +---------------------------------------------------------------------------+
*  |                                                                           |
*  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
*  |  Tutorial 3: WordCount con Hadoop commentato in italiano                  |
*  |                                                                           |
*  |  Autore: Mauro Bellone - http://www.maurobellone.com                      |
*  |  Released under BDS License                                               |
*  +---------------------------------------------------------------------------+ */

// importa librerie standard Java
import java.io.IOException;
import java.util.StringTokenizer;

// importa librerie di hadoop
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


// tutto risiede in una unica classe WordCount
public class WordCount {

  
  // classe per la mappatura dei token (elementi nella stringa)
  // eredita la classe mapper di hadoop 
  // vedi: https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/mapreduce/Mapper.html
  public static class MapperSeparatore
       extends Mapper<Object, Text, Text, IntWritable>
  {

    // ogni elemento della lista delle parole avrà un numero intero e la specifica parola da contare
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    // metodo mapping 
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException 
    {
      // classe standard di java per spezzare una frase in parole (token separati dal carattere spazio bianco)
      StringTokenizer itr = new StringTokenizer(value.toString()); 
      
      // iteriamo sulla stringa 
      while (itr.hasMoreTokens()) 
      {
        // finchè ci sono stringhe 
        word.set(itr.nextToken());

        context.write(word, one);
      }
    }    

  }

  // classe di combinazione e riduzione
  // in questo caso il combinatore e il riduttore sono la stessa funzione
  // eredita la classe reducer di hadoop 
  // vedi https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/mapreduce/Reducer.html
  public static class ReducerContatoreIntero
       extends Reducer<Text,IntWritable,Text,IntWritable> 
  {
    // risultato somma del numero delle parole
    private IntWritable result = new IntWritable();


    // metodo reduce
    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException 
    {
      // somma delle parole
      int sum = 0;
      for (IntWritable val : values) 
      {
        sum += val.get();
      }
      result.set(sum);
      
      // è interessante notare che per ogni chiave si va a sommare il risultato 
      context.write(key, result);
    }

  }

  // entry point
  public static void main(String[] args) throws Exception 
  {
    System.out.println(" DEMO CONTEGGIO PAROLE - IFOA2021");
    // oggetto di configurazione org.apache.hadoop.conf.Configuration;
    Configuration conf = new Configuration();

    // oggetto per istanziare i lavori org.apache.hadoop.mapreduce.Job; 
    // vedi https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/mapreduce/Job.html
    Job job = Job.getInstance(conf, "word count");

    // setta le configurazioni del job da eseguire
    job.setJarByClass(WordCount.class);

    // mapper
    job.setMapperClass(MapperSeparatore.class);
    // l'implementazione del metodo processa una linea alla volta, 
    // e per ogni linea crea una mappa del tipo: 
    // <parola, 1>
    // <parola_2, 1>
    // ...
    // <parola_n, 1>

    // nella mappa si indica semplicemente la parola e il numero, di fatto non stiamo ancora contando, 
    // supponiamo di dover mappare 2 stringhe, 
    // stringa 1 =  "ciao mamma ciao " 
    // stringa 2 =  "ciao mondo" 
    // 
    // le mappe risultante sarebbero:
    // 
    // MAPPA 1: 
    //
    // <ciao, 1>
    // <mamma, 1>
    // <ciao, 1>
    
    // MAPPA 2: 
    //
    // <ciao, 1>
    // <mondo, 1>


    // combiner
    job.setCombinerClass(ReducerContatoreIntero.class);
    // l'uscita del combinatore semplicemente somma le parole per ogni mappa avendo una uscita del tipo: 
    // <parola, n>     // dove n è il numero di occorrenze della parola 1 nella specifica mappa
    // <parola_2, m>
    // ...
    // <parola_n, nn>
    //
    // ritornando all'esempio precedente con le mappe il risultato del combinatore sarà:
    // 
    // MAPPA 1: 
    //
    // <ciao, 2>
    // <mamma, 1>
    
    // MAPPA 2: 
    //
    // <ciao, 1>
    // <mondo, 1>

    
    // reducer
    job.setReducerClass(ReducerContatoreIntero.class);
    // a questo punto avremo due mappe, già combinate da ridurre
    // l'uscita sarùà una unica mappa del tipo 
    // <parola, n>     // dove n è il numero di occorrenze della parola 1 nella specifica mappa
    // <parola_2, m>
    // ...
    // <parola_n, nn>
    // 
    // con l'esempio precedente otteniamo: 
    // OUTPUT : 
    //
    // <ciao, 3>
    // <mamma, 1>
    // <mondo, 1>
    
    // output --- semplicemente salva l'output come un set <key, value>
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);

    // settaggi input e output sul HDFS - 
    // vedi: https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/mapreduce/lib/output/FileOutputFormat.html
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    // fa partire un thread che attende che il job sia completo
    System.out.println("job.getName : " + job.getJobName());
    System.out.println("Siamo pronti per lanciare il job sul cluster, premi invio");
    System.in.read();

    // linea standard
    System.exit(job.waitForCompletion(true) ? 0 : 1);  // true sta sull'attributo verbose
    
    /*
    // sottomettiamo il job 
    job.submit(); 
    // e aspettiamo che sia eseguito
    while (!job.isSuccessful())
    {
      // facciamo qualcosa di utile che non blocchi la CPU :-) es. printare info di stato
      
      // considerate che mentre il job è eseguito usando il principio di data locality,
      // quindi di fatto stiamo eseguendo il job nel container dove i dati sono contenuti. 
      // 
      // al contrario, 
      // questo ciclo è eseguito sul namenode, quindi quello che possiamo fare è semplicemente 
      // inviare delle informazioni aggiuntive di monitoraggio dello stato del job 
      // (se diverso da quanto possiamo fare in http://IP:porta)

    }
    
    System.out.println("Ho finito il job invia una sveglia al mio programmatore");

    System.exit(0);*/
    
  }
}