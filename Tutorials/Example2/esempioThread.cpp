/*  +---------------------------------------------------------------------------+
*  |                                                                           |
*  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
*  |  Tutorial 1: Thread test                                                  |
*  |                                                                           |
*  |  Autore: Mauro Bellone - http://www.maurobellone.com                      |
*  |  Released under BDS License                                               |
*  +---------------------------------------------------------------------------+ */

// includiamo le variabili necessarie
#include <iostream>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>
#include <ctime>
#include <mutex>

// variabile globale contenente la somma (tutti i thread accedono a questa locazione di memoria)
unsigned long long int g_sum = 0;

// variabile atomica 
std::atomic<unsigned long long int> ag_sum(0);

// somma i numeri nel vettore _da_sommare
void SommaNumeri(const std::vector<int> &_da_sommare, int _start_idx, int _end_idx)
{
    // costruisco un ciclo che iniziando da uno specifico indice, 
	// somma i numeri del vettore fino all'indice finale
	for (int i = _start_idx; i <= _end_idx; i++)
		g_sum += _da_sommare[i];
}

// somma i numeri nel vettore _da_sommare
void SommaNumeriAtomica(const std::vector<int> &_da_sommare, int _start_idx, int _end_idx)
{
    // costruisco un ciclo che iniziando da uno specifico indice, 
	// somma i numeri del vettore fino all'indice finale
	for (int i = _start_idx; i <= _end_idx; i++)
	{	
		ag_sum += _da_sommare[i];
	}
}


// Funzione main
int main()
{
    clock_t begin_program = clock();
    
	std::vector<int> da_sommare;
	const int dimensione = 9000;
	da_sommare.resize(dimensione);
	
	{
		clock_t begin = clock();
		
		for (int i = 0; i < dimensione; i++)
		{
			//da_sommare.push_back(rand());
			da_sommare.at(i) = 3;
		}
		clock_t end = clock();
		
		double elapsed = double(end - begin) / CLOCKS_PER_SEC;
		printf("Il tempo trascorso per popolare il vettore da sommare è %f\n", elapsed);
	}
		
	
    // somma i numeri usando una variabile globale 
	{
		clock_t begin = clock();

		SommaNumeri(da_sommare, 0, dimensione-1);

		clock_t end = clock();

        std::cout << "La somma globale dei numeri è " << g_sum << std::endl; 
		
		double elapsed = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "Il tempo trascorso è " << elapsed << std::endl; 
		g_sum = 0;
	}
	std::cout << "Premi invio per continuare"<<std::endl;
    std::cin.ignore(); 


	// somma i numeri usando i thread
	std::thread t1(SommaNumeri, da_sommare, 0, int(dimensione/3) -1);
	std::thread t2(SommaNumeri, da_sommare, int(dimensione/3), 2*int(dimensione/3)-1);
	std::thread t3(SommaNumeri, da_sommare, 2*int(dimensione/3), dimensione-1);

	{
		clock_t begin = clock();

		t1.join();
		t2.join();
		t3.join();
		
		clock_t end = clock();

        std::cout << "La somma globale parallela dei numeri è " << g_sum << std::endl; 
		
		double elapsed = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "Il tempo trascorso è " << elapsed << std::endl; 
	}
	
	std::cout << "Premi invio per continuare"<<std::endl;
    std::cin.ignore(); 
	
	// somma i numeri usando i thread
	std::thread a1(SommaNumeriAtomica, da_sommare, 0, int(dimensione/3) -1);
	std::thread a2(SommaNumeriAtomica, da_sommare, int(dimensione/3), 2*int(dimensione/3)-1);
	std::thread a3(SommaNumeriAtomica, da_sommare, 2*int(dimensione/3), dimensione-1);
	
	{
		clock_t begin = clock();

		a1.join();
		a2.join();
		a3.join();
		clock_t end = clock();
		
        std::cout << "La somma globale parallela dei numeri è " << ag_sum.load() << std::endl; 
		double elapsed = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "Il tempo trascorso è " << elapsed << std::endl; 
	}
			
	clock_t end_program = clock();
	double elapsed_tot = double(end_program - begin_program) / CLOCKS_PER_SEC;
	std::cout << "Il tempo totale trascorso è "<< elapsed_tot << std::endl;
	
}
