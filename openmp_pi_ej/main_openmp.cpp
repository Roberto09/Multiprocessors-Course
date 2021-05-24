// g++ main_openmp.cpp -o main_openmp -fopenmp
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <iomanip>
using namespace std;

const int THREADS = 8;
double results[THREADS];

long cantidad_intervalos = 10000000;
double base_intervalo, acum = 0;

struct timeval start, eend;

long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}

void piFunc(double id){
    double fdx, x = id * base_intervalo + base_intervalo/2;
    double loc_acum = 0.0;

    for (int i = id * base_intervalo; i < cantidad_intervalos; i+= THREADS){
        fdx = 4 / (1+x*x);
        loc_acum += (fdx * base_intervalo);
        x += base_intervalo*THREADS;
    }
    #pragma omp atomic
        acum += loc_acum;
}

int main(){
    gettimeofday(&start, NULL);
    base_intervalo = 1.0 / cantidad_intervalos;
    
    #pragma omp parallel
    {
        int act_thread = omp_get_thread_num();
        piFunc(double(act_thread));
    }

    gettimeofday(&eend, NULL);
    cout << setprecision(18) << "Resultado = " << acum << " (" << get_millisec(start, eend) << ")" << endl;
    return 0;
}
