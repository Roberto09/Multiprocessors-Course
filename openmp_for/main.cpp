// g++ main_regular.cpp -o main_regular
#include <iostream>
#include <sys/time.h>
using namespace std;

long cantidad_intervalos = 10000000;
double base_intervalo, acum = 0;

struct timeval start, eend;

long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}

double piFunc(){
    // # pragma omp parallel
    // {
    //     double fdx, x = 0;
    //     # pragma omp for reduction(+:acum)
    //     for (int i = 0; i < cantidad_intervalos; i++){
    //         x = (i+0.5) * base_intervalo;
    //         fdx = 4 / (1+x*x);
    //         acum += fdx;
    //     }
    // }
    
    double fdx, x = 0;
    # pragma omp parallel for private(x, fdx) reduction(+:acum)
    for (int i = 0; i < cantidad_intervalos; i++){
        x = (i+0.5) * base_intervalo;
        fdx = 4 / (1+x*x);
        acum += fdx;
    }
    

    return acum * base_intervalo;
}

int main(){
    gettimeofday(&start, NULL);
    base_intervalo = 1.0 / cantidad_intervalos;
    acum = piFunc();
    gettimeofday(&eend, NULL);
    cout << "Resultado = " << acum << " (" << get_millisec(start, eend) << ")" << endl;
    return 0;
}