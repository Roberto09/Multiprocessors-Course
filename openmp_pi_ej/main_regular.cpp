// g++ main_regular.cpp -o main_regular
#include <iostream>
#include <sys/time.h>
#include <iomanip> 
using namespace std;

long cantidad_intervalos = 1000;
double base_intervalo, acum = 0;

struct timeval start, eend;

long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}

double piFunc(){
    double fdx, x = base_intervalo/2;
    double loc_acum = 0.0;

    for (int i = 0; i < cantidad_intervalos; i++){
        fdx = 4 / (1+x*x);
        loc_acum += (fdx * base_intervalo);
        x += base_intervalo;
    }
    return loc_acum;
}

int main(){
    gettimeofday(&start, NULL);
    base_intervalo = 1.0 / cantidad_intervalos;
    acum = piFunc();
    gettimeofday(&eend, NULL);
    cout << setprecision(18) << "Resultado = " << acum << " (" << get_millisec(start, eend) << ")" << endl;
    return 0;
}