// g++ main.cpp -o main
#include <stdio.h>
#include <sys/time.h>

long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}

long cantidadIntervalos = 1000000000;
double baseIntervalo;
double fdx;
double acum = 0;
struct timeval start, eend;

int main() {
   double x=0;
   long i;
   baseIntervalo = 1.0 / cantidadIntervalos;
   gettimeofday(&start, NULL);
   for (i = 0; i < cantidadIntervalos; i++) {
      x = (i+0.5)*baseIntervalo;
      fdx = 4 / (1 + x * x);
      acum += fdx;
   }
   acum *= baseIntervalo;
   gettimeofday(&eend, NULL);
   printf("Result = %20.18lf (%ld)\n", acum, get_millisec(start, eend));
   return 0;
}