// g++ main_threads.cpp -o main_threads -pthread
#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <unistd.h>
#include <sys/time.h>

using namespace std;

const int THREADS = 8;
pthread_mutex_t lock;

long cantidad_intervalos = 10000000;
double base_intervalo, acum = 0;

struct timeval start, eend;

long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}

void *piFunc(void *id_p){
    long tid = (long)id_p;
    double id = (double)tid;
    long cantidad_intervalos_thread = cantidad_intervalos / THREADS;
    double fdx, x = cantidad_intervalos_thread * id * base_intervalo;
    double loc_acum = 0.0;

    for (int i = 0; i < cantidad_intervalos_thread; i++){
        fdx = 4 / (1+x*x);
        loc_acum += (fdx * base_intervalo);
        x += base_intervalo;
    }
    pthread_mutex_lock(&lock);
    acum += loc_acum;
    pthread_mutex_unlock(&lock);
}

int main(){
    pthread_t threads[THREADS];
    pthread_attr_t attr;

    if (pthread_mutex_init(&lock, NULL)) exit(-1);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    gettimeofday(&start, NULL);
    base_intervalo = 1.0 / cantidad_intervalos;
    int rc;
    for(int i = 0; i < THREADS; i++){
        rc = pthread_create(&threads[i], &attr, piFunc, (void*)i );
        if(rc) exit(-1);
    }

    pthread_attr_destroy(&attr);
    void *status;
    for(int i = 0; i < THREADS; i++){
        rc = pthread_join(threads[i], &status);
        if(rc) exit(-1);
    }
    gettimeofday(&eend, NULL);
    pthread_mutex_destroy(&lock);
    cout << "Resultado = " << acum << " (" << get_millisec(start, eend) << ")" << endl;
    return 0;
}
