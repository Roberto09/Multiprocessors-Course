// g++ main.cpp -o main -fopenmp -pthread
#include <iostream>
#include <vector>
#include <omp.h>
#include <pthread.h>
#include <cstdlib>
#include <unistd.h>
#include <iomanip>
#include <sys/time.h>

using namespace std;
// ------------------------ global variables -----------------------
const int THREADS = 8;
double results[THREADS];
long cantidad_intervalos = 10000000;
double base_intervalo, acum = 0;
struct timeval start, eend;
pthread_mutex_t lock;


// ------------------------ auxiliar functions ---------------------
double avg(vector<long> &vec){
    double res = 0;
    for(long e :vec) res += e;
    return res/vec.size();
}
long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}
void reset(){
    base_intervalo = 1.0 / cantidad_intervalos;
    acum = 0;
    for(int i = 0; i < THREADS; i++) results[i] = 0;
}
void printAcum(long time, string method){
    cout << setprecision(10) << "Resultado (" << method << ") = " << acum << " (" << time << ")" << endl;
}


// ----------------------- regular code ----------------------------
long runPiFuncRegular(){
    gettimeofday(&start, NULL);
    double fdx, x = base_intervalo/2;

    for (int i = 0; i < cantidad_intervalos; i++){
        fdx = 4 / (1+x*x);
        acum += (fdx * base_intervalo);
        x += base_intervalo;
    }
    gettimeofday(&eend, NULL);
    return get_millisec(start, eend);
}


// ----------------------- multithreaded code ----------------------------
void *piFuncMulti(void *id_p){
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

long runPiFuncMulti(){
    pthread_t threads[THREADS];
    pthread_attr_t attr;

    if (pthread_mutex_init(&lock, NULL)) exit(-1);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    gettimeofday(&start, NULL);
    base_intervalo = 1.0 / cantidad_intervalos;
    int rc;
    for(int i = 0; i < THREADS; i++){
        rc = pthread_create(&threads[i], &attr, piFuncMulti, (void*)i );
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
    return get_millisec(start, eend);
}


// ----------------------- openmp code ----------------------------
void piFuncOpen(double id){
    long cantidad_intervalos_thread = cantidad_intervalos / THREADS;
    double fdx, x = cantidad_intervalos_thread * id * base_intervalo;
    double loc_acum = 0.0;

    for (int i = 0; i < cantidad_intervalos_thread; i++){
        fdx = 4 / (1+x*x);
        loc_acum += (fdx * base_intervalo);
        x += base_intervalo;
    }
    results[int(id)] = loc_acum;
}

long runPiFuncOpen(){
    gettimeofday(&start, NULL);
    base_intervalo = 1.0 / cantidad_intervalos;
    
    #pragma omp parallel
    {
        int act_thread = omp_get_thread_num();
        piFuncOpen(double(act_thread));
    }
    for(int i = 0; i < THREADS; i++) acum += results[i];

    gettimeofday(&eend, NULL);
    return get_millisec(start, eend);
}


// ------------------- run everything ----------------------------
int main(){
    vector<long> regular, multi, open;
    for(int i = 0; i < 5; i++){
        regular.push_back(runPiFuncRegular());
        printAcum(regular.back(), "regular");
        reset();
    }
    cout << "average regular: " << avg(regular) << endl << endl;
    for(int i = 0; i < 5; i++){
        multi.push_back(runPiFuncMulti());
        printAcum(multi.back(), "multi threaded");
        reset();
    }
    cout << "average multi threaded: " << avg(multi) << endl << endl;
    for(int i = 0; i<5; i++){
        open.push_back(runPiFuncOpen());
        printAcum(open.back(), "openmp");
        reset();
    }
    cout << "average openmp: " << avg(open) << endl << endl;
    return 0;
}