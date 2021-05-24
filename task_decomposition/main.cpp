// g++ main.cpp -o main -fopenmp
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <iomanip>

using namespace std;
const int SIZE = 16;
const int NUM_THREADS = 4;

int main(){
    float A[SIZE], B[SIZE], C[SIZE], D[SIZE], E[SIZE], F[SIZE];
    for(int i = 0; i < SIZE; i++){
        A[i] = i+10;
        B[i] = i+1;
        C[i] = D[i] = E[i] = F[i] = 0;
    }

    omp_set_num_threads(NUM_THREADS);
    # pragma omp parallel sections
    {
        # pragma omp section
        {
            for(int i = 0; i < SIZE; i++)
                C[i] = A[i] + B[i];
        }
        
        # pragma omp section
        {
            for(int i = 0; i < SIZE; i++)
                D[i] = A[i] - B[i];
        }
        
        # pragma omp section
        {
            for(int i = 0; i < SIZE; i++)
                E[i] = A[i] * B[i];
        }
        
        # pragma omp section
        {
            for(int i = 0; i < SIZE; i++)
                F[i] = A[i] / B[i];
        }
    }

    cout << "C: ";
    for(int i = 0; i < SIZE; i++) cout << C[i] << " ";
    cout << endl << "D: ";
    for(int i = 0; i < SIZE; i++) cout << D[i] << " ";
    cout << endl << "E: ";
    for(int i = 0; i < SIZE; i++) cout << E[i] << " ";
    cout << endl << "F: ";
    for(int i = 0; i < SIZE; i++) cout << F[i] << " ";
    return 0;
}