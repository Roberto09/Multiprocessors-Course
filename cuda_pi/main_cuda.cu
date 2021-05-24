#include <stdio.h>
#include <sys/time.h>

long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}

long cantidadIntervalos = 1000000000;
long ttl_threads = cantidadIntervalos;
int blockSize = 256;

struct timeval start, eend;

double baseIntervalo = 1.0 / cantidadIntervalos;

__global__ void calc_pi(double *tmp_storage, long cantidadIntervalos, long ttl_threads, double baseIntervalo){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= ttl_threads) return; // no computation needed in this one
    
    int stride = blockDim.x * gridDim.x;
    double loc_acum=0, fdx, x;
    
    for (long i = index; i < cantidadIntervalos; i+=stride) {
        x = (i+0.5)*baseIntervalo;
        fdx = 4 / (1 + x * x);
        loc_acum += fdx;
    }
    loc_acum *= baseIntervalo;
    tmp_storage[index] = loc_acum;
}


int main() {
    gettimeofday(&start, NULL);

    int size = ttl_threads * sizeof(double);
    double* h_tmp_storage = (double*)malloc(size), d_tmp_storage;
    cudaMalloc((void**)&d_tmp_storage, size);

    memset(h_tmp_storage, 0.0, size);
    cudaMemcpy(d_tmp_storage, h_tmp_storage, size, cudaMemcpyHostToDevice);

    int numberBlocks = (ttl_threads + blockSize - 1) / blockSize;
    calc_pi << <numberBlocks, blockSize> >> (d_tmp_storage, cantidadIntervalos, ttl_threads, baseIntervalo);
    
	cudaDeviceSynchronize();

    cudaMemcpy(h_tmp_storage, d_tmp_storage, size, cudaMemcpyDeviceToHost);
    
    double acum = 0;
    for(int i = 0; i < ttl_threads; i++) acum += h_tmp_storage[i];

    gettimeofday(&eend, NULL);
    printf("Result = %20.18lf (%ld)\n", acum, get_millisec(start, eend));
    return 0;
}