#include <stdio.h>

long cantidadIntervalos = 1000000000;
long ttl_threads = 256*16;
int blockSize = 256;

float baseIntervalo = 1.0 / cantidadIntervalos;

__global__ void calc_pi(float *tmp_storage, long cantidadIntervalos, long ttl_threads, float baseIntervalo){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= ttl_threads) return; // no computation needed in this one
    
    int stride = blockDim.x * gridDim.x;
    float loc_acum=0, fdx, x;
    
    for (long i = index; i < cantidadIntervalos; i+=stride) {
        x = (i+0.5)*baseIntervalo;
        fdx = 4 / (1 + x * x);
        loc_acum += fdx;
    }
    loc_acum *= baseIntervalo;
    tmp_storage[index] = loc_acum;
}


int main() {

    int size = ttl_threads * sizeof(float);
    float* h_tmp_storage = (float*)malloc(size);
    float* d_tmp_storage;
    cudaMalloc((void**)&d_tmp_storage, size);

    memset(h_tmp_storage, 0.0, size);
    cudaMemcpy(d_tmp_storage, h_tmp_storage, size, cudaMemcpyHostToDevice);

    int numberBlocks = (ttl_threads + blockSize - 1) / blockSize;
    calc_pi <<<numberBlocks, blockSize>>> (d_tmp_storage, cantidadIntervalos, ttl_threads, baseIntervalo);
    
	cudaDeviceSynchronize();

    cudaMemcpy(h_tmp_storage, d_tmp_storage, size, cudaMemcpyDeviceToHost);
    
    float acum = 0;
    for(int i = 0; i < ttl_threads; i++) acum += h_tmp_storage[i];

    printf("Result = %20.18lf\n", acum);
    return 0;
}