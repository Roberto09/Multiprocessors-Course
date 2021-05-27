// nvcc main_cuda.cu -o main_cuda
#include <stdio.h>
#include <sys/time.h>

long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}
struct timeval start, eend;

#define STEPS 2000000000
#define BLOCKS 256
#define THREADS 256

int threadidx;

// Kernel
__global__ void pi_calculation(float* pi, int nsteps, float base, int nthreads, int nblocks)
{
    int i;
    float x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate index for each thread
    float acum = 0;
    for (i = idx; i < nsteps; i += nthreads * nblocks)
    {
        x = (i + 0.5) * base;
        acum += 4.0 / (1.0 + x * x); //Save result to device memory
    }
    atomicAdd(pi, acum);
}

int main(void)
{
    gettimeofday(&start, NULL);
    dim3 dimGrid(BLOCKS, 1, 1); // Grid dimensions
    dim3 dimBlock(THREADS, 1, 1); // Block dimensions
    float base = 1.0 / STEPS; // base size

    // Launch Kernel
    int xd = 4;
    float pi = 0;
    pi_calculation << <dimGrid, dimBlock >> > (&pi, STEPS, base, THREADS, BLOCKS);

    // Sync
    cudaDeviceSynchronize();

    // Multiply by base
    pi *= base;

    // Output Results
    gettimeofday(&eend, NULL);
    printf("Result = %20.18lf (%ld)\n", pi, get_millisec(start, eend));

    return 0;
}