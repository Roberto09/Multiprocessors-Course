#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#define SIZE 1024
int main() {
    int A[SIZE] __attribute__((aligned(64)));
    int B[SIZE] __attribute__((aligned(64)));
    for (int i = 0; i < SIZE; i++)
        A[i] = B[i] = i;
        
    // This loop will be auto-vectorized
    asm("nop");
    for (int i = 0; i < SIZE; i++)
        A[i] = A[i] + B[i];
    asm("nop");
    
    for (int i = 0; i < 1024; i++)
        printf("%2d %2d %2d\n", i, A[i], B[i]);
    return 0;
} 