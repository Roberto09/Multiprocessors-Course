/*Compile command:
g++ -o main main.cpp -mavx2
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <x86intrin.h>
// Single Threaded Array Addition FLOATs

int main() {
	int i;
	float* A = NULL;
	float* B = NULL;
	float* C = NULL;
	int result = 1;
	const int elements = 52428800;
	time_t start, end;

	//Array creation
	size_t datasize = sizeof(float) * elements;
	A = (float*)_mm_malloc(datasize, 64); // 64 bit alignment since it works for 64 and 32 bit type sizes
	B = (float*)_mm_malloc(datasize, 64);
	C = (float*)_mm_malloc(datasize, 64);

	//Array initialization (Normally you would get this from a file)
	for (i = 0; i < elements; i++) {
		A[i] = (float) i;
		B[i] = (float) i;
	}

    int v;
    std::cout << "Que version desar usar? \n(1) original \n(2) intrinsicas SSE \n(3) intrinsicas AVX\n";
    std::cin >> v;

    start = clock();
    if(v == 1){
        //This loop can be optimized using Intrinsics
        for (i = 0; i < elements; i++)
            C[i] = A[i] + B[i];
    }
    else if (v == 2){
        __m128 vec_a, vec_b;
        for (i = 0; i < elements>>2; i++){
            vec_a = _mm_load_ps(A+(i<<2));
            vec_b = _mm_load_ps(B+(i<<2));
            vec_a = _mm_add_ps(vec_a, vec_b);
            _mm_store_ps(C+(i<<2), vec_a);
        }
    }
    else{
        __m256 vec_a, vec_b;
        for (i = 0; i < elements>>3; i++){
            vec_a = _mm256_load_ps(A+(i<<3));
            vec_b = _mm256_load_ps(B+(i<<3));
            vec_a = _mm256_add_ps(vec_a, vec_b);
            _mm256_store_ps(C+(i<<3), vec_a);
        }
    }
	end = clock();

	//Validation
	for (i = 0; i < elements; i++) {
		if (C[i] != i + i) {
			result = 0;
			break;
		}
	}

	//Print first 10 results
	for (i = 0; i < 10; i++) {
		printf("C[%d]=%10.2lf\n", i, C[i]);
	}

	if (result) {
		printf("Results verified!!! (%ld)\n", (long)(end - start));
	}
	else {
		printf("Wrong results!!!\n");
	}

	//Memory deallocation
	_mm_free(A);
	_mm_free(B);
	_mm_free(C);

	return 0;
}
