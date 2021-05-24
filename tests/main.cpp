#include <iostream>
#include <x86intrin.h>

using namespace std;

int main(){
    __m128 vec_a, vec_b;
    // float arr[4] __attribute__((aligned(16)));
    // ;_mm_malloc(size, size_alignment);
    // _mm_free(ptr);
    float arr[4] = {1.0, 2.0, 3.0, 4.0};
    float arr2[4] = {2.0, 9.0, 2.0, 1.0};
    vec_a = _mm_loadu_ps(arr);
    vec_b = _mm_loadu_ps(arr2);
    // vec_a = _mm_set_ps(arr[0], arr[1], arr[2], arr[3]);
    vec_a = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(arr2, vec_a);

    for(int i = 0; i < 4; i++) cout << arr2[i] << " ";
    cout << endl;
}   