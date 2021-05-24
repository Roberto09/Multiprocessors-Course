#include <iostream>
#include <omp.h>

using namespace std;
int main(){
    int N = 16;
    float a[N], b[N], c[N];
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = N-1-i;
        c[i] = 0;
    }

    int threads = 8;
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        int act_thread = omp_get_thread_num();
        int s = int(N*act_thread/threads), e = int(N*(act_thread+1)/threads);
        for(int i = s; i < max(e, N); i++) c[i] = a[i] + b[i];
    }

    for(int i = 0; i < N; i++) cout << c[i] << " ";
    return 0;
}
