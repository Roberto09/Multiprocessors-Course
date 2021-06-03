#include <iostream>
#include <string.h>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <x86intrin.h>
#include <omp.h>
#include <iomanip> 

const int OMP_TS = 8;

/* --------------------------- general code ------------------------------*/
long get_millisec(timeval &s, timeval &e){
    long seconds = e.tv_sec - s.tv_sec; //seconds
    long useconds = e.tv_usec - s.tv_usec; //milliseconds
    return ((seconds) * 1000 + useconds/1000.0);
}
struct timeval start, eend;

using namespace std;

const string MAT_A_N = "A.txt";
const string MAT_B_N = "B.txt";

bool comparar(double* mat_a, double* mat_b, int r, int c){
    for(int i = 0; i < r*c; i++){
        if(mat_a[i] != mat_b[i]) return false;
    }
    return true;
}

void output(double* mat, int r, int c, string filename){
    ofstream arch;
    arch.open(filename);
    for(int i = 0; i < r; i ++){
        for(int j = 0; j < c; j++) arch << fixed << std::setprecision(10) << mat[i*c + j] << "\n";
    }
    arch.close();
}

int read_mat(double* mat, int r, int c, bool transp, string matn){
    fstream newfile;
    newfile.open(matn,ios::in);
    if (newfile.is_open()){
        string tp;
        double act;
        int act_r, act_c, i = 0;
        while(getline(newfile, tp)){
            if(i >= r*c) {
                newfile.close();
                return -2;
            }
            act_r = i/c;
            act_c = i%c;
            if (!transp) mat[act_r*c + act_c] = stod(tp);
            else mat[act_c*r + act_r] = stod(tp);
            i++;
        }
        if(i != r*c) {
            newfile.close();
            return -2;
        }
        newfile.close();
    }
    else return -1;
    return 0;
}

int get_matrices(double* mat_a, int ra, int ca, double* mat_b, int rb, int cb){
    int res_a = read_mat(mat_a, ra, ca, false, MAT_A_N);
    int res_b = read_mat(mat_b, rb, cb, true, MAT_B_N);
    if(res_a < 0 || res_b < 0){
        if(res_a == -1) cout << "Archivo matriz A no existe" << endl;
        else if(res_a == -2) cout << "Archivo matriz A no tiene el tama;o indicado" << endl;

        if(res_b == -1) cout << "Archivo matriz B no existe" << endl;
        else if(res_b == -2) cout << "Archivo matriz B no tiene el tama;o indicado" << endl;
        return -1;
    }
    return 0;
}

void print_mat(double* mat, int r, int c){
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            cout << mat[i*c + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
/* --------------------------- general code ------------------------------*/


/* --------------------------- regular code ------------------------------*/
void mult_matrx(double* mat, int r, int c, int mat_a_c, double* mat_a, double* mat_b){
    memset(mat, 0, sizeof(double)*r*c);
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            for(int k = 0; k < mat_a_c; k++){
                mat[i*c + j] += mat_a[i * mat_a_c + k] * mat_b[j *mat_a_c + k];
            }
        }
    }
}

pair<int, double*> regular(int ra, int ca, int rb, int cb, bool output_archivo, bool ret_prod){
    gettimeofday(&start, NULL);

    // TODO check allocation is correct
    double* mat_a = new double[ra*ca];
    double* mat_b = new double[rb*cb];

    int res_get_matrices = get_matrices(mat_a, ra, ca, mat_b, rb, cb);
    if(res_get_matrices != 0){
        delete[] mat_a;
        delete[] mat_b;
        return {-1, NULL};
    }

    double* prod = new double[ra*cb];
    mult_matrx(prod, ra, cb, ca, mat_a, mat_b);

    gettimeofday(&eend, NULL);
    printf("Regular tardo: %ld ms\n", get_millisec(start, eend));

    // print_mat(prod, ra, cb);
    if(output_archivo) output(prod, ra, cb, "matrixC.txt");
    delete[] mat_a;
    delete[] mat_b;
    if(ret_prod)
        return {0, prod};
    delete[] prod;
    return {0, NULL};
}
/* --------------------------- regular code ------------------------------*/

/* --------------------------- intrinsics code ------------------------------*/
// __m256 vec_a, vec_b;
// for (i = 0; i < elements>>3; i++){
//     vec_a = _mm256_load_ps(A+(i<<3));
//     vec_b = _mm256_load_ps(B+(i<<3));
//     vec_a = _mm256_add_ps(vec_a, vec_b);
//     _mm256_store_ps(C+(i<<3), vec_a);
// }

// void mult_matrx(double* mat, int r, int c, int mat_a_c, double* mat_a, double* mat_b){
//     memset(mat, 0, sizeof(double)*r*c);
//     for(int i = 0; i < r; i++){
//         for(int j = 0; j < c; j++){
//             for(int k = 0; k < mat_a_c/2; k++){
//                 __m128 vec_a, vec_b, res;
//                 vec_a = _mm128_load_pd(mat_a);
//                 vec_b = _mm128_load_pd(mat_b);
//                 res = _mm_dp_pd(vec_a, vec_b, );
//                 mat[i*c + j] += mat_a[i * mat_a_c + k] * mat_b[j *mat_a_c + k];
//             }
//         }
//     }
// }

// int regular(int ra, int ca, int rb, int cb){
//     gettimeofday(&start, NULL);

//     // TODO check allocation is correct
//     double* mat_a = (double*)_mm_malloc(sizeof(double) * ra*ca, 32);
//     double* mat_b = (double*)_mm_malloc(sizeof(double) * rb*cb, 32);

//     int res_get_matrices = get_matrices(mat_a, ra, ca, mat_b, rb, cb);
//     if(res_get_matrices != 0) return -1;

//     double* prod = (double*)_mm_malloc(sizeof(double) * ra*cb, 32);
//     mult_matrx_intr(prod, ra, cb, ca, mat_a, mat_b);

//     gettimeofday(&eend, NULL);
//     printf("Regular tardo: %ld ms\n", get_millisec(start, eend));

//     _mm_free(mat_a);
//     _mm_free(mat_b);
//     _mm_free(prod);
//     return 0;
// }

/* --------------------------- intrinsics code ------------------------------*/


/* --------------------------- openmp code ------------------------------*/

void mult_matrx_omp(double* mat, int r, int c, int mat_a_c, double* mat_a, double* mat_b){
    memset(mat, 0, sizeof(double)*r*c);
    omp_set_num_threads(OMP_TS);
    #pragma omp parallel
    {
        int num_thread = omp_get_thread_num();
        int i, j;
        for(int p = num_thread; p < r*c; p+=OMP_TS){
            i = p/c;
            j = p%c;
            for(int k = 0; k < mat_a_c; k++){
                mat[i*c + j] += mat_a[i * mat_a_c + k] * mat_b[j *mat_a_c + k];
            }
        }
    }
}

pair<int, double*> openmp(int ra, int ca, int rb, int cb, bool ret_prod){
    gettimeofday(&start, NULL);

    // TODO check allocation is correct
    double* mat_a = new double[ra*ca];
    double* mat_b = new double[rb*cb];

    int res_get_matrices = get_matrices(mat_a, ra, ca, mat_b, rb, cb);
    if(res_get_matrices != 0){
        delete[] mat_a;
        delete[] mat_b;
        return {0, NULL};
    }

    double* prod = new double[ra*cb];
    mult_matrx_omp(prod, ra, cb, ca, mat_a, mat_b);

    gettimeofday(&eend, NULL);
    printf("Openmp tardo: %ld ms\n", get_millisec(start, eend));

    // print_mat(prod, ra, cb);
    // output(prod, ra, cb, "matrixD.txt");
    delete[] mat_a;
    delete[] mat_b;
    if(ret_prod)
        return {0, prod};
    delete[] prod;
    return {0, NULL};
}
/* --------------------------- openmp code ------------------------------*/

int main(){
    int ra, ca, rb, cb;
    cout << "Dime la cantidad de filas de A: ";
    cin >> ra;
    cout << "Dime la cantidad de columnas de A: ";
    cin >> ca;

    cout << "Dime la cantidad de filas de B: ";
    cin >> rb;
    cout << "Dime la cantidad de columnas de B: ";
    cin >> cb;

    if(ca != rb){
        cout << "Las matrices no son multiplicables" << endl;
        return 0;
    }

    double* reg_m;
    double* otra_m;

    // Correr regular
    cout << "Corriendo regular..." << endl;
    for(int i = 0; i < 5; i++){
        pair<int, double*> res = regular(ra, ca, rb, cb, i==4, i==4);
        if(i==4) reg_m=res.second;
        if(res.first != 0){
            if(i==4) delete[] reg_m;
            return 0;
        }
    }
    cout << endl;

    // Correr omp
    cout << "Corriendo omp..." << endl;
    for(int i = 0; i < 5; i++){
        pair<int, double*> res = openmp(ra, ca, rb, cb, i==4);
        if(i==4) otra_m=res.second;
        if(res.first != 0){
            delete[] reg_m;
            if(i == 4) delete[] otra_m;
            return 0;
        }
    }
    cout << endl;

    // Comparar regular vs omp
    if(comparar(reg_m, otra_m, ra, cb))
        cout << "*** matriz regular es igual a la de omp ***" << endl << endl;
    else{
        cout << "*** matriz regular no es igual a la de omp ***" << endl << endl;
        delete[] reg_m;
        delete[] otra_m;
        return 0;
    }

    delete[] reg_m;
    delete[] otra_m;

    cout << endl;
    return 0;
}