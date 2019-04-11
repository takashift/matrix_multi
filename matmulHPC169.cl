__attribute__((reqd_work_group_size(1,1,1)))
__kernel void matMul(__global double *restrict a, __global const double *restrict b, __global const double *restrict c, const int N) {
    int i, j, k, iter
    unsigned int iter2;
    double sum;

    // for (i=0; i<N; i++){
    // #pragma unroll
    //     for (iter=0; iter<N*N; iter++) {
    //         j = iter / N; k = iter % N;
    //         if(k == 0)
    //             sum = 0.0;

    //         sum += b[i*N+k]*c[k*N+j];

    //         if(k == N-1)
    //             a[i*N+j] = sum ;
    //     }
    // }

    for (iter2=0; iter2<(unsigned int)N*N*N; iter2++){
        i = iter2 / (N * N); iter = iter2 % (N * N);
        j = iter / N; k = iter % N;
        if(k == 0)
            sum = 0.0;

        sum += b[i*N+k]*c[k*N+j];

        if(k == N-1)
            a[i*N+j] = sum ;
    }
}



