__attribute__((reqd_work_group_size(1,1,1)))
__kernel void addVector(__global float *a, __global const float *b, __global const float *c, __global int *N) {
    int i, j, k, iter;
    for (i=0; i<N; i++){
    #pragma unroll
        for (iter=0; iter<N*N; iter++) {
            j = iter / N; k = iter % N;
            if(k == 0)
                float sum = 0.0;

            sum += b[i*N+k]*c[k*N+j];

            if(k == N-1)
                a[i*N+j] = sum ;
        }
    }
}