__attribute__((reqd_work_group_size(1,1,1)))
__kernel void matMul(__global double *restrict a, __global const double *restrict b, __global const double *restrict c, const int N) {


  int i, j, k ;

    for (i=0; i<N; i++){
      for (j=0; j<N; j++)
        {
	  double sum = 0.0 ;
	  for (k=0; k<N; k++) {
	    sum += b[i*N+k]*c[k*N+j] ;
      }
	  a[i*N+j] = sum ;
        }
    }
}
