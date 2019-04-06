#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif // __APPLE__

// #define MAX_BINARY_SIZE (0x100000)

static const int NumElements = 10000;
static const int MaxDevices  = 10;
static const int MaxLogSize  = 5000;

static void printError(const cl_int err);

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}

int main() {
    double elapsed_time;
    int i;
    cl_int status;
    cl_uint num_devices = 1;
//   aocl_utils::scoped_array<cl_device_id> device_id;

//   // Query the available OpenCL device.
//   device_id.reset(aocl_utils::getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
//   std::cout << "Platform: " << aocl_utils::getPlatformName(platform).c_str() << std::endl;
//   std::cout << "Using " << num_devices << " device(s)" << std::endl;
//   std::cout << " " << aocl_utils::getDeviceName(device_id[0]).c_str() << std::endl;

    // 64バイトにアライメントしないとWARNINGが出る
    double* B; // = (double *)aligned_alloc(64, NumElements * NumElements * sizeof(double));
    double* C; // = (double *)aligned_alloc(64, NumElements * NumElements * sizeof(double));
    double* A; // = (double *)aligned_alloc(64, NumElements * NumElements * sizeof(double));

    posix_memalign((void **)&B, 64, NumElements * NumElements * sizeof(double));
    posix_memalign((void **)&C, 64, NumElements * NumElements * sizeof(double));
    posix_memalign((void **)&A, 64, NumElements * NumElements * sizeof(double));

    // 1.コンテキストの作成
    cl_platform_id myp;
    status=clGetPlatformIDs(1, &myp, NULL);

    if(status != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs failed.\n");
        return 1;
    }

    cl_device_id device_id;
    status=clGetDeviceIDs(myp, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);

    if(status != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs failed.\n");
        return 1;
    }

    cl_context context;
    context = clCreateContext(NULL, num_devices, &device_id, NULL, NULL, &status);
    // clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL, NULL, NULL, &status);
    
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clCreateContextFromType failed.\n");
        return 1;
    }
    
    // 2.コンテキストに含まれるデバイスを取得
    cl_device_id devices[MaxDevices];
    size_t size_return;

    status = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &size_return);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clGetContextInfo failed.\n");
        return 2;
    }

    // 3.コマンドキューの作成
    cl_command_queue queue;
    queue = clCreateCommandQueue(context, devices[0], 0, &status);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue failed.\n");
        return 3;
    }

    // 4.カーネルを含むオブジェクトファイルの読み込み
    // FILE *fp;
    // char filename[] = "./sample_kernel.aocx";
    // size_t binary_size;
    // char *binary_buf;
    // cl_int binary_status;

    // fp = fopen(filename, "r");
    // if(!fp) {
    //     fprintf(stderr, "Faild to load kernel.\n");
    //     exit(1);
    // }
    // binary_buf = (char *)malloc(MAX_BINARY_SIZE);
    // binary_size = fread(binary_buf, 1, MAX_BINARY_SIZE, fp);
    // fclose(fp);

    size_t lengths[1];
    unsigned char* binaries[1] ={NULL};
    cl_int binary_status[1];

    FILE *fp = fopen("matmulHPC169.aocx","rb");
    fseek(fp,0,SEEK_END);
    lengths[0] = ftell(fp);
    binaries[0] = (unsigned char*)malloc(sizeof(unsigned char)*lengths[0]);
    rewind(fp);
    fread(binaries[0],lengths[0],1,fp);
    fclose(fp);

    // 5.プログラムオブジェクトの作成
    cl_program program;
    // program = clCreateProgramWithBinary(context, 1, devices, (const size_t *)&binary_size, (const unsigned char **)&binary_buf, &binary_status, &status);
    program = clCreateProgramWithBinary(context, 1, devices, lengths, (const unsigned char **)binaries, binary_status, &status);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithBinary failed.\n");
        return 5;
    }

    // 本には不要と書いてあったが、Intel のプログラミング・ガイドでは書くようにとあったので、一応追加
    const char options[] = "";
    clBuildProgram(program, 1, devices, options, NULL, NULL);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram failed.\n");
        return 5;
    }

    // 6.カーネルの作成
    cl_kernel kernel;
    kernel = clCreateKernel(program, "matMul", &status);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel failed.\n");
        return 6;
    }

    // 7.メモリオブジェクトの作成
    // for(i=0; i<NumElements; i++) {
    //     // データをセット
    //     B[i] = (float)i * 100.0f;
    //     C[i] = (float)i / 100.0f;
    //     A[i] = 0.0f;
    // }
    for (i = 0; i < NumElements * NumElements; i++)
    {
        A[i] = (double)0.0F;
    }
    for (i = 0; i < NumElements * NumElements; i++)
    {
        B[i] = (double)i;
    }
    for (i = 0; i < NumElements * NumElements; i++)
    {
        C[i] = (double)1.0F;
    }


    cl_mem memB;
    memB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * NumElements * NumElements, B, &status);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer for menB failed.\n");
        return 7;
    }

    cl_mem memC;
    memC = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * NumElements * NumElements, C, &status);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer for menC failed.\n");
        return 7;
    }

    cl_mem memA;
    memA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * NumElements * NumElements, A, &status);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer for menA failed.\n");
        return 7;
    }

    // 8.カーネルの引数のセット
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memA);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg for menA failed.\n");
        return 8;
    }
    
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memB);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg for menB failed.\n");
        return 8;
    }

    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memC);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg for menC failed.\n");
        return 8;
    }

    status = clSetKernelArg(kernel, 3, sizeof(int), (void *)&NumElements);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg for NumElements failed.\n");
        return 8;
    }

    // 9.カーネル実行のリクエスト
    // size_t globalWorkSize[3] = {NumElements, 0, 0};
    size_t globalWorkSize[3] = {1, 1, 1};
    // size_t localWorkSize[3] = {0, 0, 0};
    size_t localWorkSize[3] = {1, 1, 1};

    elapsed_time = my_timer();

    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel failed.\n");
        return 9;
    }

    // 10.結果の取得
    status = clEnqueueReadBuffer(queue, memA, CL_TRUE, 0, sizeof(cl_double) * NumElements * NumElements, A, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer failed.\n");
        return 10;
    }
    
    elapsed_time = my_timer() - elapsed_time;
    printf("Accelerator Elapsed time = %lf sec\n", elapsed_time);


    // 結果の出力
    double matrix_sum = 0.0;
    double rel_err = 0.0;

    for (i=0; i<NumElements*NumElements; i++){
        matrix_sum += A[i]*A[i];
    }
    matrix_sum = sqrt(matrix_sum);

    printf("Result = %lf\n", matrix_sum);

    // // 結果の表示（一部）
    // printf("(B, C, A)\n");
    // for(int i=0; i<100; i++) {
    //     printf("%f, %f, %f, (%f)\n", B[i], C[i], A[i], B[i]+C[i]);
    // }

    // 11.リソースの解放
    clReleaseMemObject(memA);
    clReleaseMemObject(memC);
    clReleaseMemObject(memB);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}