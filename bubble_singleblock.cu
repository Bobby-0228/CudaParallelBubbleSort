#include <cuda.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>

#define SIZE   (N*sizeof(float))

#define GRID   1
#define BLOCK  (N/2)

inline __host__ __device__ void swap(float& a, float& b){
        float temp=a; a=b; b=temp;
}

// Bubble Sort Kernel
__global__ void bubble(float *r, float *a, int N){
    //*** blockDim=N/2 ***
    int j=threadIdx.x;      //j=0,1,2,...blockDim-1 
    int k=2*threadIdx.x;    //k=0,2,4,...2*(blockDim-1)

    //allocate shared memory
    extern __shared__ float s[];

    //load data to shared memory
    __syncthreads();   
    s[j]=a[j];         //use all threads to load first half (0~N/2-1)
    s[j+N/2]=a[j+N/2]; //use all threads to load second half (N/2~N-1)
    if(j==0)           //if N is odd
        s[N-1]=a[N-1];
    

    // bubble sort algorithm
    for(int loop=0; loop<=N/2; loop++){
        // sort 0 based
        __syncthreads();
        if(s[k]>s[k+1])
            swap(s[k],s[k+1]);
        
        // sort 1 based
        __syncthreads();
        if(s[k+1]>s[k+2])
            if(k<N-2) //if N is even
                swap(s[k+1],s[k+2]);
        
    }

    // write back to global memory
    __syncthreads();
    r[j]=s[j];
    r[j+N/2]=s[j+N/2];
    if(j==0)
        r[N-1]=s[N-1];
    

}

// use only global memory to execute bubble sort in GPU
__global__ void bubble_global(float *r, float *a, int N){
    int j = threadIdx.x;      // j = 0, 1, 2, ..., blockDim-1 
    int k = 2 * threadIdx.x;  // k = 0, 2, 4, ..., 2*(blockDim-1) 

    // start bubble sort
    for (int loop = 0; loop <= N/2; loop++) {
        // sort 0 based
        __syncthreads();  
        if (k < N-1 && a[k] > a[k+1])
            swap(a[k], a[k+1]);
        
        // sort 1 based 
        __syncthreads();  
        if (k+1 < N-1 && a[k+1] > a[k+2])
            swap(a[k+1], a[k+2]);
    }

    // write back to global memory
    __syncthreads();
    r[j] = a[j];
    r[j + N/2] = a[j + N/2];
    if (j == 0 && N % 2 != 0) {
        r[N-1] = a[N-1];
    }
}

// Bubble Sort Host Function
void bubble_host(float *r, float *a, int N){
    for(int k=0; k<N; k++)
        r[k]=a[k];

    for(int loop=0; loop<=N/2; loop++){
        // sort 0 based
        for(int k=0; k<N-1; k+=2)
            if(r[k]>r[k+1])
                swap(r[k],r[k+1]);
        // sort 1 based
        for(int k=1; k<N-1; k+=2)
            if(r[k]>r[k+1])
                swap(r[k],r[k+1]);
    }
}



int get_array_length(char *filename){
    std::ifstream fin;
    fin.open(filename);
    int count = 0;
    int temp;
    while (fin >> temp)
        count++;
    fin.close();
    std::cout<<"array length: "<<count<<std::endl;
    return count;
}

void read_array(int N, float *arr, char *filename){
    std::ifstream fin;
    fin.open(filename);
    for(int k=0; k<N; k++)
        fin>>arr[k];
    fin.close();
}

void write_array(int N, float *arr, char *filename){
    std::ofstream fout;
    fout.open(filename);
    for(int k=0; k<N; k++)
        fout<<arr[k]<<" ";
    fout.close();
}

int main(int argc, char **argv){
    int N = get_array_length(argv[1]);
    float *a=(float*)malloc(SIZE); // data array
    float *b=(float*)malloc(SIZE); // host array
    float *c=(float*)malloc(SIZE); // device array (shared memory)
    float *d=(float*)malloc(SIZE); // device array (only use global memory)
    read_array(N, a, argv[1]);

    double CPU_START, CPU_END;
    float GPUTime1, GPUTime2, CPUTime;
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate (&start1); cudaEventCreate (&stop1);
    cudaEventCreate (&start2); cudaEventCreate (&stop2);


    // allocate device memory
    float  *ga, *gc, *gd;
    cudaMalloc((void**)&ga, SIZE);
    cudaMalloc((void**)&gc, SIZE);
    cudaMalloc((void**)&gd, SIZE);

    // Load
    cudaMemcpy(ga, a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gc, c, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gd, d, SIZE, cudaMemcpyHostToDevice);


    // shared memory in GPU
    cudaEventRecord(start1, 0);
    bubble<<<1,BLOCK,(N+20)*sizeof(float)>>>(gc,ga,N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&GPUTime1, start1, stop1);

    // only global memory in GPU
    cudaEventRecord(start2, 0);
    bubble_global<<<1,BLOCK>>>(gd,ga,N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&GPUTime2, start2, stop2);

    // host
    CPU_START=clock();
    bubble_host(b,a,N);
    CPU_END=clock();

    CPUTime = (CPU_END - CPU_START) / CLOCKS_PER_SEC *1000.0;
    std::cout<<"time[gpu shared]: "<<GPUTime1<<" ms\n";
    std::cout<<"time[gpu global]: "<<GPUTime2<<" ms\n";
    std::cout<<"time[host]: "<<CPUTime<<" ms\n";
    std::cout<<"ratio (host/shared): "<<CPUTime/GPUTime1<<"\n";
    std::cout<<"ratio (host/global): "<<CPUTime/GPUTime2<<"\n";


    // copy device data
    cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(d, gd, SIZE, cudaMemcpyDeviceToHost);


    //store sorted array -> shared memory in GPU
    write_array(N, c, "output_GPU_shared_mem.txt");

    //TEST -> only global memory in GPU
    write_array(N, d, "output_GPU_global_mem.txt");

    //TEST -> host
    write_array(N, b, "output_CPU.txt");

    //free memory 
    cudaFree(ga);
    cudaFree(gc);
    cudaFree(gd);
    free(a);
    free(b);
    free(c);
    free(d);

    return 0;
}
