#include <cuda.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

#define SIZE (N * sizeof(float))
#define testLoop 1

inline __host__ __device__ void swap(float& a, float& b){
        float temp=a; a=b; b=temp;
}

__global__ void bubble(float *r, float *a, int N) {
    const int j = threadIdx.x;
    const int k = 2 * threadIdx.x;
    const int blockSize = blockDim.x;

    a = a + blockIdx.x * blockSize * 2;
    r = r + blockIdx.x * blockSize * 2;
    // __shared__ float s[4096 + 10];  // capacity > N (BLOCK_SIZE*2)
    extern __shared__ float s[];

    __syncthreads();
    s[j] = a[j];
    s[j + blockSize] = a[j + blockSize];

    for (int loop = 0; loop <= blockSize; loop++) {
        __syncthreads();
        if (s[k] > s[k + 1]) {
            swap(s[k], s[k + 1]);
        }
        __syncthreads();
        if (j < blockSize - 1) {
            if (s[k + 1] > s[k + 2]) {
                swap(s[k + 1], s[k + 2]);
            }
        }
    }

    __syncthreads();
    r[j] = s[j];
    r[j + blockSize] = s[j + blockSize];
}


void bubble_host(float *r, float *a, int N) {
    for (int k = 0; k < N; k++) {
        r[k] = a[k];
    }

    for (int loop = 0; loop <= N / 2; loop++) {
        for (int k = 0; k < N - 1; k += 2) {
            if (r[k] > r[k + 1]) {
                swap(r[k], r[k + 1]);
            }
        }
        for (int k = 1; k < N - 1; k += 2) {
            if (r[k] > r[k + 1]) {
                swap(r[k], r[k + 1]);
            }
        }
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

int main(int argc, char **argv) {
    int N = get_array_length(argv[1]);
    const int grids[] = {2, 4, 8, 16};
    const int num_grids = sizeof(grids) / sizeof(grids[0]);


    float *a = (float*)malloc(SIZE);
    float *b = (float*)malloc(SIZE);
    float *c = (float*)malloc(SIZE);
    read_array(N, a, argv[1]);

    double CPU_START, CPU_END;
    float GPUTime, CPUTime;
    cudaEvent_t start1, stop1;
	cudaEventCreate (&start1); cudaEventCreate (&stop1);

    float *gc;
    cudaMalloc((void**)&gc, SIZE);
    cudaMemcpy(gc, a, SIZE, cudaMemcpyHostToDevice);

    CPU_START=clock();
    bubble_host(b, a, N);
    CPU_END=clock();
    CPUTime = (CPU_END - CPU_START) / CLOCKS_PER_SEC *1000.0;


    for (int g = 0; g < num_grids; g++) {
        int GRID = grids[g];
        int BLOCK = N / (2 * GRID);

        printf("----------------------------------\n");
        printf("N = %d, Block = %d, Thread = %d\n", N, GRID, BLOCK);
        printf("\n");

        cudaEventRecord(start1, 0);
        cudaMemcpy(gc, a, SIZE, cudaMemcpyHostToDevice);
        cudaThreadSynchronize();

        for (int g = 0; g < GRID; g++) {
            // process I : sort inner block
            bubble<<<GRID, BLOCK, (N + 10) * sizeof(float)>>>(gc, gc, N);
            cudaThreadSynchronize();

            // process II : sort btw. blocks
            bubble<<<GRID - 1, BLOCK, (N + 10) * sizeof(float)>>>(gc + BLOCK, gc + BLOCK, N);
            cudaThreadSynchronize();
        }
        cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        
        cudaEventRecord(stop1, 0);
	    cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&GPUTime, start1, stop1);
        
        std::cout<<"time[gpu]: "<<GPUTime<<" ms\n";
        std::cout<<"time[host]: "<<CPUTime<<" ms\n";
        std::cout<<"ratio (host/shared): "<<CPUTime/GPUTime<<"\n";

        // cudaMemcpy(c, gc, SIZE, cudaMemcpyDeviceToHost);
        char filename[128];
        snprintf(filename, sizeof(filename), "multiblock_%d.txt", GRID);

        write_array(N, c, filename);
        

    }
    write_array(N, b, "output_CPU.txt");

    cudaFree(gc);
    free(a);
    free(b);
    free(c);

    return 0;
}
