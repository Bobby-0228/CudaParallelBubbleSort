#!/bin/sh

## Copyiright (C) KUO I_HSUAN, LIN KUN_YUAN 
## USAGE: User provides paths to the file of the array to be sorted and the golden file
## Script will compile and execute bubble.cu and compare the resulting outputs with golden using the "diff" command
## EXAMPLE: run.sh array1.txt golden1.txt 

if [ $# -ne 2 ]; then
    echo "Usage: $0 array_filename golden_filename"
    exit 1
fi

array_filename=$1; golden_filename=$2;

echo -n "Compiling bubble.cu..."
nvcc -c bubble_GPU_singleblock.cu
g++ -o bubble_GPU_singleblock bubble_GPU_singleblock.o -L/usr/local/cuda/lib64 -lcudart
echo "Done!"
echo "----------------------------------------------"

echo "Executing bubble_GPU_singleblock..."
./bubble_GPU_singleblock $array_filename
echo "----------------------------------------------"

echo "Comparing outputs and golden..."
echo -n "GPU using shared memory: "
if diff $golden_filename output_GPU_shared_mem.txt > /dev/null; then
    echo "pass"
else
    echo "fail"
fi

echo -n "GPU using global memory: "
if diff $golden_filename output_GPU_global_mem.txt > /dev/null; then
    echo "pass"
else
    echo "fail"
fi

echo -n "CPU: "
if diff $golden_filename output_CPU.txt > /dev/null; then
    echo "pass"
else
    echo "fail"
fi
