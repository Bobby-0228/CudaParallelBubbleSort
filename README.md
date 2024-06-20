# Parallel Bubble Sort with CUDA


## Description
This project implements parallel bubble sort using CUDA.
The project includes scripts to compile and execute the CUDA files for GPU using single block and multiple blocks configurations.
The scripts also compare the output against golden files to validate the correctness.


## Files
- **Input Arrays**: `array1.txt` to `array5.txt`
- **Golden Outputs**: `golden1.txt` to `golden5.txt`
- **Scripts**:
  - `run_singleblock.sh`: Compile and run with single block
  - `run_multiblock.sh`: Compile and run with multiple blocks
- **Python File**:
  - `gen_array.py`: Generate array and golden files


## Usage
To execute the project, use the provided bash scripts with the appropriate array and golden filenames.


### Single Block Execution
```sh
sh run_singleblock.sh array_filename golden_filename
```
Example:
```sh
sh run_singleblock.sh array1.txt golden1.txt
```

### Multiple Blocks Execution
```sh
sh run_multiblock.sh array_filename golden_filename
```
Example:
```sh
sh run_multiblock.sh array1.txt golden1.txt
```

### Generating Arrays and Golden Files
To generate array and golden files, use the `gen_array.py` script:
```sh
python gen_array.py data_number array_length
```
Example:
```sh
python gen_array.py 0 256
```
This will create `array0.txt` and `golden0.txt` with 256 elements.

## Output
The scripts will display the runtime for both CPU and GPU implementations and indicate if the output matches the golden file (pass) or not (fail).


## Notes
- Replace `array_filename` and `golden_filename` with the actual filenames of the input array and the corresponding golden output.
