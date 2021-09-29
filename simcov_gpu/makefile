buildpath = "/home/kirtus/Projects/simcov_gpu/simcov_gpu_b"
all:
	g++ -Iinclude src/utils.cpp src/options.cpp src/tissue.cpp src/main.cpp -o simcov_cpu.out -O3
gpu:
    nvcc -std=c++11 -Iinclude --ptxas-options=-v --compiler-options '-fPIC' -o libtissue_gpu.so --shared src/tissue_gpu.cu -L/usr/local/cuda/lib64 -lcurand
    nvcc -std=c++11 -Iinclude -c src/utils.cpp
    nvcc -std=c++11 -Iinclude -c src/options.cpp
    nvcc -std=c++11 -Iinclude -c src/main.cpp
    nvcc -std=c++11 -o simcov_gpu.out utils.o options.o main.o -L$(buildpath) -ltissue_gpu
debug:
	g++ -Iinclude src/utils.cpp src/options.cpp src/tissue.cpp src/main.cpp -o simcov_cpu.out -g