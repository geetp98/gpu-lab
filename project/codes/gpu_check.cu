#include <stdio.h>

int main()
{
	cudaDeviceProp dev;
	int dev_cnt = 0;
	cudaGetDeviceCount (&dev_cnt);
	
	for(int i = 0; i < dev_cnt; i++)
	{
		cudaGetDeviceProperties(&dev, i);
		printf("Device %d:\n", i);
		printf("\tName: %s\n",dev.name);
		printf("\tCompute Capability %d.%d\n",dev.major, dev.minor);
		printf("\tMultiprocessor Count: %d\n", dev.multiProcessorCount);
		printf("\tTotal Global Memory (GB): %0.2f\n", 1.0*dev.totalGlobalMem/(1024*1024*1024));
		printf("\tTotal Constant Memory (kB): %d\n", dev.totalConstMem/1024);
		printf("\tShared Memory per Block (kB): %d\n",dev.sharedMemPerBlock/1024);
		printf("\tRegisters per Block: %d\n", dev.regsPerBlock);
		printf("\tMax Threads per Block: %d\n",dev.maxThreadsPerBlock);
		printf("\tMax Threads per Multiprocessor: %d\n", dev.maxThreadsPerMultiProcessor);
		printf("\tWarp Size: %d\n", dev.warpSize);
		printf("\tMax Thread Dim: %d,%d,%d\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
		printf("\tMax Grid Dim: %d,%d,%d\n", dev.maxGridSize[0],dev.maxGridSize[1], dev.maxGridSize[2]);
		printf("\tMemory Pitch: %d\n", dev.memPitch);
		printf("\tL2 Cache Size: %d\n", dev.l2CacheSize);
		printf("\tClock Rate (kHz): %d\n",dev.clockRate);
		printf("\tMemory Clock Rate (kHz): %d\n", dev.memoryClockRate);
		printf("\tMemory Bus Width (bits): %d\n", dev.memoryBusWidth);
	}
	return 0;
}
