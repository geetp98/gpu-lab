#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

__global__ void vector_add(double* a, double* b, double* c, int N){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid<N){
		a[tid] = b[tid]+c[tid];
	}
}

int main(){

	const int T_min = 1<<1;
	const int T_max = 1<<10;
	const int minsize = 1<<10;
	const int maxsize = 1<<27;
	int T;
	int size;
	int count = 0;
	for(T = T_min; T <= T_max; T*=2){
		int j = 10;
		printf("%d:\n",T);
		for(size = minsize; size<= maxsize; size*=2){
			const int size_2 = size*sizeof(double);
			double* h_a = (double*)malloc(size_2);
			double* h_b = (double*)malloc(size_2);
			double* h_c = (double*)malloc(size_2);

			int i = 0;
			for(i = 0; i < size; i++){
				h_b[i] = i;h_c[i] = i+1;
			}

			double *dev_a, *dev_b, *dev_c;
			cudaMalloc((double**)&dev_a, size_2);
			cudaMalloc((double**)&dev_b, size_2);
			cudaMalloc((double**)&dev_c, size_2);

			cudaMemcpy(dev_b, h_b, size_2, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_c, h_c, size_2, cudaMemcpyHostToDevice);

			int B = (int)(size/T) + (size%T?1:0); //NUM OF BLOCKS

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);
			vector_add<<< B,T >>>(dev_a, dev_b, dev_c, size);
			cudaEventRecord(stop);
			cudaMemcpy(h_a, dev_a, size_2, cudaMemcpyDeviceToHost);

			cudaEventSynchronize(stop);

			for(i = 0; i < size; i++){
				(h_a[(int)i]!=h_b[(int)i]+h_c[(int)i])?count+=1:count+=0;
				//printf("%d+%d=%d\n", h_b[i], h_c[i], h_a[i]);
			}

			cudaFree(dev_a);cudaFree(dev_b);cudaFree(dev_c);

			float time_elapsed = 0;
			cudaEventElapsedTime(&time_elapsed, start, stop);
			float throughput = size_2*1.0/time_elapsed;
			//printf("%0.1lf+%0.1lf=%0.1lf\n",h_b[size-1], h_c[size-1], h_a[size-1]);
			//(h_a[size-1]!=2*size-1)?count+=1:count+=0;
			printf("%d ",j);
			printf("%d ",size);
			printf("%f ", time_elapsed);
			printf("%0.2f\n", throughput);

			free(h_a);free(h_b);free(h_c);j+=1;
		}
		printf("\n");
	}
	printf("Total Errors: %d\n", count);
	return 0;
}
