#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

__global__ void vector_sqr(double* a, double* b, int N){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid<N){
		a[tid] = b[tid]*b[tid];
	}
}

int main(){

	const int T_min = 1<<1;
	const int T_max = 1<<10;
	const int minsize = 1<<10;
	const int maxsize = 1<<27;
	int count = 0;
	int size;
	int T;
	for(T = T_min; T <= T_max; T*=2){
		int j = 10;
		printf("%d:\n",T);
		for(size = minsize; size <= maxsize; size*=2){
			const int size_2 = size*sizeof(double);
			double* h_a = (double*)malloc(size_2);
			double* h_b = (double*)malloc(size_2);
			double i = 0;
			for(i = 0; i < size; i=i+1){
				h_b[(int)i]=i+1;
			}

			double *dev_a;
			double *dev_b;
			cudaMalloc((double**)&dev_a, size_2);
			cudaMalloc((double**)&dev_b, size_2);

			cudaMemcpy(dev_b, h_b, size_2, cudaMemcpyHostToDevice);

			int B = (int)(size/T) + (size%T?1:0); //NUM OF BLOCKS

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);
			vector_sqr<<< B,T >>>(dev_a, dev_b, size);
			cudaEventRecord(stop);
			cudaMemcpy(h_a, dev_a, size_2, cudaMemcpyDeviceToHost);

			cudaEventSynchronize(stop);

			for(i = 0; i < size; i=i+1){
				(h_a[(int)i]!=(i+1)*(i+1))?count+=1:count+=0;
				//(h_a[(int)i]!=(i+1)*(i+1))?printf("%0.1lf: %0.1lf\n", h_b[(int)i], h_a[(int)i]):0;
			}

			cudaFree(dev_a);cudaFree(dev_b);

			float time_elapsed = 0;
			cudaEventElapsedTime(&time_elapsed, start, stop);
			float throughput = size_2/time_elapsed;
			//printf("%0.1lf: %0.1lf\n",h_b[size-1], h_a[size-1]);
			printf("%d ",j);
			printf("%d ",size);
			printf("%lf ", time_elapsed);
			printf("%0.2f\n", throughput);

			free(h_a);free(h_b);j+=1;
		}
		printf("\n");
	}
	printf("Total Errors: %d\n", count);
	return 0;
}
