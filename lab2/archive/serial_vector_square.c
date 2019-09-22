#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

double vector_sqr(int b){
	return (double)b*b;
}

int main(){

	int minsize = 1<<20;
	int maxsize = 1<<27;
	int size;
	int j = 20;
	for(size = minsize; size <= maxsize; size*=2){
		int size_2 = sizeof(double)*size;
		double* a = (double*)malloc(size_2);
		double* b = (double*)malloc(size_2);

		int i;
		for(i = 0; i < size; i++){
			b[i] = i;
		}

		int k;
		clock_t start_time, end_time;
		start_time = clock();
		for(k = 0; k < size; k++){
			a[k] = vector_sqr(b[k]);
		}
		end_time = clock();

		float clock_time = (end_time-start_time)*1000/(1.0*CLOCKS_PER_SEC);
		float throughput = size_2/clock_time;

		printf("%d ",j);
		printf("%d ",size);
		printf("%0.1lf ",clock_time);
		printf("%f\n", throughput);

		free(a);free(b);j+=1;
	}

	return 0;
}
