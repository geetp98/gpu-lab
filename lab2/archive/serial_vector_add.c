#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

double vector_add(double b, double c){
	return b+c;
}

int main(){

	int minsize = 1<<20;
	int maxsize = 1<<27;
	int size;
	int j = 20;
	for(size = minsize; size <= maxsize; size*=2){
		//printf("1\n");
		int size_2 = size*sizeof(double);
		double* a = (double*)malloc(size_2);
		double* b = (double*)malloc(size_2);
		double* c = (double*)malloc(size_2);

		int i;
		for(i = 0; i < size; i++){
			b[i]=i;c[i]=i+1;
		}

		int k;
		clock_t start_time, end_time;
		start_time = clock();
		for(k = 0; k < size; k++){
			a[k] = vector_add(b[k], c[k]);
		}
		end_time = clock();

		double clock_time = (end_time-start_time)*1000/(1.0*CLOCKS_PER_SEC);
		double throughput = size_2/clock_time;

		printf("%d ",j);
		printf("%d ",size);
		printf("%0.1lf ",clock_time);
		printf("%f\n",throughput);

		free(a);free(b);free(c);j+=1;
	}

	return 0;
}
