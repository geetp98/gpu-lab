#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <sys/file.h>
#include <unistd.h>

double vector_sqr(double a){
	return a*a;
}

int main(){

	int minsize = 1<<8;
	int maxsize = 1<<29;

	int size;
	for(size = minsize; size <= maxsize; size*=2){
		//printf("1\n");
		int runs = maxsize/size;
		double* a = (double*)malloc(size*sizeof(double));

		int i, k;
		clock_t start_time, end_time;
		start_time = clock();
		for(i = 0; i < runs; i++){
			for(k = 0; k < size; k++){
				a[k] = vector_sqr(a[k]);
			}
		}
		end_time = clock();

		free(a);

		double clock_time = (end_time-start_time)/(1.0*CLOCKS_PER_SEC);
		double throughput = sizeof(double)*1.0*maxsize/clock_time;
		printf("%d ",size);
		printf("%0.2lf ",clock_time);
		printf("%0.2f\n", throughput);

	}

	return 0;
}
