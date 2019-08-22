#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/file.h>
#include <unistd.h>

int main(){

	int fd;
	char *name = "output";
	fd = open(name, O_WRONLY | O_CREAT, 0644);
	if (fd == -1) {
		perror("open failed");
		exit(1);
	}

	if (dup2(fd, 1) == -1) {
		perror("dup2 failed"); 
		exit(1);
	}
	double minsize = pow(2,8);
	double maxsize = pow(2,29);

	double total = maxsize;
	long long size;
	for (size = minsize; size < maxsize; size*=2)
	{
		long long runs = total/size;
		double *A = (double*)malloc(size*sizeof(double));
		double *B = (double*)malloc(size*sizeof(double));
		double *C = (double*)malloc(size*sizeof(double));
		double *D = (double*)malloc(size*sizeof(double));

		clock_t start_time, end_time;
		start_time = clock();

		long long i;
		long long k;
		for (k = 0; k < runs; k++)
		{
			for (i = 0; i < size; i++)
			{
				A[i] = B[i] + C[i]*D[i];
			}
		}
		free(A);free(B);free(C);free(D);
		end_time = clock() - start_time;
		double clock_time = (double)end_time/(double) CLOCKS_PER_SEC;
		double throughput = (sizeof(double)*2*total)/clock_time;
		printf("%lld ",size);
		printf("%0.2lf\n", throughput);
	}

	return 0;
}
