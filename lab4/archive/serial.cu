#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int compare_ints(const void *p, const void *q)
{
	int x = *(const int *)p;
	int y = *(const int *)q;
	if (x < y)
	{
		return -1;
	}
	else if (x > y)
	{
		return 1;
	}
	return 0;
}

float medianFilterSerial(int **img, int height, int width, int size, int padsize)
{
	int i = 0, j = 0;

	int **val = (int **)malloc((height + (2 * padsize)) * sizeof(int *));
	while (i < height + (2 * padsize))
	{
		val[i] = (int *)malloc((width + (2 * padsize)) * sizeof(int *));
		memset(val[i], 0, width * sizeof(int));
		i++;
	}

	i = padsize;
	while (i < height + padsize)
	{
		j = padsize;
		while (j < width + padsize)
		{
			val[i][j] = img[i - padsize][j - padsize];
			j++;
		}
		i++;
	}

	i = padsize;
	int retval = 0;
	int i1, i2;
	int *arr = (int *)malloc(size * size * sizeof(int));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time_out = 0;

	cudaEventRecord(start);
	while (i < height + padsize)
	{
		j = padsize;
		while (j < width + padsize)
		{
			i1 = i - padsize;
			i2 = j - padsize;
			while (i1 <= i + padsize)
			{
				i2 = j - padsize;
				while (i2 <= j + padsize)
				{
					arr[((i1 - i + padsize) * size) + i2 - j + padsize] = val[i1][i2];
					i2++;
				}
				i1++;
			}
			qsort(arr, size * size, sizeof(int), compare_ints);
			retval = arr[size * size / 2];
			memset(arr, 0, size * size);
			img[i - padsize][j - padsize] = retval;
			j++;
		}
		i++;
	}
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time_out, start, stop);
	free(arr);
	free(val);
	return time_out;
}

int main(int argc, char *argv[])
{
	int c = 1;
	while (c < argc)
	{
		int size = 3;
		while (size <= 15)
		{
			FILE *fp = fopen(argv[c], "r");
			//FILE *fp2 = fopen(argv[2], "w");

			int i = 0, j = 0;
			char type[4];
			fscanf(fp, "%s", &type);
			//printf("Type: %s\n", type);
			//fprintf(fp2, "%s\n", type);
			int width, height;
			fscanf(fp, "%d %d", &width, &height);
			//fprintf(fp2, "%d %d\n", width, height);
			printf("%d ", width);
			int colour_depth;
			fscanf(fp, "%d", &colour_depth);
			//fprintf(fp2, "%d\n", colour_depth);
			//printf("Colour Depth: %d\n", colour_depth);

			int **red = (int **)malloc(height * sizeof(int *));
			int **green = (int **)malloc(height * sizeof(int *));
			int **blue = (int **)malloc(height * sizeof(int *));
			while (i < height)
			{
				red[i] = (int *)malloc(width * sizeof(int));
				green[i] = (int *)malloc(width * sizeof(int));
				blue[i] = (int *)malloc(width * sizeof(int));
				i++;
			}

			int c1, c2, c3;
			i = 0;
			while (i < height)
			{
				j = 0;
				while (j < width)
				{
					fscanf(fp, "%d %d %d", &c1, &c2, &c3);
					red[i][j] = c1;
					green[i][j] = c2;
					blue[i][j] = c3;
					j++;
				}
				i++;
			}

			const int padsize = size / 2;
			printf("%d ", size);

			/*i = 0;
	while (i < height)
	{
		j = 0;
		while (j < width)
		{
			printf("%d\t%d\t%d\t", red[i][j], green[i][j], blue[i][j]);
			j++;
		}
		printf("\n");
		i++;
	}*/

			float time_elapsed = 0;

			time_elapsed += medianFilterSerial(red, height, width, size, padsize);
			time_elapsed += medianFilterSerial(green, height, width, size, padsize);
			time_elapsed += medianFilterSerial(blue, height, width, size, padsize);
			printf("%0.2f\n", time_elapsed);
			i = 0;
			/*while (i < height)
	{
		j = 0;
		while (j < width)
		{
			printf("%d\t%d\t%d\t", red[i][j], green[i][j], blue[i][j]);
			j++;
		}
		printf("\n");
		i++;
	}*/

			/*i = 0;
	while (i < height)
	{
		j = 0;
		while (j < width)
		{
			fprintf(fp2, "%d %d %d\n", red[i][j], green[i][j], blue[i][j]);
			j++;
		}
		i++;
	}*/

			free(red);
			free(green);
			free(blue);
			fclose(fp);
			//fclose(fp2);
			size += 2;
		}
		printf("\n\n");
		c++;
	}
	return 0;
}
