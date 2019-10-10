#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define shared_window 68
#define tile_width 64
#define size 11
#define padsize 5

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

__global__ void medianFilterKernel(int *dev_in, int *dev_out, int height, int width)
{
	const int tlocal_x = threadIdx.x;
	const int tlocal_y = threadIdx.y;
	const int tglobal_x = blockIdx.x * blockDim.x + tlocal_x;
	const int tglobal_y = blockIdx.y * blockDim.y + tlocal_y;

	int i;
	int j;

	if (tglobal_x >= 0 && tglobal_x < height && tglobal_y >= 0 && tglobal_y < width)
	{
		//const int shared_window = tile_width+(2*padsize);
		__shared__ int shared_mem[shared_window][shared_window];
		// pad the shared memory
		if (tlocal_x == 0 && tlocal_y == 0)
		{
			i = 0;
			while (i < padsize)
			{
				j = 0;
				while (j < padsize)
				{
					shared_mem[i][j] = dev_in[(tglobal_x - padsize + i) * width + tglobal_y - padsize + j];
					j++;
				}
				i++;
			}
			// bring data at the upper left corner
		}
		else if (tlocal_x == tile_width - 1 && tlocal_y == tile_width - 1)
		{
			// bring data at the lower right corner
			i = padsize;
			while (i < 2 * padsize)
			{
				j = padsize;
				while (j < 2 * padsize)
				{
					shared_mem[tlocal_x + i][tlocal_y + j] = dev_in[(tglobal_x - padsize + i) * width + tglobal_y - padsize + j];
					j++;
				}
				i++;
			}
		}
		else if (tlocal_x == 0 && tlocal_y == tile_width - 1)
		{
			// bring data at the upper right corner
			i = 0;
			while (i < padsize)
			{
				j = padsize;
				while (j < 2 * padsize)
				{
					shared_mem[i][tlocal_y + j] = dev_in[(tglobal_x - padsize + i) * width + tglobal_y - padsize + j];
					j++;
				}
				i++;
			}
		}
		else if (tlocal_y == 0 && tlocal_x == tile_width - 1)
		{
			// bring data at the lower left corner
			i = padsize;
			while (i < 2 * padsize)
			{
				j = 0;
				while (j < padsize)
				{
					shared_mem[tlocal_x + i][j] = dev_in[(tglobal_x - padsize + i) * width + tglobal_y - padsize + j];
					j++;
				}
				i++;
			}
		}
		else if (tlocal_x == 0)
		{
			// bring data of the rows above
			i = 0;
			while (i < padsize)
			{
				shared_mem[i][tlocal_y] = dev_in[(tglobal_x - padsize + i) * width + tglobal_y];
				i++;
			}
		}
		else if (tlocal_x == tile_width - 1)
		{
			// bring data of the rows below
			i = padsize;
			while (i < 2 * padsize)
			{
				shared_mem[tlocal_x + i][tlocal_y] = dev_in[(tglobal_x - padsize + i) * width + tglobal_y];
				i++;
			}
		}
		else if (tlocal_y == 0)
		{
			// bring data of the columns at left
			j = 0;
			while (j < padsize)
			{
				shared_mem[tlocal_x][j] = dev_in[(tlocal_x * width) + tglobal_y];
				j++;
			}
		}
		else if (tlocal_y == tile_width - 1)
		{
			// bring data at the columns at right
			j = padsize;
			while (j < 2 * padsize)
			{
				shared_mem[tlocal_x][tlocal_y + j] = dev_in[(tlocal_x * width) + tglobal_y];
				j++;
			}
		}

		// bring in the threadwise data
		shared_mem[tlocal_x + padsize][tlocal_y + padsize] = dev_in[tglobal_x * (width + (2 * padsize)) + tglobal_y];
		__syncthreads();

		// sorting and selecting
		int arr[size * size];
		i = 0;
		while (i < size)
		{
			j = 0;
			while (j < size)
			{
				arr[i * size + j] = shared_mem[tlocal_x + i][tlocal_y + i];
				j++;
			}
			i++;
		}

		int swap;
		for (i = 0; i < (size * size) - 1; i++)
		{
			for (j = 0; j < (size * size) - i - 1; j++)
			{
				if (arr[j] > arr[j + 1])
				{
					swap = arr[j + 1];
					arr[j + 1] = arr[j];
					arr[j] = swap;
				}
			}
		}

		// assign final values
		dev_out[(tglobal_x + padsize) * width + height] = arr[size * size / 2];
	}
}

float medianFilterParallel(int **img, int height, int width)
{
	int i = 0, j = 0;
	int *dev_in, *dev_out;
	int **image = (int **)malloc((height + (2 * padsize)) * sizeof(int *));
	cudaMalloc((int **)&dev_in, (height + 2 * padsize) * (width + 2 * padsize) * sizeof(int));
	cudaMalloc((int **)&dev_out, (height + 2 * padsize) * (width + 2 * padsize) * sizeof(int));
	while (i < height + (2 * padsize))
	{
		image[i] = (int *)malloc((width + (2 * padsize)) * sizeof(int *));
		memset(image[i], 0, width * sizeof(int));
		i++;
	}
	//printf("Create dummy array\n");
	i = padsize;
	while (i < height + padsize)
	{
		j = padsize;
		while (j < width + padsize)
		{
			image[i][j] = img[i - padsize][j - padsize];
			j++;
		}
		i++;
	}
	//printf("Copy Data\n");

	cudaMemcpy(dev_in, image, (height + 2 * padsize) * (width + 2 * padsize) * sizeof(int), cudaMemcpyHostToDevice);
	//printf("Cudamemcpy\n");
	//cudaDeviceProperties properties;
	//int sharedMemSize;
	//cudaGetDeviceProperties(&properties, 0);
	//sharedMemSize = properties.sharedMemPerBlock;
	int T = 16;
	dim3 gridSize(((height - 1) / T) + 1, ((width - 1 / T)) + 1, 1);
	dim3 blockSize(T, T, 1);
	// tile_width = f(sharedMemSize)
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time_elapsed;

	cudaEventRecord(start);
	medianFilterKernel<<<gridSize, blockSize>>>(dev_in, dev_out, height, width);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time_elapsed, start, stop);
	//printf("Cuda Function call returned\n");

	cudaMemcpy(image, dev_out, (height + 2*padsize)*(width + 2*padsize)*sizeof(int), cudaMemcpyDeviceToHost);
	//printf("CudaMemCpy back successful\n");
	// image -> img

	i = padsize;
	while (i < height + padsize)
	{
		j = padsize;
		while (j < width + padsize)
		{
			//printf("%d,%d:%d,%d\n", i,j,i-padsize,j-padsize);
			img[i - padsize][j - padsize] = image[i][j];
			j++;
		}
		//printf("\n\n\n\n");
		i++;
	}

	//printf("Everything Done\n");

	free(image);
	cudaFree(dev_in);
	cudaFree(dev_out);
	return time_elapsed;
}

int main(int argc, char *argv[])
{
	int c = 1;
	while (c < argc)
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

		time_elapsed += medianFilterParallel(red, height, width);
		time_elapsed += medianFilterParallel(green, height, width);
		time_elapsed += medianFilterParallel(blue, height, width);
		printf("%d %0.2f", size, time_elapsed);
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
	}
	return 0;
}
