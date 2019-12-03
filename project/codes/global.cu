#include <stdio.h>
#include <unistd.h>

/*Parallel implementation with global memory*/


/*
	This function initialises the grid. For observing performance we have
	hard coded the grid. Although, this can be made dynamic with certain changes.
*/

void setGrid(int *grid, int N)
{
	// In future, this function will set
	// the grid to whatever we want it to be
	// maybe with a string argument

	int dummy_grid[10][10] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					{0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					{0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
					{0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
					{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			grid[i*N + j] = dummy_grid[i][j];
		}
	}
}
/*
	This function prints the grid.
*/

void display(int *arr, int N)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			arr[i*N + j] ? printf("* ") : printf(". ");
		}
		printf("\n");
	}
	printf("\n");
}
/*
	This function works as kernel on gpu and it implements the logic
	to modify the grid so that it can go into next state.
*/


__global__ void calculateParallel(int* d_grid, int* d_newgrid, int N, int SIMULATE_TIME){
	const int tx_local = threadIdx.x;
	const int ty_local = threadIdx.y;
	const int tx_global = blockIdx.x * blockDim.x + tx_local;
	const int ty_global = blockIdx.y * blockDim.y + ty_local;
	int index = tx_global*N + ty_global;

	// Runs for SIMULATE_TIME times

	// For valid elements, calculates number of alive
		// neighbours and applies appropriate logic

	// Waits for barrier synchronisation,
		// then copies data into the original grid

	// Again waits for barrier synchronisation,
		// and goes again, for SIMULATE_TIME times

	for(int run = 0; run < SIMULATE_TIME; run++){
		if(tx_global >= 1 && tx_global <= N-1 && ty_global >= 1 && ty_global <= N-1){
			int aliveNeighbours = 0;
			for(int i = tx_global-1; i <= tx_global+1; i++){
				for(int j = ty_global-1; j <= ty_global+1; j++){
					aliveNeighbours += d_grid[i*N + j];
				}
			}
			aliveNeighbours -= d_grid[index];

			d_newgrid[index] = aliveNeighbours == 3 || (aliveNeighbours == 2 && d_grid[index]);

			__syncthreads();
			d_grid[index] = d_newgrid[index];
			__syncthreads();
		}
	}
}

/*
	Driver code for running the code on GPU
*/


float simulateParallel(int *grid, int N, int SIMULATE_TIME)
{
	// This function copies first set of data
	// to the GPU Global Memory, launches the
	// kernel and copies the data back

	int *d_grid, *d_newgrid;
	cudaMalloc((int**)&d_grid, N * N * sizeof(int));
	cudaMalloc((int**)&d_newgrid, N * N * sizeof(int));

	cudaMemcpy(d_grid, grid, N * N * sizeof(int), cudaMemcpyHostToDevice);

	int T = 16;
	int B = N%T ? (N/T)+1 : (N/T);

	dim3 gridSize(B,B,1);
	dim3 blockSize(T,T,1);

	float elapsed_time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	calculateParallel<<< gridSize, blockSize >>>(d_grid, d_newgrid, N, SIMULATE_TIME);

	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed_time, start, stop);

	cudaMemcpy(grid, d_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_grid);
	cudaFree(d_newgrid);
	return elapsed_time;
}

int main()
{
	int SIMULATE_TIME = 100;

	for(int N = 16; N <= 16384; N*=2){
		int *grid;
		grid = (int *)malloc(N * N * sizeof(int ));

		memset(grid, 0, N * N * sizeof(int));

		setGrid(grid, N);
		// display(grid, N);

		float elapsed_time = simulateParallel(grid, N, SIMULATE_TIME);
		// display(grid, N);

		printf("%d, %0.4f\n", N, elapsed_time/SIMULATE_TIME);

		free(grid);
	}
	return 0;
}
