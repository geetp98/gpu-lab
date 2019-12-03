#include <stdio.h>
#include <unistd.h>

// Format: 2^n + 2
#define THREADS 16
#define TILE_SIZE 16

/*Parallel Implementation with shared memory*/



/*
	This function initialises the grid. For observing performance we have
	hard coded the grid. Although, this can be made dynamic with certain changes.
*/
void setGrid(int *grid, int N)
{
	// In future, this function will set
	// the grid to whatever we want it to be
	// maybe with a string argument

	int dummy_grid[16][16] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
								{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
							};

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
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


__global__ void calculateParallel2(int* d_grid, int N, int SIMULATE_TIME){
	const int tx_local = threadIdx.x;
	const int ty_local = threadIdx.y;
	const int column = blockIdx.x * blockDim.x + tx_local;
	const int row = blockIdx.y * blockDim.y + ty_local;

	__shared__ unsigned int sharedMem[TILE_SIZE+2][TILE_SIZE+2];

	if(row<N && column<N){
		for(int run = 0; run < SIMULATE_TIME; run++){
			//first row padding
			if(ty_local == 0){
				sharedMem[0][tx_local+1]=0;
			}
			//first column padding
			if(tx_local == 0){
				sharedMem[ty_local+1][0]=0;
			}
			//last row padding
			if(ty_local==N-1){
				
				//tile_size+1=17
				sharedMem[TILE_SIZE+1][tx_local+1]=0;
			}

			//last column padding
			if(tx_local==N-1){
				sharedMem[ty_local+1][TILE_SIZE+1]=0;
			}

			//corner cells padding
			if(tx_local==0 && ty_local==0){
				sharedMem[0][0]=0;
			}
			if(tx_local==0 && ty_local==N-1){
				sharedMem[0][TILE_SIZE+1]=0;
			}
			if(tx_local==N-1 && ty_local==N-1){
				sharedMem[TILE_SIZE+1][TILE_SIZE+1]=0;
			}
			if(tx_local==N-1 && ty_local==0){
				sharedMem[TILE_SIZE+1][0]=0;
			}

			// Each cell copies a cell from global to shared memory
			sharedMem[ty_local+1][tx_local+1] = d_grid[row*N+column];

			//Barrier synchronisation for initialising shared memory
			__syncthreads();

			int aliveNeighbours=0;
			int x=tx_local+1;
			int y=ty_local+1;

			// Game's logic
			if(x<N-1 && y<N-1){
				for(int i=-1;i<=1;i++){
					for(int j=-1;j<=1;j++){
						aliveNeighbours+=sharedMem[y+i][x+j];
					}
				}
			}

			aliveNeighbours-=sharedMem[y][x];

			// Each cell copies back from shared to global memory
			d_grid[row*N + column] = aliveNeighbours == 3 || (aliveNeighbours == 2 && sharedMem[y][x]);

			//Barrier synchronisation for completing each state
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

	int *d_grid;
	cudaMalloc((int**)&d_grid, N * N * sizeof(int));
	// cudaMalloc((int**)&d_newgrid, N * N * sizeof(int));

	cudaMemcpy(d_grid, grid, N * N * sizeof(int), cudaMemcpyHostToDevice);

	int BLOCKS = N%THREADS ? (N/THREADS)+1 : (N/THREADS);

	dim3 gridSize(BLOCKS,BLOCKS,1);
	dim3 blockSize(THREADS,THREADS,1);

	float elapsed_time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	calculateParallel2<<< gridSize, blockSize >>>(d_grid, N, SIMULATE_TIME);

	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed_time, start, stop);

	cudaMemcpy(grid, d_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_grid);
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

		printf("%d, %0.4f\n", N, N, elapsed_time/SIMULATE_TIME);

		free(grid);
	}

	return 0;
}
