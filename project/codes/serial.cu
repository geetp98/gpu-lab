#include <stdio.h>
#include <unistd.h>

#define DISPLAY_ON 1
#define DISPLAY_OFF 0

/*Serial implementation*/


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
					{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
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
	This function implements the actual logic of the game. This function
	implements the condition to decide the value of a particular cell
	in its next generation.
*/
int setValue(int *grid, int x, int y, int N)
{
	int aliveNeighbours = 0;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			aliveNeighbours += grid[((x+i)*N) + (y+j)];
		}
	}
	aliveNeighbours -= grid[x*N + y];

	return aliveNeighbours == 3 || (aliveNeighbours == 2 && grid[x*N + y]);
}
/*
	This function helps creating new grid based on previous grid
*/

void nextGen(int *grid, int *newgrid, int N)
{
	for (int i = 1; i < N - 1; i++)
	{
		for (int j = 1; j < N - 1; j++)
		{
			newgrid[i*N + j] = setValue(grid, i, j, N);
		}
	}

	for (int i = 1; i < N-1; i++)
	{
		for (int j = 1; j < N-1; j++)
		{
			grid[i*N + j] = newgrid[i*N + j];
		}
	}
}

/*
	Driver code for running the code on GPU
*/

float simulateSerial(int *grid, int N, int SIMULATE_TIME, int disp_var)
{
	// calls the function nextGen for the new grid
	// Updates the grid, sleeps and then displays.
	// It does this for SIMULATE_TIME times,
	// and returns the total runtime

	int *newgrid;
	newgrid = (int *)malloc(N * N * sizeof(int *));

	if(disp_var){
		display(grid, N);
	}

	float elapsed_time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (int i = 0; i < SIMULATE_TIME; i++)
	{
		nextGen(grid, newgrid, N);
        // usleep(100000);
        if(disp_var){
			display(grid, N);
		}
	}

	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed_time, start, stop);

	free(newgrid);
	return elapsed_time;
}

int main()
{
	int SIMULATE_TIME = 100;
	for(int N = 16; N <= 2048; N*=2){
		int *grid;
		grid = (int *)malloc(N * N * sizeof(int ));

		memset(grid, 0, N * N * sizeof(int));

		setGrid(grid, N);
		// display(grid, N);

		float elapsed_time = simulateSerial(grid, N, SIMULATE_TIME, DISPLAY_OFF);
		// display(grid, N);

		printf("%d %0.4f\n", N, N, elapsed_time/SIMULATE_TIME);

		free(grid);
	}

	return 0;
}
