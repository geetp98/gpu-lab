#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHANNELS 3
typedef struct
{
	unsigned char red, green, blue;
} PPMPixel;

typedef struct
{
	int x, y;
	unsigned char *data;
} PPMImage;

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

__global__ void colCon(unsigned char *greyImg, unsigned char *rgbImage, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int Offset = y * width + x;
	if (x < width && y < height)
	{
		unsigned char sum = 0.21f * rgbImage[CHANNELS * Offset] + 0.71f * rgbImage[CHANNELS * Offset + 1] + 0.07f * rgbImage[CHANNELS * Offset + 2];
		greyImg[CHANNELS * Offset] = sum;
		greyImg[CHANNELS * Offset + 1] = sum;
		greyImg[CHANNELS * Offset + 2] = sum;
	}
}

static PPMImage *readPPM(const char *filename)
{
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	//open PPM file for reading
	fp = fopen(filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//read image format
	if (!fgets(buff, sizeof(buff), fp))
	{
		perror(filename);
		exit(1);
	}

	//check the image format
	if (buff[0] != 'P' || buff[1] != '6')
	{
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	//alloc memory form image
	img = (PPMImage *)malloc(sizeof(PPMImage));
	if (!img)
	{
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//check for comments
	c = getc(fp);
	while (c == '#')
	{
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	//read image size information
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2)
	{
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1)
	{
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color != RGB_COMPONENT_COLOR)
	{
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;
	//memory allocation for pixel data
	img->data = (unsigned char *)malloc(img->x * img->y * sizeof(unsigned char) * 3);

	if (!img)
	{
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//read pixel data from file
	if (fread(img->data, 3 * img->x, img->y, fp) != img->y)
	{
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}
void writePPM(const char *filename, PPMImage *img)
{
	FILE *fp;
	//open file for output
	fp = fopen(filename, "wb");
	if (!fp)
	{
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//write the header file
	//image format
	fprintf(fp, "P6\n");

	//comments
	fprintf(fp, "# Created by %s\n", CREATOR);

	//image size
	fprintf(fp, "%d %d\n", img->x, img->y);

	// rgb component depth
	fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

	// pixel data
	fwrite(img->data, 3 * img->x, img->y, fp);
	fclose(fp);
}

int main(int argc, char **argv)
{
	PPMImage *image, *image_device;
	unsigned char *grey, *rgb;
	int size_of_image;
	float elapsed;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int T;
	for(T = 2; T <= 32; T *= 2)
	{
		printf("%d\n",T*T);
		for (int i = 1; i < argc; i++)
		{
			image = readPPM(argv[i]);
			image_device = (PPMImage *)malloc(sizeof(PPMImage));
			image_device->x = image->x;
			image_device->y = image->y;
			size_of_image = 3 * image->x * image->y;
			image_device->data = (unsigned char *)malloc(size_of_image);
			printf("%d %d ", image->x, image->y);
			cudaMalloc(&rgb, size_of_image * sizeof(char));
			cudaMalloc(&grey, size_of_image * sizeof(char));
			cudaMemcpy(rgb, image->data, size_of_image, cudaMemcpyHostToDevice);
			dim3 gridSize((image->x - 1) / T + 1, (image->y - 1) / T + 1, 1);
			dim3 blockSize(T, T, 1);
			cudaEventRecord(start);
			colCon<<<gridSize, blockSize>>>(grey, rgb, image->x, image->y);
			cudaEventRecord(stop);
			cudaDeviceSynchronize();
			cudaEventElapsedTime(&elapsed, start, stop);
			//char *file_name = ((char) (i));
			printf("%f ", elapsed);
			printf("%f\n", 0.001*image->x*image->y*3*sizeof(int)/elapsed);
			cudaMemcpy(image_device->data, grey, size_of_image, cudaMemcpyDeviceToHost);
			//writePPM("output_images/abc.ppm", image_device);
			//printf("conversion complete\n");
		}
		printf("\n\n");
	}
	return 0;
}
