#include "complex.cu"
#include <stdio.h>
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define	DIM 1000

__device__ int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale*(float)(DIM/2 - x)/(DIM/2);
	float jy = scale*(float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.2, 0);
	cuComplex a(jx,jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a*a*a+c;
		if (a.magnitude2()>1000)
			return 0;

	}

	return 1;
}


__global__ void kernel(unsigned char *ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x+y*gridDim.x;

	int juliaValue = julia(x,y);
	ptr[offset*4 + 0 ] = 122*juliaValue;
	ptr[offset * 4 + 1] = 0 * juliaValue;
	ptr[offset * 4 + 2] = 122 * juliaValue;
	ptr[offset*4 +3] = 255;

}

int main(void) {
	clock_t begin = clock();
	CPUBitmap bitmap(DIM,DIM);
	unsigned char *dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	dim3 grid(DIM,DIM);

	kernel<<<grid,1>>>(dev_bitmap);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Tempo gasto %lf \n", time_spent);

	return 0;
}
