#include <stdio.h>
#include <stdlib.h>
#include "../common/cpu_bitmap.h"
#include "../common/book.h"
#include "Spheres.cu"

#define rnd(x) (x*rand()/RAND_MAX)
#define SPHERES 20
#define DIM 1024

#define INF 2e10f;

struct Sphere {
	float r, b, g; //sphere color in RGB
	float radius; //radius
	float x, y, z; //center position

	__device__ float hit(float ox, float oy, float *n) {
		/*Verifies if a straight ray coming from ox,oy hist the sphere,
		returns the z coordinate of the hit.*/
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy  <= radius*radius) {
			//If hits
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius*radius);
			return dz + z;
		}

		return -INF;
	}

};


__global__ void kernel(unsigned char *ptr, Sphere *s) {

	//Mapping pixel positions
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;

	float ox = (x-DIM/2);
	float oy = (y-DIM/2);

	//starting the spheres
	float r=0,g=0,b=0;
	float maxz = -INF;
	int i =0;

	for (i = 0; i < SPHERES; i++) {
		float n, t = s[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r *fscale;
			g = s[i].g *fscale;
			b = s[i].b *fscale;
			maxz = t;
		}

	}

	ptr[offset*4 + 0] = (int)(r*255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}


int main(void) {
	//Capture start time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	CPUBitmap bitmap(DIM,DIM);
	unsigned char *dev_bitmap;
	
	//Alocate space in dev for bitmap and the spheres

	cudaMalloc((void **)&dev_bitmap, bitmap.image_size());

	Sphere *s;

	cudaMalloc((void **)&s, sizeof(Sphere)*SPHERES);

	//Temp sphere so we initiate sphere in the host
	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
	int i =0;
	for (i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(10.0f) + 20;
	}
	//We now copy temp_s to the device

	cudaMemcpy(s,temp_s,sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice);

	free(temp_s);

	//Lets generate the bitmap
	dim3	grids(DIM/16,DIM/16);
	dim3	threads(16,16);

	kernel<<<grids,threads>>>(dev_bitmap,s);

	//Copying bitmap from device to host
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	printf("Time = %3.1f ms \n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	//bitmap.display_and_exit();

	cudaFree(dev_bitmap);
	cudaFree(s);


	return 0;

}