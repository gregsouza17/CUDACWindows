#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	//tid = thread index
	c[tid] = a[tid] + b[tid];
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

void random_ints(int *a, int M);

int main(void) {
	int *a, *b, *c; //pointers to host copies of the values
	int *d_a, *d_b, *d_c; //device copies of a,b,c
	int size = N * sizeof(int);

	//Alocating space in the device
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	//Alloc space of rhost copies of a,b,c
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);

	//Copy to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//Launch function on GPU
	add << <(N+127)/128, 128 >> >(d_a, d_b, d_c);

	//Copy results back
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	int i;
	for (i = 0; i < N; i++) {
		if (i % 100000) {
			printf("%d + %d = %d\n", a[i], b[i], c[i]);
		}
	}

	//cleanup

	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);


	return 0;
}

void random_ints(int *a, int M) {
	int i;

	for (i = 0; i < M; i++) {
		a[i] = rand() % 5000;

	}

}