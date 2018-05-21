#include <stdio.h>

__global__ void add(int *v1, int *v2, int *sol) {
	*sol = *v1 + *v2;
}


int main(void) {
	int v1, v2, sol; //Host copies of the values
	int *d_v1, *d_v2, *d_sol; //int vector for v1,v2,v3 in the device
	int size = sizeof(int);

	//Allocating space in the device
	cudaMalloc((void **)&d_v1, size);
	cudaMalloc((void **)&d_v2, size);
	cudaMalloc((void **)&d_sol, size);

	//setup input
	v1 = 11;
	v2 = 13;

	//Input values to device
	cudaMemcpy(d_v1, &v1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v2, &v2, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sol, &sol, size, cudaMemcpyHostToDevice);

	//Lauch add() in device
	add << <1, 1 >> >(d_v1, d_v2, d_sol);

	//Copy result from device to host
	cudaMemcpy(&sol, d_sol, size, cudaMemcpyDeviceToHost);

	//Cleanup
	printf(" Hello %d \n", sol);
	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_sol);

	return 0;


}