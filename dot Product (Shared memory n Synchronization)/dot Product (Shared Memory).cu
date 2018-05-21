#include <stdio.h>

#define imin(a,b) (a<b?a:b)

const int N = 32;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/ threadsPerBlock) ;
//const int blocksPerGrid = 32 ;

__global__ void dot(float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < N) {
		temp += a[tid]*b[tid];
		tid+= blockDim.x * gridDim.x;
	}

	//set the cache values
	cache[cacheIndex] = temp;

	//synchronize threas in this block
	__syncthreads();
	//Summing all entires in cache
	//Threards per block must be power of 2

	int i = blockDim.x/2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;

	}

	if(cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main(void) {
	float *a,*b,c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	
	//CPU allocation
	a = (float *)malloc(N*sizeof(float));
	b = (float *)malloc(N * sizeof(float));
	partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

	//Device allocation
	cudaMalloc((void **)&dev_a, N * sizeof(float));
	cudaMalloc((void **)&dev_b, N * sizeof(float));	
	cudaMalloc((void **)&dev_partial_c, blocksPerGrid * sizeof(float));

	//filling host memory with data
	int i=0;
	for (i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i*2;
	}

	cudaMemcpy(dev_a, a , N*sizeof(float) , cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float) , cudaMemcpyHostToDevice);

	dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a,dev_b,dev_partial_c);

	cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost);

	c=0;
	for (i = 0; i < blocksPerGrid; i++) {
		c+= partial_c[i];
	}


	printf("%.6g \n",c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	free(a); free(b); free(partial_c);

}
