#include <stdio.h>
#include "../common/book.h"
#include "../common/cpu_anim.h"


//Global
#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f 

//setting texture references memory in GPU
texture<float> texConstSrc;
texture<float> texIn;
texture<float> texOut;

//Global

//KKKKKKKKKKKKKKKKKKKKKKKKKKKKK
__global__ void copy_const_kernel(float *iptr) {
	/*Given a grid to input temperatures iptr, copies the constant temperatures in the texture texConstSrc
	to the pointer iptr.*/

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y *blockDim.x*gridDim.x;
	//Giving x,y unique values and linearizing it in offset

	//The constants are stored in the texConstSrc texture
	float c = tex1Dfetch(texConstSrc, offset);

	if (c != 0) iptr[offset] = c;


}

__global__ void blend_kernel(float *dst, bool dstOut) {
	/*Using texture memory we cant use directly the pointers to the memory, we will use a boolean
	flag instead to know which memory to use as input or output, sending the result through dst*/

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y *blockDim.x*gridDim.x;


	//setting neighboorhod
	int left = offset - 1;
	int right = offset + 1;
	if (x == 0) left++;
	if (x == DIM - 1) right--;

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0) top += DIM;
	if (y == DIM - 1) bottom -= DIM;

	float t,l,c,r,b;
	if (dstOut) { //if dstOut is TRUE
		t = tex1Dfetch(texIn,top);
		l = tex1Dfetch(texIn, left);
		c = tex1Dfetch(texIn, offset);
		r = tex1Dfetch(texIn, right);
		b = tex1Dfetch(texIn, bottom);
	} //We take the input from the texture texIn
	else { //ELSE
		t = tex1Dfetch(texOut, top);
		l = tex1Dfetch(texOut, left);
		c = tex1Dfetch(texOut, offset);
		r = tex1Dfetch(texOut, right);
		b = tex1Dfetch(texOut, bottom);
	} //We Take the input from texOut


	dst[offset] = c + SPEED*(t+b+r+l-4.0*c);

}
//KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK

//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
struct DataBlock {
	//Setting bitmaps and grid of temperature
	unsigned char *output_bitmap;
	float *dev_inSrc; //Input
	float *dev_outSrc; //Output
	float *dev_constSrc; //Constant Sources
	CPUAnimBitmap	*bitmap;

	//Setting time variables to mesure time taken.
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

void anim_gpu(DataBlock *d, int ticks) {
	cudaEventRecord(d->start, 0); //start recording time

								  //blocks and grids dimension
	dim3	blocks(DIM / 16, DIM / 16);
	dim3	threads(16, 16);
	//starting bitmap from the DataBlock
	CPUAnimBitmap	*bitmap = d->bitmap;

	//Make heat flow 90 times
	//since tex is global and bound we set up a boolean to flag which to use
	volatile bool dstOut = true;
	for (int i = 0; i < 90; i++) {
		//Setting input and output
		float *in , *out;
		if (dstOut) { //if True
			in = d->dev_inSrc;
			out = d->dev_outSrc;
		} //We take input from the expected
		else { //Else
			out = d->dev_inSrc;
			in = d->dev_outSrc;
		} //We change the order

		copy_const_kernel<<<blocks, threads>>>(in);
		blend_kernel<<<blocks,threads>>>(out , dstOut);
		dstOut = !dstOut;

	}
	//Make bitmap from output
	float_to_color << <blocks, threads >> >(d->output_bitmap, d->dev_inSrc);
	//copy bitmap to host
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

	//get and print time
	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);

	d->totalTime += elapsedTime;
	++d->frames;

	printf("Avarage Time Per Frame %3.1f ms \n", d->totalTime / d->frames);

}


void anim_exit(DataBlock *d) {
	/*Cleans and wraps datablock*/

	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConstSrc);

	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}
//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


int main(void) {
	//Setting Bitmaps
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;

	//Setting timeers = 0
	data.totalTime = 0;
	data.frames = 0;
	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);



	//Allocating space to bitmap and sources.
	cudaMalloc((void **)&data.output_bitmap, bitmap.image_size());

	cudaMalloc((void **)&data.dev_inSrc, bitmap.image_size());
	cudaMalloc((void **)&data.dev_outSrc, bitmap.image_size());
	cudaMalloc((void **)&data.dev_constSrc, bitmap.image_size());

	//Bind texture reference to memory allocate in the device with the designed name (pg 126)
	cudaBindTexture(NULL, texConstSrc, data.dev_constSrc, bitmap.image_size());
	cudaBindTexture(NULL, texIn, data.dev_inSrc, bitmap.image_size());
	cudaBindTexture(NULL, texOut, data.dev_outSrc, bitmap.image_size());


	//setting heaters in temp
	float *temp = (float *)malloc(bitmap.image_size());
	int i = 0;
	for (i = 0; i < DIM*DIM; i++) {
		temp[i] = 0;
		int x = i%DIM;
		int y = i / DIM;

		if ((x > 300) && (x < 600) && (y > 310 && (y < 601)))
			temp[i] = MAX_TEMP;
	}

	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;

	for (int y = 800; y < 900; y++) {
		for (int x = 400; x < 500; x++) {
			temp[x + y*DIM] = MIN_TEMP;
		}
	}

	cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);

	for (int y = 800; y < DIM; y++) {
		for (int x = 0;x < 200; x++) {
			temp[x + y*DIM] = MAX_TEMP;
		}
	}

	cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);

	free(temp);

	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);

	return 0;
}