#include "complex.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include <time.h>

#define DIM 1000

int julia(int x, int y); //Verifies if x,y is in julia set

void kernel(unsigned char *ptr); //set bitmap colors right

int main(void) {
	clock_t begin = clock();
	CPUBitmap bitmap(DIM,DIM); //Creates bitmap
	unsigned char *ptr = bitmap.get_ptr(); //Set pointer to bitmap

	kernel(ptr);

	bitmap.display_and_exit();

	clock_t end = clock();
	double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
	printf("Tempo gasto %lf \n", time_spent);

	return 0;
}

int julia(int x, int y) {
	const float scale = 1.3;
	float jx = scale*(float)(DIM/2-x)/(DIM/2);
	float jy = scale*(float)(DIM/2 - y)/(DIM/2); //Positions relative to the grid

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx,jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a*a +c;
		if (a.magnitude2()>1000)
			return 0;
	}

	return 1;


}

void kernel(unsigned char *ptr) {
	int y,x;
	for (y = 0; y < DIM; y++) {
		for (x = 0; x < DIM;x++) {
			int offset =x+y*DIM;

			int juliaValue = julia(x,y);
			//printf("%d \t", juliaValue);
			ptr[offset*4+0] = 122*juliaValue;
			ptr[offset * 4 + 1] = 0 * juliaValue;
			ptr[offset * 4 + 2] = 122*juliaValue;
			ptr[offset * 4 + 3] = 255;
		}
	}
	
}