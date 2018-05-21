/*#include <stdio.h>

#define INF 2e10f;

struct Sphere {
	float r,b,g; //sphere color in RGB
	float radius; //radius
	float x,y,z; //center position

	__device__ float hit(float ox, float oy, float *n) {
		/*Verifies if a straight ray coming from ox,oy hist the sphere,
		returns the z coordinate of the hit.
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dx < radius*radius) {
			//If hits
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius*radius);
			return dz + z;
		}

		return -INF;
	}

}; */