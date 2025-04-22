#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Color.h"
#include <stdio.h>
#include <cuComplex.h>

// these values are all set a bit weird to crop better and add some margins - parts are cut off if you round these off to "nicer" numbers
double xMin = -2.1;
double xMax = 0.6;
double yMin = -1.2;
double yMax = 1.2;

const int maxIterations = 100;
const int limit = 4;


__device__ double calculateMandelbrot(cuDoubleComplex c);
__global__ void findMandelbrotImage(Color* colors, int width, int height);
__device__ Color findColor(double iterationCount);

namespace Wrapper {
    Color* wrapper(Color* colors, int width, int height);
}