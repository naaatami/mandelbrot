#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Color.h"
#include <stdio.h>
#include <cuComplex.h>


__device__ double calculateMandelbrot(cuDoubleComplex c, int limit, int maxIterations);
__global__ void findMandelbrotImage(Color* colors, int width, int height, double xMin, double xMax, double yMin, double yMax);
__device__ Color findColor(double iterationCount, int maxIterations);


Color* wrapper(Color* colors, int width, int height, double xMin, double xMax, double yMin, double yMax, int limit, int maxIterations);
