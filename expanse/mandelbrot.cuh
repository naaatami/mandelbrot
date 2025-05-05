#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Color.h"
#include "MandelbrotConstant.h"
#include <stdio.h>
#include <cuComplex.h>

__device__ double calculateMandelbrot(cuDoubleComplex c, int limit, int maxIterations);
__global__ void findMandelbrotImage(Color* colors, int width, int height, MandelbrotConstant* constants);
__device__ Color findColor(double iterationCount, int maxIterations);
Color* wrapper(int width, int height, MandelbrotConstant constants);
