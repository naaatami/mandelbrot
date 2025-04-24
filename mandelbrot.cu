// nvcc -o mandelcu mandelbrot.cu

#include "mandelbrot.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


__global__ void findMandelbrotImage(Color* colors, int width, int height, double xMin, double xMax, double yMin, double yMax, int limit, int maxIterations){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double xScaled = xMin + x*(xMax - xMin) / width;
    double yScaled = yMin + y*(yMax - yMin) / height;

    if (x < width && y < height) {
        cuDoubleComplex c = make_cuDoubleComplex(xScaled, yScaled);
        double iterationCount = calculateMandelbrot(c, limit, maxIterations);
        Color color = findColor(iterationCount, maxIterations);
        colors[y * width + x] = color;
    }
}

namespace Wrapper {
	Color* wrapper(Color* colors, int width, int height, double xMax, double xMin, double yMax, double yMin, int limit, int maxIterations)
	{
        cudaMallocManaged(&colors, width * height * sizeof(double));
        int numThreads = 10; //change this obviously.
        int numBlocks = ceil((double)(width*height)/numThreads);
        // numBlocks = ur mom
		findMandelbrotImage<<<numBlocks, numThreads>>>(colors, width, height,xMin,xMax,yMin,yMax,limit, maxIterations);
        return colors;
	} 
}

__device__ double calculateMandelbrot(cuDoubleComplex c, int limit, int maxIterations)
{
    int currentIterations = 0;
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);

    while(fabsf(cuCreal(z)) < limit and maxIterations > currentIterations)
    {
        z = cuCadd(cuCmul(z,z), c);
        //z = (z * z) + c;
        currentIterations++;
    }

    if(currentIterations == maxIterations)
        return currentIterations;

    //double normalizedIterations = currentIterations + 1 - log2f(log(norm(z)) * 0.5);
    //double normalizedIterations = currentIterations + 1 - cuCmul(cuCabs(z),cuCabs(z));
    // return normalizedIterations;
    return currentIterations;
}


__device__ Color findColor(double iterationCount, int maxIterations)
{
    double baseColor = 360.0 * iterationCount / maxIterations;
    double baseColorPercent = (baseColor - 0.0) / (360.0 - 0.0);
    double smoothPercent = pow(baseColorPercent, 0.32);
    double scaledColor = ((360 - 200) * smoothPercent) + 200;

    double saturation;
    if (iterationCount == maxIterations)
    {
        saturation = 0;
    } else {
        saturation = 1.0;
    }
    Color color = Color{scaledColor, 1.0, saturation};

    return color;
}


double getTime(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec/1000000.0;
    
}

