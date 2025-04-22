// nvcc -o mandelcu mandelbrot.cu

#include "mandelbrot.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


__global__ void findMandelbrotImage(Color* colors, int width, int height){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double xScaled = xMin + x*(xMax - xMin) / width;
    double yScaled = yMin + y*(yMax - yMin) / height;

    if (x < width && y < height) {
        cuDoubleComplex c = make_cuDoubleComplex(xScaled, yScaled);
        double iterationCount = calculateMandelbrot(c);
        Color color = findColor(iterationCount);
        colors[y * width + x] = color;
    }
}

namespace Wrapper {
	Color* wrapper(Color* colors, int width, int height)
	{
        colors = cudaMallocManaged(&colors, width * height * sizeof(double));
		findMandelbrotImage <<<1, 1>>> (colors, width, height);
        return colors;
	} 
}

__device__ double calculateMandelbrot(cuDoubleComplex c)
{
    int currentIterations = 0;
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);

    while(fabsf(cuCreal(z)) < limit and maxIterations > currentIterations)
    {
        
        z = (z * z) + c;
        currentIterations++;
    }

    if(currentIterations == maxIterations)
        return currentIterations;

    //double normalizedIterations = currentIterations + 1 - log2f(log(norm(z)) * 0.5);
    double normalizedIterations = __norm(z);
    return normalizedIterations;
}


__device__ Color findColor(double iterationCount)
{
    double baseColor = 360.0 * iterationCount / maxIterations;
    double baseColorPercent = (baseColor - 0.0) / (360.0 - 0.0);
    double smoothPercent = pow(baseColorPercent, 0.32);
    double scaledColor = ((360 - 200) * smoothPercent) + 200;

    Color color = Color(
        scaledColor,
        1.0,
        (iterationCount < maxIterations) ? 1.0 : 0
    );

    return color;
}


double getTime(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec/1000000.0;
    
}
