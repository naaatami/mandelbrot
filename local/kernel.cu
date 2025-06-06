#include "mandelbrot.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

__global__ void findMandelbrotImage(Color* colors, int width, int height, MandelbrotConstant constants)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double xScaled = constants.xMin + x*(constants.xMax - constants.xMin) / width;
    double yScaled = constants.yMin + y*(constants.yMax - constants.yMin) / height;

    if (x < width && y < height) {
        cuDoubleComplex c = make_cuDoubleComplex(xScaled, yScaled);
        double iterationCount = calculateMandelbrot(c, constants.limit, constants.maxIterations);
        Color color = findColor(iterationCount, constants.maxIterations);
        colors[y * width + x] = color;
    }
}


Color* wrapper(int width, int height, MandelbrotConstant constants)
{
    Color* colors;
    cudaMallocManaged(&colors, width * height * sizeof(Color));
    dim3 blockDim(32, 32); //each block will have 32x32 threads
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    findMandelbrotImage<<<gridDim, blockDim>>>(colors, width, height, constants);
    cudaDeviceSynchronize(); //WE NEED TO WAIT, OR ELSE IT RETURNS GARBAGE DATA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    return colors;
} 


__device__ double calculateMandelbrot(cuDoubleComplex c, int limit, int maxIterations)
{
    int currentIterations = 0;
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);

    while(cuCabs(z) < limit and maxIterations > currentIterations)
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
