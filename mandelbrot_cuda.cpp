// compile with g++ mandelbrot.cpp -L/usr/X11R6/lib -lm -lpthread -lX11
// also make sure cimg is installed
// usage:
#include "CImg.h"
#include "Color.h"
#include "cuda.cu"
#include <complex.h>
#include <cmath>
#include <iostream>
#include <climits>
#include <iostream>
#include <vector>
#include "mandelbrot.cu"

using namespace cimg_library;
using namespace std;

double calculateMandelbrot(complex<double> c);
vector<double> findColor(double iterationCount);

int main(int argc, char *argv[])
{
    if(argc != 4)
    {
        cout << "Give me more arguments you loser!!\n";
        return 0;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    string filename = argv[3];

    CImg<float> mandelbrotImage(width, height, 1, 3, 0);

    // for(int x = 0; x < width; x++)
    // {
    //     for(int y = 0; y < height; y++)
    //     {
    //         double xScaled = xMin + x*(xMax - xMin) / width;
    //         double yScaled = yMin + y*(yMax - yMin) / height;

    //         complex<double> c(xScaled, yScaled);
    //         double iterationCount = calculateMandelbrot(c);
    //         vector<double> color = findColor(iterationCount);
    //         
    //     }
    // }

    color* balls;
    balls = Wrapper::wrapper(width, height);

    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            mandelbrotImage.draw_point(x, y, balls[y * width + x].data());
        }
    }

    mandelbrotImage.HSVtoRGB().save_png(filename.c_str());
    


    return 0;
}
