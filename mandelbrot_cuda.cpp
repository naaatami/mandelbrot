// compile with g++ mandelbrot.cpp -L/usr/X11R6/lib -lm -lpthread -lX11
// also make sure cimg is installed
// usage:
#include "CImg.h"
#include "Color.h"
#include <complex.h>
#include <cmath>
#include <iostream>
#include <climits>
#include <iostream>
#include <vector>
#include "mandelbrot.cuh"

double xMin = -2.1;
double xMax = 0.6;
double yMin = -1.2;
double yMax = 1.2;
const int maxIterations = 100;
const int limit = 4;

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

    Color* balls;
    balls = Wrapper::wrapper(balls, width, height, xMin, xMax, yMax, yMin, limit, maxIterations);

    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            mandelbrotImage.draw_point(x, y,
                vector<double>{
                    balls[y * width + x].h,
                    balls[y * width + x].s,
                    balls[y * width + x].v
                }.data()
            );
        }
    }

    mandelbrotImage.HSVtoRGB().save_png(filename.c_str());
    


    return 0;
}
