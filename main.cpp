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


int main(int argc, char *argv[])
{

    printf("okay");
    if(argc != 4)
    {
        cout << "Give me more arguments you loser!!\n";
        return 0;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    string filename = argv[3];
    printf("okay");
    CImg<float> mandelbrotImage(width, height, 1, 3, 0);


    
    Color* balls;
    balls = wrapper(balls, width, height, xMin, xMax, yMax, yMin, limit, maxIterations);
    printf("\nokay\n");
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            // mandelbrotImage.draw_point(x, y,
            //     vector<double>{
            //         balls[y * width + x].h,
            //         balls[y * width + x].s,
            //         balls[y * width + x].v
            //     }.data()
            // );

            printf("\n%f\n",balls[x*width + y].h);
        }
    }

    //mandelbrotImage.HSVtoRGB().save_png(filename.c_str());
    


    return 0;
}
