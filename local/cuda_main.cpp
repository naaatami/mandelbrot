#include "CImg.h"
#include "Color.h"
#include "MandelbrotConstant.h"
#include <complex.h>
#include <cmath>
#include <iostream>
#include <climits>
#include <iostream>
#include <vector>
#include "mandelbrot.cuh"
#include <chrono>
#include "time.h"

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
    if(argc != 4)
    {
        cout << "Incorrect usage. Use " << argv[0] << " <width> <height> <filename.png> \n";
        return 0;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    string filename = argv[3];
    CImg<float> mandelbrotImage(width, height, 1, 3, 0);

    clock_t start, end;
    double elapsed;
    start = clock();

    // initializing all mandelbrot constants
    MandelbrotConstant constants = MandelbrotConstant{xMin, xMax, yMin, yMax, maxIterations, limit};

    // init and process mandelbrot (kernel.cu)
    Color* colors;
    colors = wrapper(width, height, constants);

    // map colors to CImg output
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            mandelbrotImage.draw_point(x, y,
                vector<double>{
                    colors[y * width + x].h,
                    colors[y * width + x].s,
                    colors[y * width + x].v
                }.data()
            );
        }
    }

    mandelbrotImage.HSVtoRGB().save_png(filename.c_str());
    end = clock();
    elapsed = double(end - start)/CLOCKS_PER_SEC;
    cout << "Total time to find: " << elapsed << endl;
    


    return 0;
}

