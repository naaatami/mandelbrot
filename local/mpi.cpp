// compile with mpicxx -g -Wall mpi.cpp -L/usr/X11R6/lib -lm -lpthread -lX11
// run with mpiexec --n <processors> ./a.out <width> <height> <name>
// also make sure cimg is installed
// usage:
#include "CImg.h"
#include <complex.h>
#include <cmath>
#include <iostream>
#include <climits>
#include <vector>
#include <mpi.h>
#include <tuple>
using namespace cimg_library;
using namespace std;

const int maxIterations = 100;
const int limit = 4;

struct Color {
    double h;
    double s;
    double v;
};

// these values are all set a bit weird to crop better and add some margins - parts are cut off if you round these off to "nicer" numbers
double xMin = -2.1;
double xMax = 0.6;
double yMin = -1.2;
double yMax = 1.2;

double calculateMandelbrot(complex<double> c);
Color findColor(double iterationCount);
MPI_Datatype create_mpi_color_type();

int main(int argc, char *argv[])
{

    int rank, numberOfProcessors;
    MPI_Init(&argc, &argv);
    MPI_Comm mainComm = MPI_COMM_WORLD;
    MPI_Comm_size(mainComm, &numberOfProcessors);
    MPI_Comm_rank(mainComm, &rank);
    int width, height;
    string filename;
    Color* gatheredColors = nullptr;

    if(rank == 0)
    {
        if(argc != 4)
        {
            cout << "Incorrect usage. Use " << argv[0] << " <width> <height> <filename.png> \n";
            return 0;
        }

        width = atoi(argv[1]);
        height = atoi(argv[2]);
        filename = argv[3];
        gatheredColors = new Color[width * height];
    }

    double startTime, elapsedTime;
    MPI_Barrier(mainComm);
    startTime = MPI_Wtime();

    MPI_Bcast(&width, 1, MPI_INT, 0, mainComm);
    MPI_Bcast(&height, 1, MPI_INT, 0, mainComm);

    int localHeight = height/numberOfProcessors; //range of workelapsedTime
    int startHeight = rank * localHeight;    //where it starts

    Color* localColorArray = new Color[width * localHeight];
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < localHeight; y++)
        {
            double xScaled = xMin + x*(xMax - xMin) / width;
            double yScaled = yMin + (y+startHeight) * (yMax - yMin) / height;
            complex<double> c(xScaled, yScaled);
            double iterationCount = calculateMandelbrot(c);
            localColorArray[y * width + x] = findColor(iterationCount);

        }
    }


    MPI_Datatype MPI_COLOR_TYPE = create_mpi_color_type();
    MPI_Gather(localColorArray, width * localHeight, MPI_COLOR_TYPE, gatheredColors, width * localHeight, MPI_COLOR_TYPE, 0, mainComm);
    delete[] localColorArray;

    

    if(rank == 0)
    {
        CImg<float> mandelbrotImage(width, height, 1, 3, 0);

        for (int x = 0; x < width; x++) {
         for (int y = 0; y < height; y++) {
         mandelbrotImage.draw_point(x, y,
          vector<double>{
             gatheredColors[y * width + x].h,
               gatheredColors[y * width + x].s,
                 gatheredColors[y * width + x].v
             }.data()
             );
           }
         }
        delete[] gatheredColors;
        
        mandelbrotImage.HSVtoRGB().save_png(filename.c_str());
        elapsedTime = MPI_Wtime() - startTime;
        cout << "Total time to find: " << elapsedTime << endl;
    }

    MPI_Finalize();
    return 0;
}

// calculates the total iterations for a given complex number
// note that this returns a double, not an int! it's normalizing the iterationcount, which makes it a double
double calculateMandelbrot(complex<double> c)
{

    int currentIterations = 0;
    complex<double> z(0.0, 0.0);

    while(abs(z) < limit and maxIterations > currentIterations)
    {
        z = (z * z) + c;
        currentIterations++;
    }

    if(currentIterations == maxIterations)
        return currentIterations;

    // you could just return currentIterations here instead of doing this, and it'd still work fine.
    // doing this normalization thing just gets rid of the color banding and makes it look better!
    // https://linas.org/art-gallery/escape/smooth.html
    double normalizedIterations = currentIterations + 1 - log2(log(norm(z)) * 0.5);
    return normalizedIterations;
}

// the color representation in color[] is HSV (hue, saturation, value)
// hue can range from 0 to 360
// saturation can range from 0 to 1 (white to fully satured)
// value also ranges from 0 to 1 (black to bright)
// ---- okay, so how does this work? ----
// basecolor finds the original color in the range of 0 to 360
// we want to have fun, so we change that into a percent so we can then change it to the range 200 - 360 (which makes it blue -> pink)
// then we do smoothPercent because otherwise, we have a VERY strong change from blue to pink. this smooths it out

Color findColor(double iterationCount)
{
    double baseColor = 360.0 * iterationCount / maxIterations;
    double baseColorPercent = (baseColor - 0.0) / (360.0 - 0.0);
    double smoothPercent = pow(baseColorPercent, 0.32);
    double scaledColor = ((360 - 200) * smoothPercent) + 200;

    Color color = {
        scaledColor,
        1.0, //max saturation
        (iterationCount < maxIterations) ? 1.0 : 0 //if part of set, make black
    };

    return color;
}

MPI_Datatype create_mpi_color_type(){

    MPI_Datatype MPI_COLOR_TYPE;

    int count = 3;
    int block_lengths[3] = {1, 1, 1};
    MPI_Datatype old_types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    MPI_Aint indices[3];
    indices[0] = (MPI_Aint) offsetof(struct Color, h);
    indices[1] = (MPI_Aint) offsetof(struct Color, s);
    indices[2] = (MPI_Aint) offsetof(struct Color, v);


    MPI_Type_create_struct(count, block_lengths, indices, old_types, &MPI_COLOR_TYPE);
    MPI_Type_commit(&MPI_COLOR_TYPE);

    return MPI_COLOR_TYPE;
}
