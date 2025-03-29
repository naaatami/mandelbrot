#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <complex.h>
#include <cmath>
#include <iostream>
#include <climits>
#include <iostream>

double incrementValue = 0.05;
int upperBound = 2;
int lowerBound = upperBound * -1;
int incrementCount = ((int)upperBound * 2) / incrementValue;

void printMandelbrot(const std::vector<std::vector<bool>> &mandelbrot);
bool calculateMandelbrot(std::complex<double> c);

int main(int argc, char *argv[])
{

    std::vector<std::vector<bool>> mandelbrot(incrementCount, std::vector<bool>(incrementCount));

    for(int i = 0; i < incrementCount; i++)
    {
        for(int j = 0; j < incrementCount; j++)
        {
            std::complex<double> c((j*incrementValue) - upperBound, (i*incrementValue) - upperBound);
            mandelbrot[i][j] = calculateMandelbrot(c);
        }
    }

    printMandelbrot(mandelbrot);
}

bool calculateMandelbrot(std::complex<double> c)
{
    int maxIterations = 100;
    int currentIterations = 0;
    std::complex<double> z(0.0, 0.0);


    while(std::abs(z) < upperBound and maxIterations > currentIterations)
    {
        z = (z * z) + c;
        currentIterations++;
    }

    if(currentIterations == maxIterations)
    {
        return true;
    }
    return false;
}

void printMandelbrot(const std::vector<std::vector<bool>> &mandelbrot)
{
    for(int i = 0; i < mandelbrot.size(); i++)
    {
        for(int j = 0; j < mandelbrot[0].size(); j++)
        {
            if(mandelbrot[i][j])
            {
                std::cout << "# ";
            } else {
                std::cout << "  ";
            }
        }
        std::cout << "\n";
    }
}
