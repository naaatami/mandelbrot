#ifndef MANDELBROTCONSTANT_H
#define MANDELBROTCONSTANT_H

struct MandelbrotConstant {
    double xMin;
    double xMax;
    double yMin;
    double yMax;
    const int maxIterations;
    const int limit;
};

#endif
