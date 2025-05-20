![image](https://github.com/user-attachments/assets/0d5e3759-b1a3-4cd6-8f6c-258e4557d83a)

### Mandelbrot image generation
This program generates images of the Mandelbrot. This is an example output image:
<img src="https://github.com/user-attachments/assets/85a17387-5620-49cd-a393-92ca0c38c969" width=50% height=50%>
The two main subfolders, `expanse` and `local`, are for usage on the Expanse supercomputer versus a local home computer. The local folder will probably have what you need. Both folders contain three versions of the code. The first version, the serial version, simply calculates the Mandelbrot point by point. The MPI version parallelizes and splits the work between CPU cores, resulting in a speedup at a large enough size. The CUDA version also parallelizes the Mandelbrot generation using your GPU!

### How does it work?
The serial version simply generates the Mandelbrot as usual. Each point is calculated to see if it goes out to infinity or not, and if not, the point is colored black. Otherwise, the point is colored based on the amount of iterations it took to determine that it's not part of the set. The more iterations it took, the more pink it is colored.

The MPI version has each core calculate a different section of the final image. A new MPI datatype, MPI_COLOR_TYPE, was created in order for the information to be transmitted. Rank 0 then collects the data and generates the final image.

The CUDA version has four files - cuda_main.cpp, mandelbrot.cuh, kernel.cu, and Color.h. cuda_main.cpp is simply used to call kernel.cu, and handles the final image saving. mandelbrot.cuh is a header file for kernel.cu, and Color.h defines the color struct used across the other three files. kernel.cu is the meat of the CUDA section. The kernel function `findMandelbrotImage` finds the color for a point depending on the thread's coordinate in the block system. 

All versions additionally report the amount of time it took to calculate the Mandelbrot image.

### Differences between the local and Expanse version
- `save_png()` is commented out on Expanse, since it does not work 
- Normalization is not done on the MPI and serial version on Expanse, but is for the local version
- Expanse version has sbatch job scripts
- Makefile differences to account for different CUDA library locations - makefile may need tweaking to run locally. We had to change X11_LIB_PATH and CUDA_PATH. 
- Local version includes an extra Python mockup 

### How to run locally
**Serial version:**
- Compile with `g++ -o serial serial.cpp -L/usr/X11R6/lib -lm -lpthread -lX11`
- Run with `./serial <width> <height> <filename.png>`

**MPI version**
- Compile with `mpicxx -o mpi -g -Wall mpi.cpp -L/usr/X11R6/lib -lm -lpthread -lX11`
- Run with `mpiexec --n <processors> ./mpi <width> <height> <name>`

**CUDA version**
- Simply use the makefile: `make`
- Run with `./program <width> <height> <filename.png>`

### How to run on Expanse
**Serial version:**
- `module purge`
- `module load cpu/0.17.3b gcc/10.2.0 slurm`
- Compile with `g++ -o serial serial.cpp -L/usr/X11R6/lib -lm -lpthread -lX11`
- Submit `serialScript.sb`

**MPI version**
- `module purge`
- `module load cpu/0.17.3b gcc/10.2.0 openmpi/4.1.3 slurm`
- Compile with `mpicxx -o mpi -g -Wall mpi.cpp -L/usr/X11R6/lib -lm -lpthread -lX11`
- Submit `mpiScript.sb`

**CUDA version**
- `module purge`
- `module load gpu/0.17.3b slurm cuda11.7/toolkit/11.7.1`
- Simply use the makefile: `make`
- Submit `cudaScript.sb`

### Other notes
- CImg.h is only provided for user convenience in case it's not installed. If your system has it installed, it is not needed.
- Make sure your filename ends with .png - it will save as a PPM image instead otherwise, which your computer probably does not support! (This will be fixed eventually.)
