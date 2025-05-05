![image](https://github.com/user-attachments/assets/0d5e3759-b1a3-4cd6-8f6c-258e4557d83a)

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
