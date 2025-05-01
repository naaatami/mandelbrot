"""
serial mandelbrot python implementation

Wanted to get a better grasp on the implementation before jumping into c++

updated to generate images instead of the ascii art output
will need to play around with the coloing at some point but it works for now

"""
import time
from PIL import Image
import numpy as np
from math import log, log2

start = time.time()


real_number_resolution = 500
imag_number_resolution = 500

real_min = -2.1
real_max = 0.6
imag_min = -2
imag_max = 2

real_increment_value = (real_max - real_min) / real_number_resolution
imag_increment_value = (imag_max - imag_min) / imag_number_resolution
max_iterations = 100
max_color_value = 256

"""
takes a complex number and returns the number of iterations required to reach 
the boundary or returns 100 if complex number is considered part of the mandelbrot set
"""
def get_mandelbrot_iterations(c:complex)->int:
    boundary = 2  #generally accepted boundary for mandelbrot calculations
    z = complex(0, 0)
    cur_iterations = 0
    while(abs(z) < boundary and max_iterations > cur_iterations):
        z = (z**2) + c
        cur_iterations += 1

    #normalized = normalize_iteration_value(cur_iterations, z)
    return cur_iterations, z

#used to make the output picture look nicer and remove the "waves" that sometimes appear
def normalize_iteration_value(iteration_count, z):
    if(iteration_count == max_iterations):
        return 0
    

#used to make the output picture look nicer, original function created by Natalie
def find_color(iteration_count, z):
    if (iteration_count == max_iterations):
        return (0, 0, 0)  # Black for points inside the Mandelbrot set
    iteration_count = iteration_count + 1 - log2(log(abs(z)) * 0.5)
    t = iteration_count / max_iterations
    r = max(0, int(9 * (1 - t) * t**3 * 255))
    g = max(0, int(15 * (1 - t)**2 * t**2 * 255))
    b = max(0, int(8.5 * (1 - t)**3 * t * 255))
    
    return (r, g, b)



image_array = np.zeros((imag_number_resolution, real_number_resolution, 3), dtype=np.uint8)

cur_imag_val = imag_min #loop through the imaginary values until we hit the max imaginary value
for imag in range(imag_number_resolution):
    for real in range(real_number_resolution):
        c = complex((real * real_increment_value) + real_min, (imag * imag_increment_value) + imag_min)
        normalized_iteration_count, z = get_mandelbrot_iterations(c)
        image_array[imag][real] = find_color(normalized_iteration_count, z)

end = time.time()
print(f"Runtime: {end-start}")


image = Image.fromarray(image_array)
image.show()
