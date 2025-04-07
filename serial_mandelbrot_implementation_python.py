"""
serial mandelbrot python implementation
Mason

Wanted to get a better grasp on the implementation before jumping into c++
No need for any special outputs, just enough to prove it is in fact calculating the mandelbrot set

If you want to make the increment value smaller to get a better image resolution, I'd recommend using the outfile instead of the terminal but that's just me
"""
import time
start = time.time() #get starting time to help time the program


boundary = 2 #set the boundary for the mandelbrot window, -2 -> 2 is the common one used
increment_value = 0.05 #make this smaller to get a better "resolution" for the ascii art output

real_lower_bound = boundary * -1 #use the previously defined boundary to get the boundaries for each real, imaginary, upper, and lower boundaries
real_upper_bound = boundary
imag_lower_bound = boundary * -1
imag_upper_bound = boundary
increment_count = int((boundary * 2) // increment_value) #calculate the number of increments required based on the boundary size and increment values

#define an output filepath and return the outfile
def get_file():
    output_file_path = "mandelbrot_out.txt"
    file = open(output_file_path, "w")
    return file

#checks if a given complex number is a part of the mandelbrot set
def check_mandelbrot_validity(c:complex):

    boundary = 2 #this one should stay the same, as it is commonly accepted to be the "point of no return" for mandelbrot calculations
    max_iterations = 100 #could increase this number, especially when demonstrating the power of multithreading

    z = complex(0, 0) #starting point for the mandelbrot set is always 0,0
    cur_iterations = 0 

    #while the z value is not more than 2 units from the origin (abs does vector math on complex numbers), and while we have not reached the designated number of iterations
    while(abs(z) < boundary and max_iterations > cur_iterations):
        z = (z**2) + c #square the Z value and add the c value
        cur_iterations += 1 #increment the iteration count
    if cur_iterations == max_iterations: #if we have reached the end of the iteration count and we have broken free of the loop, then number is part of the set
        return True
    return False #otherwise it is not part of the set

#prints the mandelbrot set in the terminal
def terminal_print_mandelbrot(mandelbrot:list):
    for line in mandelbrot:
        for val in line:
            if val:
                print("#", end=" ")
            else:
                print(" ", end=" ")
        print("")

#prints the mandelbrot set in the outfile
def write_mandelbrot_to_outfile(mandelbrot:list):
    file = get_file()
    for line in mandelbrot:
        for val in line:
            if val:
                file.write("# ")
            else:
                file.write("  ")
        file.write("\n")
    file.close()


mandelbrot = []

for i in range(increment_count):
    temp_list = []
    for ii in range(increment_count):
        c = complex((ii*increment_value)-boundary, (i*increment_value)-boundary)
        temp_list.append(check_mandelbrot_validity(c))
    mandelbrot.append(temp_list)


terminal_print_mandelbrot(mandelbrot)
#write_mandelbrot_to_outfile(mandelbrot)

end = time.time()
print(f"Runtime: {end-start}")
