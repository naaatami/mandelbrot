"""
serial mandelbrot python implementation
Mason

Wanted to get a better grasp on the implementation before jumping into c++
No need for any special outputs, just enough to prove it is in fact calculating the mandelbrot set

If you want to make the increment value smaller to get a better image resolution, I'd recommend using the outfile instead of the terminal but that's just me
"""
import time
start = time.time()


boundary = 2
increment_value = 0.05 #make this smaller to get a better "resolution" for the ascii art output

real_lower_bound = boundary * -1
real_upper_bound = boundary
imag_lower_bound = boundary * -1
imag_upper_bound = boundary
increment_count = int((boundary * 2) // increment_value)


def get_file():
    output_file_path = "mandelbrot_out.txt"
    file = open(output_file_path, "w")
    return file


def check_mandelbrot_validity(c:complex):

    boundary = 2
    max_iterations = 100

    z = complex(0, 0)
    cur_iterations = 0

    while(abs(z) < boundary and max_iterations > cur_iterations):
        z = (z**2) + c
        cur_iterations += 1
    if cur_iterations == max_iterations:
        return True
    return False

def terminal_print_mandelbrot(mandelbrot:list):
    for line in mandelbrot:
        for val in line:
            if val:
                print("#", end=" ")
            else:
                print(" ", end=" ")
        print("")

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
