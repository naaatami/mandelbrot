CUDA_PATH       := /usr/local/cuda
CIMG_INC        := /usr/include     # or wherever your CImg.h lives
X11_LIB_PATH    := /usr/X11R6/lib

CPPFLAGS        := -I$(CIMG_INC) -I$(CUDA_PATH)/include
LDFLAGS         := -L$(CUDA_PATH)/lib64 -L$(X11_LIB_PATH)
LDLIBS          := -lcudart -lcuda -lX11 -lm -lpthread

# adjust ARCH for your GPU (e.g. sm_50, sm_60, etc)
CUDA_ARCH       := sm_61

# your target executable
TARGET          := program

# object files
OBJS            := main.o kernel.o

.PHONY: all clean

all: $(TARGET)

# Link step: use g++ (or nvcc) to pull in both sets of libraries
$(TARGET): $(OBJS)
	g++ $^ -o $@ $(LDFLAGS) $(LDLIBS)

# C++ compile
main.o: main.cpp
	g++ -c $< -o $@ $(CPPFLAGS)

# CUDA compile
kernel.o: kernel.cu
	nvcc -c $< -o $@ -arch=$(CUDA_ARCH)

clean:
	rm -f $(OBJS) $(TARGET)
