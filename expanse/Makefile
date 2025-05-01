CUDA_PATH       := /usr/local/cuda
CIMG_INC        := /usr/include
X11_LIB_PATH    := /usr/lib/x86_64-linux-gnu

CPPFLAGS        := -I$(CIMG_INC) -I$(CUDA_PATH)/include
LDFLAGS         := -L$(CUDA_PATH)/lib64 -L$(X11_LIB_PATH)
LDLIBS          := -lcudart -lcuda -lX11 -lm -lpthread

CUDA_ARCH       := sm_61

TARGET          := program
OBJS            := main.o kernel.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	nvcc $^ -o $@ $(LDFLAGS) $(LDLIBS)

# C++ compile with nvcc
main.o: cuda_main.cpp
	nvcc -g -x c++ -c $< -o $@ $(CPPFLAGS)

# CUDA compile
kernel.o: kernel.cu
	nvcc -g -c $< -o $@ -arch=$(CUDA_ARCH)

clean:
	rm -f $(OBJS) $(TARGET)

