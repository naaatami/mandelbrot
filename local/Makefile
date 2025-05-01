CUDA_PATH       := /opt/cuda/targets/x86_64-linux
CIMG_INC        := /usr/include
X11_LIB_PATH    := /usr/X11R6/lib

CPPFLAGS        := -I$(CIMG_INC) -I$(CUDA_PATH)/include
LDFLAGS         := -L$(CUDA_PATH)/lib -L$(X11_LIB_PATH)
LDLIBS          := -lcudart -lcuda -lX11 -lm -lpthread

CUDA_ARCH       := sm_61

TARGET          := program

OBJS            := main.o kernel.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	g++ $^ -o $@ $(LDFLAGS) $(LDLIBS)

# C++ compile
main.o: cuda_main.cpp
	g++ -c $< -o $@ $(CPPFLAGS)

# CUDA compile
kernel.o: kernel.cu
	nvcc -c $< -o $@ -arch=$(CUDA_ARCH)

clean:
	rm -f $(OBJS) $(TARGET)
