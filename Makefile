#-lcuda 问题： WSL中不需要安装nvidia驱动，通过特殊的配置使得linux可以访问windows中的nvidia驱动。
#这导致cuda libraries（即lcuda）被储存在特殊的 \usr\lib\wsl\lib 中，
#因此我们可以 cp \usr\lib\wsl\lib\* \usr\lib 把所需的cuda库放到gcc可以找到的地方。

CC = mpic++ -O3 -ffast-math
NVCC = nvcc -O3  -arch=compute_86 -code=sm_86

LD = mpic++ -O3
LIBS = -lcusparse -lcudart -lcufft -lcublas -lcuda  -lstdc++ -lm -lhdf5  -lhdf5 -lhdf5_hl -lconfig
PATHS = -L/usr/local/cuda/lib64/ -L/usr/lib  -L/usr/lib64 -L/usr/local/hdf5/lib -L/usr/local/lib

INCLUDES = -I/usr/local/cuda/include -I/usr/local/hdf5/include -I/usr/local/include
DEBUG = 
NX:= $(shell python stripsizes.py NX)
NY:= $(shell python stripsizes.py NY)
NZ:= $(shell python stripsizes.py NZ)
SIZE = -DNX=${NX} -DNY=${NY} -DNZ=${NZ}
GPU_SOURCES = $(wildcard src/*.cu)
CPU_SOURCES = $(wildcard src/*.c)
GPU_OBJECTS = $(GPU_SOURCES:.cu=.o)
CPU_OBJECTS = $(CPU_SOURCES:.c=.o)


all: $(GPU_OBJECTS) $(CPU_OBJECTS)
	$(LD) -o channelMPI.bin $(CPU_OBJECTS) $(GPU_OBJECTS) $(PATHS) $(LIBS) $(DEBUG)

$(CPU_OBJECTS): src/%.o: src/%.c
	$(CC) -c $(INCLUDES) $(SIZE) $(PATHS) $(DEBUG)  $< -o $@

$(GPU_OBJECTS): src/%.o: src/%.cu
	$(NVCC) -c $(INCLUDES) $(SIZE) $(PATHS) $(DEBUG)  $< -o $@

tools: interpolator.o meanpolator.o
	$(CC) $(PATHS) $(LIBS) tools/interpolator.o -g -o interpolator.bin
	$(CC) $(PATHS) $(LIBS) tools/meanpolator.o -g -o meanpolator.bin

interpolator.o:
	$(CC) $(INCLUDES) $(SIZE) -c tools/interpolator.c -g -o tools/interpolator.o

meanpolator.o:
	$(CC) $(INCLUDES) $(SIZE) -c tools/meanpolator.c -g -o tools/meanpolator.o


clean:
	rm src/*.o channelMPI.bin
