# Compiler
NVCC = nvcc
CXX = g++

# Directories
SRC_DIR = src
INCLUDE_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Files
TARGET = $(BIN_DIR)/electrostatic_simulation
CUDA_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/particles.cu $(SRC_DIR)/visualization.cu
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
HEADERS = $(INCLUDE_DIR)/cuda_utils.h $(INCLUDE_DIR)/opengl_utils.h

# Compilation flags
# Paths
GLEW_INCLUDE = /usr/include
GLEW_LIB = /usr/lib/x86_64-linux-gnu

# Compilation flags
CUDA_FLAGS = -I$(INCLUDE_DIR) -I$(GLEW_INCLUDE) -L$(GLEW_LIB) -lGLEW -lGL -lGLU -lglut -Xcompiler -fopenmp
CXX_FLAGS = -I$(INCLUDE_DIR) -I/usr/local/lib -std=c++11 -O3
LDFLAGS = -lcuda -lcudart -lGL -lGLU -lglut -lGLEW

# Targets
all: $(TARGET)

$(TARGET): $(CUDA_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) -o $@ $^ $(CUDA_FLAGS) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

# Clean up object files and binary
clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean
