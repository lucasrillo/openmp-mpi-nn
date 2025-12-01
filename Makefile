# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS
    LLVM = /opt/homebrew/opt/llvm
    CC = $(LLVM)/bin/clang
    FLAGS = -Wall -Werror -fopenmp -I$(LLVM)/include
else
    # Linux (and other Unix-like systems)
    CC = gcc
    FLAGS = -Wall -Werror -fopenmp
endif

SRC_DIR = src
BUILD_DIR = build
SRC = $(wildcard $(SRC_DIR)/*.c) \
      $(wildcard $(SRC_DIR)/*/*.c)
OBJ = $(SRC:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

all: main.exe

main.exe: $(OBJ)
	$(CC) $(FLAGS) -o $@ $^ -lm

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) main.exe

run: main.exe
	./main.exe