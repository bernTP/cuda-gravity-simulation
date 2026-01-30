
SRC	= $(wildcard ./*.cu)
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -O3 -arch=sm_75
OBJ	=	$(SRC:.cu=.o)

NAME	=	nbody_sim

all: $(NAME)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(NAME):	 $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(NAME)

clean:
	rm -f $(OBJ)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all clean fclean re
