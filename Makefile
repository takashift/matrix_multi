
#The following are example Makefile fragments for compiling and linking
#a host program against the host runtime libraries included with the 
#Intel(R) FPGA SDK for OpenCL(TM).


#Example GNU makefile on Linux, with GCC toolchain:

AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_LINK_CONFIG=$(shell aocl link-config)

matmulHPC169_exe : matmulHPC169.o
		gcc -o matmulHPC169_exe matmulHPC169.o -lm $(AOCL_LINK_CONFIG)

matmulHPC169.o : matmulHPC169.c
		gcc -c matmulHPC169.c $(AOCL_COMPILE_CONFIG)

