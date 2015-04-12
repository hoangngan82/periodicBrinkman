CPP = nvcc
CPPFLAGS13 = -arch=sm_13 -ftz=true -prec-div=false -prec-sqrt=false
CPPFLAGS20 = -arch=sm_20 -ftz=false -prec-div=true -prec-sqrt=true
CPPFLAGS30 = -arch=sm_30 -ftz=false -prec-div=true -prec-sqrt=true --ptxas-options="-v"
CPPFLAGS = $(CPPFLAGS30) -O3
Brinkman: periodicBrinkman.cu periodicBrinkman.h 
	$(CPP) $(CPPFLAGS) $< -o /tmp/hoangngan/periodicBrinkman

test: matrixtest.cu periodicBrinkman.h
	$(CPP) $(CPPFLAGS) $< -o flagellum

optNum: optNumShells.cu periodicBrinkman.h
	$(CPP) $(CPPFLAGS) $< -o /tmp/hoangngan/optNumShells

runmatrix: 
	optirun --no-xorg cuda-memcheck /tmp/hoangngan/matrixtest

clean:
	rm -f *.o
