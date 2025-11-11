
all:
#	g++-15 -I$(shell python -m pylmcf -i) -std=c++20 -O3 -pg -g example.cpp
	g++ -I$(shell python -m pylmcf -i) -std=c++20 -Og -fsanitize=address -g example.cpp
