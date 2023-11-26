#include "../cugrid/cugrid.h"


#include <iostream>


int main () {
	Grid<32> grid(4,4,4,4); // Vx:2 Vy:2 Vz:2 Vt:1
	cart coords = {1,2,0,0};
	flat fcoords = grid.toFlat(coords);
	std::cout << grid.Vx << std::endl;
	std::cout << grid.Vy << std::endl;
	std::cout << grid.Vz << std::endl;
	std::cout << grid.Vt << std::endl;
	std::cout << "n: " << fcoords.n << std::endl;
	std::cout << "l: " << fcoords.l << std::endl;
	cart ccoords = grid.toCart(fcoords);
	std::cout << ccoords.x << " " << ccoords.y << " " << ccoords.x << " " << ccoords.t << std::endl;
}
