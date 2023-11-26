#include "../cugrid/cugrid.h"


#include <iostream>


int main () {
	Grid<32> grid(6,6,6,8); // Vx:2 Vy:2 Vz:2 Vt:1
	cart coords = {3,3,3,3};
	Neighbors neighbors = grid.getNeighbors(coords);
	for (int mu = 0; mu < 4; mu++) {
		for (int sign = 0; sign < 2; sign++) {
			cart nb = grid.toCart(neighbors.data[mu][sign]);
			printf("mu: %d, sign: %d    :    %u %u %u %u\n", mu, sign, nb.x, nb.y, nb.z, nb.t);
		}
	}
}
