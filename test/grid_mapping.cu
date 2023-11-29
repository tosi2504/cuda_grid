#include "../cugrid/cugrid.h"

#include <iostream>

const unsigned lenLane = 32;

int main () {
	Grid<lenLane> grid(8,4,4,4); // Vx:2 Vy:2 Vz:2 Vt:1

	unsigned mu = 1;
	bool isForward = false;
	auto mapping = grid.getStencilTargetInfoMap(mu, isForward);
	for (unsigned n = 0; n < grid.calcSizeVNodes(); n++) {
		printf("n: %u -> %u, %s\n", n, mapping[n].n_target, mapping[n].isBorder ? "true" : "false");
		flat coords = {n, 0};
		Neighbors neighbors = grid.getNeighbors(coords);
		printf("    n_target by neighbors: %u -> %u\n", n, neighbors[mu][isForward ? 0 : 1].n);
		cart ccoords = grid.toCart(coords);
		printf("    x, y, z, t: %u %u %u %u \n", ccoords.x, ccoords.y, ccoords.z, ccoords.t);
	}
}
