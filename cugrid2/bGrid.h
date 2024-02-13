#pragma once

#include "cugrid2.h"


struct bGrid {
	const unsigned Lx, Ly, Lz, Lt;
	const unsigned numSites;

	bGrid (unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt) : Lx(Lx), Ly(Ly), Lz(Lz), Lt(Lt), numSites(Lx*Ly*Lz*Lt) {}

	bool isCompatible(const bGrid & other) const {
		return Lx == other.Lx and Ly == other.Ly and Lz == other.Lz and Lt == other.Lt;
	}
};
