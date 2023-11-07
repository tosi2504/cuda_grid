#pragma once

#include <stdexcept>


// a grid class that does all the geometrical things
// we use the interleaved memory model and the grid class needs to know how all that comes about
// the Grid class needs to split the lattice into # of lenLane subnodes



// here we go, mario!

// a metafunction to calculate the layout of the virtual nodes -> yaaaay

template<unsigned lenLane>
struct virtualLayout {
	// only template specilizations shall compile 
	static_assert(lenLane == 0, "lenLane is not supported ... try 32");
}; 

template<>
struct virtualLayout<32U> {
	static constexpr unsigned Nx = 2;
	static constexpr unsigned Ny = 2;
	static constexpr unsigned Nz = 2;
	static constexpr unsigned Nt = 4;
};

template<unsigned lenLane = 32>
class Grid {
	// okay so what do we need?
	// -> the lattice dimensions
	private:
	const unsigned Lx, Ly, Lz, Lt;
	const unsigned vol;
	using vLayout = virtualLayout<lenLane>;
	
	public:
	Grid(unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt):
		Lx(Lx), Ly(Ly), Lz(Lz), Lt(Lt), vol(Lx*Ly*Lz*Lt)
	{}
	unsigned calcLatticeBufferSize() const {
		if (vol % lenLane != 0) {
			throw std::logic_error("Grid dimensions not compatible with lenLane");
		}
		return vol / lenLane;
	}
	bool operator == (const Grid<lenLane> & other) const {
		return Lx == other.Lx and Ly == other.Ly and Lz == other.Lz and Lt == other.Lt;
	}
	bool operator != (const Grid<lenLane> & other) const {
		return not ((*this) == other);
	}

};


// okay so what we need is to calculate the layout of the virtual nodes
// this is not so trivial
// maybe just hardcode for lenLane = 32 atm?
// 
// also, what now
// I think we should just go ahead and program the lattice class
// Thats actually the hot shit 
