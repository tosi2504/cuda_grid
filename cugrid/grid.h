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
struct virtualLayout<1U> {
	static constexpr unsigned Nx = 1;
	static constexpr unsigned Ny = 1;
	static constexpr unsigned Nz = 1;
	static constexpr unsigned Nt = 1;
};
template<>
struct virtualLayout<2U> {
	static constexpr unsigned Nx = 1;
	static constexpr unsigned Ny = 1;
	static constexpr unsigned Nz = 1;
	static constexpr unsigned Nt = 2;
};
template<>
struct virtualLayout<32U> {
	static constexpr unsigned Nx = 2;
	static constexpr unsigned Ny = 2;
	static constexpr unsigned Nz = 2;
	static constexpr unsigned Nt = 4;
};

struct cart {
	unsigned x, y, z, t;
};

struct flat {
	unsigned n, l;
	// n is flattened index in first vNode
	// l is flattened index of the vNode
};

struct Neighbors {
	flat data[4][2];
};

template<unsigned lenLane = 32>
class Grid {
	// okay so what do we need?
	// -> the lattice dimensions
	private:
	using vLayout = virtualLayout<lenLane>;
	
	public:
	const unsigned Lx, Ly, Lz, Lt; // extend of the physical lattice
	const unsigned Vx, Vy, Vz, Vt; // extend of one virtual node
	const unsigned vol;
	Grid(unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt):
		Lx(Lx), Ly(Ly), Lz(Lz), Lt(Lt), vol(Lx*Ly*Lz*Lt)
		, Vx(Lx/vLayout::Nx), Vy(Ly/vLayout::Ny), Vz(Lz/vLayout::Nz), Vt(Lt/vLayout::Nt) 
	{
		if (Lx % vLayout::Nx != 0 or Ly % vLayout::Ny != 0 or Lz % vLayout::Nz != 0 or Lt % vLayout::Nt != 0) {
			throw std::logic_error("Lattice dimensions not divisible by vNode layout");
		}
	}
	unsigned calcNumVNodes() const {
		if (vol % lenLane != 0) {
			throw std::logic_error("Grid dimensions not compatible with lenLane");
		}
		return vol / lenLane;
	}

	// for assert statements in lattice operations with two or more operants 
	bool operator == (const Grid<lenLane> & other) const {
		return Lx == other.Lx and Ly == other.Ly and Lz == other.Lz and Lt == other.Lt;
	}
	bool operator != (const Grid<lenLane> & other) const {
		return not ((*this) == other);
	}

	// geometrical structure of the grid
	__host__ __device__ inline flat toFlat(const cart & coords) const {
		flat res;
		res.n = (coords.x % Vx) * Vy*Vz*Vt + (coords.y % Vy) * Vz*Vt + (coords.z % Vz) * Vt + coords.t % Vt;
		res.l = (coords.x / Vx) * vLayout::Ny*vLayout::Nz*vLayout::Nt
				+ (coords.y / Vy) * vLayout::Nz*vLayout::Nt
				+ (coords.z / Vz) * vLayout::Nt
				+ (coords.t / Vt);
		return res;
	}

	__host__ __device__ inline cart toCart(const flat & coords) const {
		// okay, we want to map flat coords to cartesian ones
		// we get x, y, z, t within virtual nodes
		// and ofc vx, vy, vz, vt of virtual notes
		unsigned x, y, z, t, vx, vy, vz, vt;
		x = coords.n / (Vy*Vz*Vt);
		y = (coords.n - x * Vy*Vz*Vt) / (Vz*Vt);
		z = (coords.n - x * Vy*Vz*Vt - y * Vz*Vt) / Vt;
		t = coords.n -  x * Vy*Vz*Vt - y * Vz*Vt - z * Vt;
		vx = coords.l / (vLayout::Ny*vLayout::Nz*vLayout::Nt);
		vy = (coords.l - vx * vLayout::Ny*vLayout::Nz*vLayout::Nt) / (vLayout::Nz*vLayout::Nt);
		vz = (coords.l - vx * vLayout::Ny*vLayout::Nz*vLayout::Nt - vy * vLayout::Nz*vLayout::Nt) / vLayout::Nt;
		vt = coords.l -  vx * vLayout::Ny*vLayout::Nz*vLayout::Nt - vy * vLayout::Nz*vLayout::Nt - vz * vLayout::Nt;
		cart res;
		res.x = x + vx * Vx;
		res.y = y + vy * Vy;
		res.z = z + vz * Vz;
		res.t = t + vt * Vt;
		return res;
	}

	__host__ __device__ inline Neighbors getNeighbors(const cart & coords) {
		Neighbors neighbors;

		neighbors.data[0][0] = toFlat({(coords.x + 1)      % Lx, coords.y, coords.z, coords.t}); // positive x direction
		neighbors.data[0][1] = toFlat({(coords.x + Lx - 1) % Lx, coords.y, coords.z, coords.t}); // negative x direction

		neighbors.data[1][0] = toFlat({coords.x, (coords.y + 1)      % Ly, coords.z, coords.t}); // positive y direction
		neighbors.data[1][1] = toFlat({coords.x, (coords.y + Ly - 1) % Ly, coords.z, coords.t}); // negative y direction

		neighbors.data[2][0] = toFlat({coords.x, coords.y, (coords.z + 1)      % Lz, coords.t}); // positive z direction
		neighbors.data[2][1] = toFlat({coords.x, coords.y, (coords.z + Lz - 1) % Lz, coords.t}); // negative z direction

		neighbors.data[3][0] = toFlat({coords.x, coords.y, coords.z, (coords.t + 1)      % Lt}); // positive t direction
		neighbors.data[3][1] = toFlat({coords.x, coords.y, coords.z, (coords.t + Lt - 1) % Lt}); // negative t direction

		return neighbors;
	}
};


// okay so what we need is to calculate the layout of the virtual nodes
// this is not so trivial
// maybe just hardcode for lenLane = 32 atm?
// 
// also, what now
// I think we should just go ahead and program the lattice class
// Thats actually the hot shit 
