#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>

struct cartesian {
  unsigned x, y, z, t;
};

std::ostream & operator << (std::ostream & s, const cartesian & c) {
  return s<<"{"<<c.x<<","<<c.y<<","<<c.z<<","<<c.t<<"}";
}

struct bGrid {
	const unsigned Lx, Ly, Lz, Lt;
	const unsigned numSites;

	bGrid (unsigned Lx, unsigned Ly, unsigned Lz, unsigned Lt) : Lx(Lx), Ly(Ly), Lz(Lz), Lt(Lt), numSites(Lx*Ly*Lz*Lt) {}

	bool isCompatible(const bGrid & other) const {
		return Lx == other.Lx and Ly == other.Ly and Lz == other.Lz and Lt == other.Lt;
	}

	unsigned toFlat(const cartesian & c) const {
		return c.x*Ly*Lz*Lt + c.y*Lz*Lt + c.z*Lt + c.t;
	}
	cartesian toCartesian(const unsigned site) const {
		cartesian res;
		res.x = site/(Ly*Lz*Lt);
		res.y = (site%(Ly*Lz*Lt))/(Lz*Lt);
		res.z = (site%(Lz*Lt))/Lt;
		res.t = site%Lt;
		return res;
	}

  cartesian shift(const cartesian & c, unsigned mu, bool isForward) const {
    cartesian res(c);
    switch (mu) {
      case 0:
        res.x = isForward ? (c.x+1)%Lx : (c.x+Lx-1)%Lx;
        break;
      case 1:
        res.y = isForward ? (c.y+1)%Ly : (c.y+Ly-1)%Ly;
        break;
      case 2:
        res.z = isForward ? (c.z+1)%Lz : (c.z+Lz-1)%Lz;
        break;
      case 3:
        res.t = isForward ? (c.t+1)%Lt : (c.t+Lt-1)%Lt;
        break;
      default:
        throw std::invalid_argument("mu must be 0,1,2,3");
    }
    return res;
  }

  unsigned shift(unsigned site, unsigned mu, bool isForward) const {
    return toFlat(shift(toCartesian(site), mu, isForward));
  }

	std::vector<unsigned> calcTargetMap(const unsigned mu, bool isForward) const {
		std::vector<unsigned> res(numSites);
		for(unsigned site = 0; site < numSites; site++) {
			cartesian c = toCartesian(site);
			switch (mu) {
				case 0:
					c.x = isForward ? (c.x+1)%Lx : (c.x+Lx-1)%Lx;
					break;
				case 1:
					c.y = isForward ? (c.y+1)%Ly : (c.y+Ly-1)%Ly;
					break;
				case 2:
					c.z = isForward ? (c.z+1)%Lz : (c.z+Lz-1)%Lz;
					break;
				case 3:
					c.t = isForward ? (c.t+1)%Lt : (c.t+Lt-1)%Lt;
					break;
				default:
					throw std::invalid_argument("mu must be 0,1,2,3");
			}
			res[site] = toFlat(c);
		}
		return res;
	}
};
