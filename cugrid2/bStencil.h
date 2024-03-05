#pragma once




struct bMuStencil {
	const bGrid grid;
	const unsigned mu;
	bMuStencil(const bGrid & grid, const unsigned mu) : grid(grid), mu(mu) {
		
	}
	~bMuStencil() {}
}
