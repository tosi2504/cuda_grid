#pragma once

#include "bLattice.h"
#include <chrono>
#include <type_traits>
#include <iostream>

template<class T, unsigned N>
bVectorField<T,N> ** createBatchVecFields(const unsigned numRHS, const bGrid & grid) {
	bVectorField<T,N> ** res = new bVectorField<T,N>*[numRHS];
	for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) res[iRHS] = new bVectorField<T,N>(grid);
	return res;
}
template<class T, unsigned N>
bVectorField<T,N> ** createAndFillAndUploadBatchVecFields(const unsigned numRHS
					, const bGrid & grid
					, std::mt19937 & gen
					, T min, T max) {
	bVectorField<T,N> ** res = new bVectorField<T,N>*[numRHS];
	for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) { 
		res[iRHS] = new bVectorField<T,N>(grid);
		res[iRHS]->fill_random(gen, min, max);
		res[iRHS]->upload();
	}
	return res;
}

template<class T, unsigned N>
void downloadBatchVecFields(unsigned numRHS, bVectorField<T, N> ** fields) {
    for (unsigned iRHS = 0; iRHS < numRHS; iRHS++) {
        fields[iRHS]->download();
    }
}
