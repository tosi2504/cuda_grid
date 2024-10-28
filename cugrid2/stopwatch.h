#pragma once

#include <chrono>
#include <vector>
#include <string>

class Stopwatch {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    std::vector<int> diffs;

    int new_timestamp() {
        auto temp = std::chrono::high_resolution_clock::now();
        int res = std::chrono::duration_cast<std::chrono::microseconds>(temp - timestamp).count();
        timestamp = temp;
        return res;
    }
public:
    Stopwatch(): diffs(0) {
        timestamp = std::chrono::high_resolution_clock::now();
    }

    void reset() {
        diffs.resize(0);
        timestamp = std::chrono::high_resolution_clock::now();
    }
    void press() {
        diffs.push_back(new_timestamp());
    }
    int getdiff(unsigned i) {
        return diffs[i];
    }
    unsigned getSize() const {
        return diffs.size();
    }
};

Stopwatch stopwatch;
