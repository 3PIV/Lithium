#include <metal_stdlib>

using namespace metal;

float getSeed(const unsigned seed1, const unsigned seed2 = 2, const unsigned seed3 = 3);
float rand(thread float &seed);
