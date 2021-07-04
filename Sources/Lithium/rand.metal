#include <metal_stdlib>

using namespace metal;

// This is a non object oriented version of Loki. If this interests you, check it out here!
// https://github.com/Joe-Victor/Loki

unsigned tausStep(const unsigned z, const int s1, const int s2, const int s3, const unsigned M) {
    unsigned b=(((z << s1) ^ z) >> s2);
    return (((z & M) << s3) ^ b);
}

float getSeed(const unsigned seed1, const unsigned seed2, const unsigned seed3) {
    unsigned seed = seed1 * 1099087573UL;
    unsigned seedb = seed2 * 1099087573UL;
    unsigned seedc = seed3 * 1099087573UL;

    // Round 1: Randomise seed
    unsigned z1 = tausStep(seed,13,19,12,429496729UL);
    unsigned z2 = tausStep(seed,2,25,4,4294967288UL);
    unsigned z3 = tausStep(seed,3,11,17,429496280UL);
    unsigned z4 = (1664525*seed + 1013904223UL);

    // Round 2: Randomise seed again using second seed
    unsigned r1 = (z1^z2^z3^z4^seedb);

    z1 = tausStep(r1,13,19,12,429496729UL);
    z2 = tausStep(r1,2,25,4,4294967288UL);
    z3 = tausStep(r1,3,11,17,429496280UL);
    z4 = (1664525*r1 + 1013904223UL);

    // Round 3: Randomise seed again using third seed
    r1 = (z1^z2^z3^z4^seedc);

    z1 = tausStep(r1,13,19,12,429496729UL);
    z2 = tausStep(r1,2,25,4,4294967288UL);
    z3 = tausStep(r1,3,11,17,429496280UL);
    z4 = (1664525*r1 + 1013904223UL);

    return (z1^z2^z3^z4) * 2.3283064365387e-10;
}

float rand(thread float &seed) {
    unsigned hashed_seed = seed * 1099087573UL;

    unsigned z1 = tausStep(hashed_seed,13,19,12,429496729UL);
    unsigned z2 = tausStep(hashed_seed,2,25,4,4294967288UL);
    unsigned z3 = tausStep(hashed_seed,3,11,17,429496280UL);
    unsigned z4 = (1664525*hashed_seed + 1013904223UL);

    float old_seed = seed;
    seed = (z1^z2^z3^z4) * 2.3283064365387e-10;
    return old_seed;
}
