#include <metal_stdlib>
#include <metal_atomic>
#include "rand_header.metal"
#include "sdf_header.metal"

using namespace metal;

//MARK: Ray Class Definition
class Ray {
  // Methods
public:
  Ray(const thread float3 &origin, const thread float3 &direction): o(origin), d(normalize(direction)) {};
  
  float3 at(float t) {
    return o + t * d;
  }
  // Members
public:
  float3 o;
  float3 d;
};

struct HitRecord {
  float3 p;
  float3 n;
  bool h;
};

float sceneDistance(const float3 p) {
  SdfBox b(float3(-0.05, 0, -1), float3(0.25));
  SdfTorus t(float3(-0.1, 0, -1), float2(0.4, 0.2));
  SdfTorus t2(float3(-0.1, 0, -1), float2(1.0, 0.2));
  SdfSphere s2(float3(1, 0, -1), 0.7);
  SdfSphere s3(float3(-2, 1, -3), 0.5);
  SdfSphere s4(float3(0.5, 0, -1), 0.4);
  SdfSphere s5(float3(0.5, 0, -1), 0.39);
  SdfSphere g(float3(0, -200.5, -1), 200.0);
  
  const auto bd = b.distance(p);
  const auto td = t.distance(p);
  const auto t2d = t2.distance(p);
  const auto s2d = s2.distance(p);
  const auto s3d = s3.distance(p);
  const auto s4d = s4.distance(p);
  const auto s5d = s5.distance(p);
  const auto gd = g.distance(p);
  
  auto glob = sdfSmoothSubtraction(td, sdfSmoothUnion(bd, s2d, 0.5), 0.1);
  glob = sdfSmoothSubtraction(t2d, sdfSmoothSubtraction(s4d, glob, 0.1), 0.1);
  
  return sdfUnion(s5d, sdfUnion(sdfUnion(glob, gd), s3d));
}

void testForHit(float (*sdfFunc)(const float3), thread const Ray& r, thread HitRecord& hr) {
  float d = 0.0;
  float t = 0.0;
  hr.h = false;
  for (int i = 0; i < 1000; ++i) {
    d = sdfFunc(r.o + r.d * t);
    
    if (abs(d) < 0.001 * (0.125 + t)) {
      hr.h = true;
      hr.p = r.o + r.d * t;
      hr.n = sdfNormalEstimate(sceneDistance, r.o + r.d * t);
      return;
    }
    t += d * 0.75;
    if (t > 1000.0) return;
  }
}

float3 default_atmosphere_color(const thread Ray &r) {
  float3 white = float3(1.0, 1.0, 1.0);
  float3 atmos = float3(0.5, 0.7, 1.0);
  return mix(white, atmos, 0.5 * r.d.y + 1.0);
}

inline float3 random_f3_in_unit_sphere(thread float &seed) {
  const float x = rand(seed) * 2.0 - 1.0;
  const float y = rand(seed) * 2.0 - 1.0;
  const float z = rand(seed) * 2.0 - 1.0;
  float3 p = float3(x, y, z);
  return normalize(p);
}

//MARK: Ray Spawner
kernel void spawn_rays(const device float3 &origin [[ buffer(0) ]],
                       device float3 *rayDirectionBuffer [[ buffer(1) ]],
                       const device uint &imageWidth [[ buffer(2) ]],
                       const device uint &imageHeight [[ buffer(3) ]],
                       const device uint &samplesPerPixel [[ buffer(4) ]],
                       const uint index [[thread_position_in_grid]]){
  if (index >= imageWidth * imageHeight) {
    return;
  }
  
  const uint row = index / imageWidth;
  const uint col = index % imageWidth;
  
  //const float3 origin = float3(0.0);
  const float aspect = float(imageWidth) / float(imageHeight);
  const float3 vph = float3(0.0, 2.0, 0.0);
  const float3 vpw = float3(2.0 * aspect, 0.0, 0.0);
  const float3 llc = float3(-(vph / 2.0) - (vpw / 2.0) - float3(0.0, 0.0, 1.0));
  
  float seed = getSeed(index, col, row);
  
  for (uint i = 0; i < samplesPerPixel; ++i) {
    float ranX = fract(rand(seed));
    float ranY = fract(rand(seed));
    float u = (float(col) + ranX) / float(imageWidth - 1);
    float v = 1.0 - (float(row) + ranY) / float(imageHeight - 1);
    rayDirectionBuffer[(index * samplesPerPixel) + i] = normalize(llc + u * vpw + v * vph - origin);
  }
}

//MARK: Primary Ray Cast Function
kernel void ray_trace(device float3 *directions [[ buffer(0) ]],
                      const device float3 &origin [[ buffer(1) ]],
                      const device uint &imageWidth [[ buffer(2) ]],
                      const device uint &imageHeight [[ buffer(3) ]],
                      const device uint &samplesPerPixel [[ buffer(4) ]],
                      const device uint &rayBounces [[ buffer (5) ]],
                      const uint index [[thread_position_in_grid]]) {
  
  if (index >= imageWidth * imageHeight * samplesPerPixel) {
    return;
  }
  const uint pixelIndex = index / 16;
  
  float3 rayDirection = directions[index];
  float3 rayOrigin = origin;
  
  Ray r(rayOrigin, rayDirection);
  float3 color = float3(0.0);

  HitRecord hr = {0.0, 0.0, false};
  float attenuation = 1.0;
  float seed = getSeed(index, pixelIndex, pixelIndex % imageWidth);
  
  for (uint bounceIndex = 0; bounceIndex < rayBounces; ++bounceIndex) {
    testForHit(sceneDistance, r, hr);
    if (hr.h) {
      float3 target = hr.p + hr.n + random_f3_in_unit_sphere(seed);
      attenuation *= 0.5;
      r = Ray(hr.p, target - hr.p);
    } else {
      color = default_atmosphere_color(r) * attenuation;
      break;
    }
  }

  directions[index] = color / samplesPerPixel;
}

kernel void combine_results(device float3 *intermediateResults [[ buffer(0) ]],
                            device float4 *results [[ buffer(1) ]],
                            const device uint &imageWidth [[ buffer(2) ]],
                            const device uint &imageHeight [[ buffer(3) ]],
                            const device uint &samplesPerPixel [[ buffer(4) ]],
                            const uint index [[thread_position_in_grid]]){
  if (index >= imageWidth * imageHeight) {
    return;
  }
  
  auto accumulatedColor = float3(0.0);
  for (uint i = 0; i < samplesPerPixel; ++i) {
    accumulatedColor += intermediateResults[(index * samplesPerPixel) + i];
  }
  
  results[index] = float4(sqrt(accumulatedColor), 1.0);
}
