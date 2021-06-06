#include <metal_stdlib>
#include "rand_header.metal"

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

//MARK: Sphere Class Definition
class Sphere {
  // Methods
public:
  Sphere(const thread float3 &origin, float radius): o(origin), r(radius){};
  
  float distance(const float3 p) {
    return length(p - o) - r;
  }
  
  float3 normal(const thread float3 &h) {
    return normalize(h - o);
  }
  
  // Members
public:
  float3 o;
  float r;
};

float sceneDistance(const float3 p) {
  Sphere s(float3(0, 0, -1), 0.5);
  Sphere g(float3(0, -200.5, -1), 200.0);
  
  return min(s.distance(p), g.distance(p));
}

float3 sdfNormalEstimate(float (*sdfFunc)(const float3), const float3 p) {
  float epsilon = 0.01;
  float3 estimate = float3(
                           sdfFunc(float3(p.x + epsilon, p.y, p.z)) - sdfFunc(float3(p.x - epsilon, p.y, p.z)),
                           sdfFunc(float3(p.x, p.y + epsilon, p.z)) - sdfFunc(float3(p.x, p.y - epsilon, p.z)),
                           sdfFunc(float3(p.x, p.y, p.z  + epsilon)) - sdfFunc(float3(p.x, p.y, p.z - epsilon))
                           );
  return normalize(estimate);
}

float3 default_atmosphere_color(const thread Ray &r) {
  float3 white = float3(1.0, 1.0, 1.0);
  float3 atmos = float3(0.5, 0.7, 1.0);
  return mix(white, atmos, 0.5 * r.d.y + 1.0);
}

//MARK: Primary Ray Cast Function
kernel void primary_ray(device float4 *result [[ buffer(0) ]],
                        const device uint& dataLength [[ buffer(1) ]],
                        const device int& imageWidth [[ buffer(2) ]],
                        const device int& imageHeight [[ buffer(3) ]],
                        const uint index [[thread_position_in_grid]]) {
  
  if (index > dataLength) {
    return;
  }
  
  const float3 origin = float3(0.0);
  const float aspect = float(imageWidth) / float(imageHeight);
  const float3 vph = float3(0.0, 2.0, 0.0);
  const float3 vpw = float3(2.0 * aspect, 0.0, 0.0);
  const float3 llc = float3(-(vph / 2.0) - (vpw / 2.0) - float3(0.0, 0.0, 1.0));
  
  float d = 0.0;
  float t = 0.0;
  
  float3 accumulatedColor = float3(0.0);
  const int samplesPerPixel = 100;
  float seed = getSeed(index, index % imageWidth, index / imageWidth);
  
  float row = float(index / imageWidth);
  float col = float(index % imageWidth);
  
  for (int aai = 0; aai < samplesPerPixel; ++aai) {
    
    float ran = fmod(rand(seed), 1.0);
    float u = (col + ran) / float(imageWidth - 1);
    float v = 1.0 - (row + ran) / float(imageHeight - 1);
    Ray r(origin, llc + u * vpw + v * vph - origin);
    
    float3 color = default_atmosphere_color(r);
    for (int i = 0; i < 100; ++i) {
      d = sceneDistance(r.o + r.d * t);
      
      if (abs(d) < 0.0001 * (1.0 + t)) {
        color = abs(sdfNormalEstimate(sceneDistance, r.o + r.d * t));
        break;
      }
      t += d * 0.75;
      if (t > 1000.0) return;
    }
    accumulatedColor += color / samplesPerPixel;
  }

  result[index] = float4(accumulatedColor, 1.0);
}
