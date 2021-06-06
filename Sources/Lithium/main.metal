#include <metal_stdlib>
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
  SdfSphere s(float3(0, 0, -1), 0.5);
  SdfSphere s2(float3(1, 0, -1), 0.7);
  SdfSphere s3(float3(-2, 1, -3), 0.5);
  SdfSphere g(float3(0, -200.5, -1), 200.0);
  
  const auto sd = s.distance(p);
  const auto s2d = s2.distance(p);
  const auto s3d = s3.distance(p);
  const auto gd = g.distance(p);
  
  const auto glob = sdfSmoothUnion(sd, s2d, 0.2);
  
  return sdfUnion(sdfUnion(glob, gd), s3d);
}

void testForHit(float (*sdfFunc)(const float3), thread const Ray& r, thread HitRecord& hr) {
  float d = 0.0;
  float t = 0.0;
  hr.h = false;
  for (int i = 0; i < 1000; ++i) {
    d = sdfFunc(r.o + r.d * t);
    
    if (abs(d) < 0.001 * (0.1 + t)) {
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

inline float3 random_f3_in_unit_sphere(float seed) {
  while (true) {
    const float x = rand(seed) * 2.0 - 1.0;
    const float y = rand(seed) * 2.0 - 1.0;
    const float z = rand(seed) * 2.0 - 1.0;
    float3 p = float3(x, y, z);
    if (length(p) >= 1.0) continue;
    return normalize(p);
  }
}

//MARK: Primary Ray Cast Function
kernel void ray_trace(device float4 *result [[ buffer(0) ]],
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
  
  float3 accumulatedColor = float3(0.0);
  const int samplesPerPixel = 16;
  thread float seed = getSeed(index, index % imageWidth, index / imageWidth);
  
  float row = float(index / imageWidth);
  float col = float(index % imageWidth);
  
  for (int aai = 0; aai < samplesPerPixel; ++aai) {
    float ranX = fract(rand(seed));
    float ranY = fract(rand(seed));
    float u = (col + ranX) / float(imageWidth - 1);
    float v = 1.0 - (row + ranY) / float(imageHeight - 1);
    Ray r(origin, llc + u * vpw + v * vph - origin);
    
    float3 color = float3(0.0);
    HitRecord hr = {0.0, 0.0, false};
    
    float attenuation = 1.0;
    for (int bounces = 0; bounces < 15; ++bounces) {
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
    
    
    //testForHit(sceneDistance, r, hr);
    //color = mix(color, abs(hr.n), float(hr.h));
    accumulatedColor += color / samplesPerPixel;
  }

  result[index] = float4(sqrt(accumulatedColor), 1.0);
}
