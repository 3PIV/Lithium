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

struct Material {
  float3 a;
  float r;
  float f;
  float ir;
};

struct HitRecord {
  float3 p;
  float3 n;
  Material m;
  bool h;
};

float sceneDistance(const float3 p) {
  SdfBox b(float3(-0.05, 0, -1), float3(0.25));
  SdfTorus t(float3(-0.1, 0, -1), float2(0.4, 0.2));
  SdfTorus t2(float3(-0.1, 0, -1), float2(1.0, 0.2));
  SdfTorus t3(float3(2.5, 1, -3), float2(0.5, 0.2));
  SdfSphere s2(float3(0.5, 0, -1), 0.4);
  SdfSphere s3(float3(2.5, 1, -3), 0.5);
  SdfSphere s4(float3(0.5, 0, -1), 0.4);
  SdfSphere s5(float3(0.5, 0, -1), 0.39);
  SdfSphere s6(float3(0.0, 0.0, -5), 2.13);
  SdfSphere s7(float3(-.4, -0.2, -0.65), 0.13);
  SdfSphere g(float3(0, -200.5, -1), 200.0);
  
  const auto bd = b.distance(p);
  const auto td = t.distance(p);
  const auto t2d = t2.distance(p);
  const auto t3d = t3.distance(p);
  const auto s2d = s2.distance(p);
  const auto s3d = s3.distance(p);
  const auto s4d = s4.distance(p);
  const auto s5d = s5.distance(p);
  const auto s6d = s6.distance(p);
  const auto s7d = s7.distance(p);
  const auto gd = g.distance(p);
  
  auto glob = sdfSmoothSubtraction(td, sdfSmoothUnion(bd, s2d, 0.5), 0.1);
  glob = sdfSubtraction(gd, sdfSmoothSubtraction(t2d, sdfSmoothSubtraction(s4d, glob, 0.1), 0.1));
  
  return sdfUnion(s7d, sdfUnion(s6d, sdfUnion(s5d, sdfUnion(sdfUnion(glob, gd), sdfSmoothSubtraction(t3d, s3d, 0.1)))));
}

inline float3 random_f3_in_unit_sphere(thread float &seed) {
  const float x = rand(seed) * 2.0 - 1.0;
  const float y = rand(seed) * 2.0 - 1.0;
  const float z = rand(seed) * 2.0 - 1.0;
  float3 p = float3(x, y, z);
  return normalize(p);
}

float shlickApprox(float cosine, float dr) {
  auto r0 = (1.0 - dr) / (1.0 + dr);
  r0 *= r0;
  return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

float3 scatter(thread const Material &m, thread const float3 &d, thread const float3 &n, thread float &s) {
  float3 rd = reflect(d, n);
  float3 direction = m.r > 0.0 ? rd : n;
  
  bool ff = dot(d, n) < 0.0;
  float  dr = ff ? 1.0 / m.ir : m.ir;
  float3 nd = ff ? normalize(n) : -normalize(n);
  float3 ud = normalize(d);
  float cos_theta = fmin(dot(-ud, nd), 1.0);
  float sin_theta = sqrt(1.0 - pow(cos_theta, 2));
  bool will_reflect = shlickApprox(cos_theta, dr) > rand(s);
  bool can_refract = !(dr * sin_theta > 1.0);
  float3 rfct = can_refract && !will_reflect ? refract(ud, nd, dr) : rd;
  
  float3 sd = m.ir > 0.0 ? rfct : direction;
  sd += m.f * random_f3_in_unit_sphere(s);
  length(sd) < 0.005 ? sd = nd : sd = sd;
  return normalize(sd);
}

Material sceneMaterial(const float3 p){
  Material mat = {float3(0.5, 0.5, 0.8), 0.0, 1.0};
  
  SdfBox b(float3(-0.05, 0, -1), float3(0.25));
  SdfTorus t(float3(-0.1, 0, -1), float2(0.4, 0.2));
  SdfTorus t2(float3(-0.1, 0, -1), float2(1.0, 0.2));
  SdfTorus t3(float3(2, 1, -3), float2(0.5, 0.2));
  SdfSphere s2(float3(0.5, 0, -1), 0.4);
  SdfSphere s3(float3(2, 1, -3), 0.5);
  SdfSphere s4(float3(0.5, 0, -1), 0.4);
  SdfSphere s5(float3(0.5, 0, -1), 0.39);
  SdfSphere s6(float3(0.0, 0.0, -5), 2.13);
  SdfSphere s7(float3(-.4, -0.2, -0.65), 0.13);
  SdfSphere g(float3(0, -1000.5, -1), 1000.0);
  
  const auto bd = b.distance(p);
  const auto td = t.distance(p);
  const auto t2d = t2.distance(p);
  const auto t3d = t3.distance(p);
  const auto s2d = s2.distance(p);
  const auto s3d = s3.distance(p);
  const auto s4d = s4.distance(p);
  const auto s5d = s5.distance(p);
  const auto s6d = s6.distance(p);
  const auto s7d = s7.distance(p);
  const auto gd = g.distance(p);
  
  auto glob = sdfSmoothSubtraction(td, sdfSmoothUnion(bd, s2d, 0.5), 0.1);
  glob = sdfSubtraction(gd, sdfSmoothSubtraction(t2d, sdfSmoothSubtraction(s4d, glob, 0.1), 0.1));
  
  auto moon = sdfSmoothSubtraction(t3d, s3d, 0.1);
  
  float3 a = 0.0;
  float r = -1.0;
  float f = 1.0;
  float ir = -1.0;
  float mv = sdfUnion(s7d, sdfUnion(sdfUnion(s5d, sdfUnion(sdfUnion(glob, gd), moon)), s6d));
  
  if (mv == s5d) {
    a = float3(1.0, 0.4, 0.4);
    r = 1.0;
    f = 0.0;
  } else if (mv == gd) {
    a = float3(0.1, 0.9, 0.1);
  } else if (mv == glob) {
    a = float3(0.3, 0.4, 0.9);
    r = 1.0;
    f = 0.5;
  } else if (mv == moon) {
    a = float3(0.6, 0.6, 0.6);
    r = 0.0;
    f = 1.0;
  } else if (mv == s6d) {
    a = float3(0.6, 0.1, 0.6);
  } else if (mv == s7d) {
    a = float3(0.9, 0.8, 0.9);
    f = 0.0;
    ir = 1.5;
  }
  
  mat.a = a;
  mat.r = r;
  mat.f = f;
  mat.ir = ir;
  
  return mat;
}

void testForHit(float (*sdfFunc)(const float3), thread const Ray& r, thread HitRecord& hr) {
  float d = 0.0;
  float t = 0.0;
  hr.h = false;
  for (int i = 0; i < 1000; ++i) {
    d = sdfFunc(r.o + r.d * t);
    
    if (abs(d) < 0.0001 * (1.0 + t)) {
      hr.h = true;
      hr.p = r.o + r.d * t;
      hr.n = sdfNormalEstimate(sdfFunc, r.o + r.d * t, r.d);
      hr.m = sceneMaterial(hr.p);
      return;
    }
    t += abs(d) * 0.75;
    if (t > 1000.0) return;
  }
}

float3 default_atmosphere_color(const thread Ray &r) {
  float3 white = float3(1.0, 1.0, 1.0);
  float3 atmos = float3(0.5, 0.7, 1.0);
  return mix(white, atmos, 0.5 * saturate(r.d.y) + 1.0);
}

//MARK: Ray Spawner
kernel void spawn_rays(const device float3 &origin [[ buffer(0) ]],
                       const device float3 &target [[ buffer(1) ]],
                       const device float &fieldOfView [[ buffer(2) ]],
                       const device float &focusDistance [[ buffer(3) ]],
                       const device float &apertureRadius [[ buffer(4) ]],
                       device float3 *rayOriginBuffer [[ buffer(5) ]],
                       device float3 *rayDirectionBuffer [[ buffer(6) ]],
                       const device uint &imageWidth [[ buffer(7) ]],
                       const device uint &imageHeight [[ buffer(8) ]],
                       const device uint &samplesPerPixel [[ buffer(9) ]],
                       const uint index [[thread_position_in_grid]]){
  if (index >= imageWidth * imageHeight) {
    return;
  }
  
  const uint row = index / imageWidth;
  const uint col = index % imageWidth;
  
  const float theta = fieldOfView * (1.0 / 180.0) * 3.1415926535;
  const float desiredHeight = tan(theta / 2);
  
  //const float3 origin = float3(0.0);
  const float aspect = float(imageWidth) / float(imageHeight);
  const float vph = 2.0 * desiredHeight;
  const float vpw = aspect * vph;
  
  const auto cw = normalize(origin - target);
  const auto cu = normalize(cross(float3(0.0, 1.0, 0.0), cw));
  const auto cv = cross(cw, cu);
  
  const float3 horizontal = focusDistance * vpw * cu;
  const float3 vertical = focusDistance * vph * cv;
  
  const float3 llc = origin - horizontal / 2.0 - vertical / 2.0 - focusDistance * cw;
  
  const float lensRadius = apertureRadius / 2.0;
  
  float seed = getSeed(index, col, row);
  
  for (uint i = 0; i < samplesPerPixel; ++i) {
    const float2 rd = (fmod(normalize(random_f3_in_unit_sphere(seed).xy), lensRadius) - 0.5 * lensRadius) * 2.0;
    const float3 amt = lensRadius * float3(rd, 0.0);
    const float3 offset = cu * amt.x + cv * amt.y;
    
    const float ranX = fract(rand(seed));
    const float ranY = fract(rand(seed));
    const float u = (float(col) + ranX) / float(imageWidth - 1);
    const float v = 1.0 - (float(row) + ranY) / float(imageHeight - 1);
    rayOriginBuffer[(index * samplesPerPixel) + i] = origin + offset;
    rayDirectionBuffer[(index * samplesPerPixel) + i] = normalize(llc + u * horizontal + v * vertical - origin - offset);
  }
}

//MARK: Primary Ray Cast Function
kernel void ray_trace(device float3 *origins [[ buffer(0) ]],
                      device float3 *directions [[ buffer(1) ]],
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
  float3 rayOrigin = origins[index];
  
  Ray r(rayOrigin, rayDirection);
  float3 color = float3(0.0);
  
  HitRecord hr = {0.0, 0.0, {float3(0.5, 0.5, 0.8), 0.0, 1.0}, false};
  float3 attenuation = 1.0;
  float seed = getSeed(index, pixelIndex, pixelIndex % imageWidth);
  
  for (uint bounceIndex = 0; bounceIndex < rayBounces; ++bounceIndex) {
    testForHit(sceneDistance, r, hr);
    if (hr.h) {
      float3 target = scatter(hr.m, r.d, hr.n, seed);
      attenuation *= hr.m.a;
      r = Ray(hr.p + target * 0.001, target);
    } else {
      color = default_atmosphere_color(r) * attenuation;
      break;
    }
  }
  
  directions[index] = color / samplesPerPixel;
}

kernel void combine_results(device float3 *intermediateResults [[ buffer(0) ]],
                            device float4 *results [[ buffer(1) ]],
                            texture2d <float, access::write> resultingImage [[ texture(0) ]],
                            const device uint &imageWidth [[ buffer(2) ]],
                            const device uint &imageHeight [[ buffer(3) ]],
                            const device uint &samplesPerPixel [[ buffer(4) ]],
                            const uint index [[thread_position_in_grid]]){
  if (index >= imageWidth * imageHeight) {
    return;
  }
  
  const uint row = index / imageWidth;
  const uint col = index % imageWidth;
  const uint2 gid = uint2(col, row);
  
  auto accumulatedColor = float3(0.0);
  for (uint i = 0; i < samplesPerPixel; ++i) {
    accumulatedColor += intermediateResults[(index * samplesPerPixel) + i];
  }
  
  results[index] = float4(sqrt(accumulatedColor), 1.0);
  resultingImage.write(float4(sqrt(accumulatedColor), 1.0), gid);
}
