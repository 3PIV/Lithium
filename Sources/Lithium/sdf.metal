#include <metal_stdlib>
#include "sdf_header.metal"

using namespace metal;

// MARK: Normal Estimation
float3 sdfNormalEstimate(float (*sdfFunc)(const float3), const float3 p, const float3 d) {
  float epsilon = 0.001;
  float3 estimate = float3(
                           sdfFunc(float3(p.x + epsilon, p.y, p.z)) - sdfFunc(float3(p.x - epsilon, p.y, p.z)),
                           sdfFunc(float3(p.x, p.y + epsilon, p.z)) - sdfFunc(float3(p.x, p.y - epsilon, p.z)),
                           sdfFunc(float3(p.x, p.y, p.z  + epsilon)) - sdfFunc(float3(p.x, p.y, p.z - epsilon))
                           );
  bool inside = sdfFunc(p + epsilon * normalize(estimate)) < 0.0;
  return inside ? -normalize(estimate) : normalize(estimate);
}

// MARK: Sphere Definitions
SdfSphere::SdfSphere(const thread float3 &origin, float radius): o(origin), r(radius){};

float SdfSphere::distance(const float3 p) {
  return length(p - o) - r;
}

float3 SdfSphere::normal(const thread float3 &h) {
  return normalize(h - o);
}

SdfBox::SdfBox(const thread float3 &center, const thread float3 &bounds): c(center), b(bounds) {};
  
float SdfBox::distance(const float3 p) {
  float3 q = abs(p - c) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

SdfTorus::SdfTorus(const thread float3 &center, const thread float2 &dimensions): c(center), d(dimensions) {};
  
float SdfTorus::distance(const float3 p) {
  float3 diff = p - c;
  float2 q = float2(length(diff.xz) - d.x, diff.y);
  return length(q) - d.y;
}

// MARK: CSG Definitions
float sdfUnion(float d1, float d2) {
  return min(d1, d2);
}

float sdfSubtraction(float d1, float d2) {
  return max(-d1, d2);
}

float sdfIntersection(float d1, float d2) {
  return max(d1, d2);
}

float sdfSmoothUnion( float d1, float d2, float k ) {
  float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) - k * h * (1.0 - h);
}

float sdfSmoothSubtraction( float d1, float d2, float k ) {
  float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
  return mix(d2, -d1, h) + k * h * (1.0 - h);
}

float sdfSmoothIntersection( float d1, float d2, float k ) {
  float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) + k * h * (1.0 - h);
}
