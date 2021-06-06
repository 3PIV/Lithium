#include <metal_stdlib>

using namespace metal;

#include <metal_stdlib>

using namespace metal;

float3 sdfNormalEstimate(float (*sdfFunc)(const float3), const float3 p);

//MARK: Sphere Class Definition
class SdfSphere {
  // Methods
public:
  SdfSphere(const thread float3 &origin, float radius);
  
  float distance(const float3 p);
  
  float3 normal(const thread float3 &h);
  
  // Members
public:
  float3 o;
  float r;
};


float sdfUnion(float d1, float d2);

float sdfSubtraction(float d1, float d2);

float sdfIntersection(float d1, float d2);

float sdfSmoothUnion( float d1, float d2, float k );

float sdfSmoothSubtraction( float d1, float d2, float k );

float sdfSmoothIntersection( float d1, float d2, float k );
