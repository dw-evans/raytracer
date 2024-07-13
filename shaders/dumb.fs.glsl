#version 460 core

in vec3 fragPosition;
out vec4 color;

// view params includes width(x), height(y) and the plane depth(z)
uniform uint screenWidth;
uniform uint screenHeight;

uniform vec3 ViewParams;
uniform mat4x4 CamLocalToWorldMatrix;
uniform vec3 CamGlobalPos;
uniform float time;

const int SPHERES_COUNT_MAX = 256;
const int TRIANGLES_COUNT_MAX = 16384;
const int MESHES_COUNT_MAX = 256;

uniform int spheresCount;
uniform int triCount;
uniform int meshCount;

uniform sampler2D previousFrame;

uniform uint frameNumber;

uniform bool STATIC_RENDER;

uniform vec3 skyColor;
uniform vec3 groundColor;

uniform int selectedMeshId;

float inf = 1.0 / 0.0;
float pi = 3.14159265359;

struct Material
{
    vec4 color; // 16 
    vec3 emissionColor; // 12
    float emissionStrength; // 4
    float smoothness; // 4 + 12
    float transition;
    float ior;
};

layout(std140) uniform materialBuffer
{
    Material highlightMaterials[4];
};

struct Sphere
{
    vec3 position; // 12 + 4
    float radius; // 4 + 12
    Material material; // ...
};

// layout(std140)
layout(std140) uniform sphereBuffer 
{
    // int spheresCount; // I cannot figure out how to get this to work
    Sphere spheres[SPHERES_COUNT_MAX];
};

struct Triangle
{
    vec3 posA; // 12 + 4
    vec3 posB; // 12 + 4
    vec3 posC; // 12 + 4
    vec3 normalA; // 12 + 4
    vec3 normalB; // 12 + 4
    vec3 normalC; // 12 + 4
    int meshIndex; // 4 + 12
    int triId;
    Material material; // ...
};


layout(std140, binding=9) buffer triBuffer
{
    Triangle triangles[TRIANGLES_COUNT_MAX];
};

Triangle getTriangle(int index)
// legacy for implementation with multiple uniform buffers
{
    Triangle ret;

    return triangles[index];
}


struct Mesh
{
    int index;
    vec3 bboxMin;
    vec3 bboxMax;
};

layout(std140) uniform meshBuffer 
{
    Mesh meshes[MESHES_COUNT_MAX];
};


struct Ray 
{
    vec3 origin;
    vec3 dir;
};

struct HitInfo 
{
    bool didHit;
    float dst;
    vec3 hitPoint;
    vec3 normal;
    Material material;
    int meshIndex;
    int facetId;
    bool hitBackside;
};


HitInfo defaultHitInfo() {
    HitInfo hitInfo;
    hitInfo.didHit = false;
    hitInfo.dst = inf;
    hitInfo.hitPoint = vec3(0.0);
    hitInfo.normal = vec3(0.0);
    hitInfo.material = Material(
        vec4(0.0),
        vec3(0.0),
        0.0,
        0.0,
        0.0,
        1.0
    );
    return hitInfo;
}

// calculate the hit information between a ray and a sphere
HitInfo raySphere(Ray ray, vec3 spherePos, float sphereRadius) 
{
    HitInfo hitInfo = defaultHitInfo();

    vec3 offsetRayOrigin = ray.origin - spherePos;

    float a = dot(ray.dir, ray.dir); // a = 1 (assuming unit vector)
    float b = 2 * dot(offsetRayOrigin, ray.dir);
    float c = dot(offsetRayOrigin, offsetRayOrigin) - sphereRadius * sphereRadius;

    float discriminant = b * b - 4 * a * c; 

    // No solution when d < 0 (ray misses sphere)
    if (discriminant >= 0) {
        // Distance to nearest intersection point (from quadratic formula)
        float dst = (-b - sqrt(discriminant)) / (2 * a);
        // Ignore intersections that occur behind the ray
        if (dst >= 0) {
            hitInfo.didHit = true;
            hitInfo.dst = dst;
            hitInfo.hitPoint = ray.origin + ray.dir * dst;
            hitInfo.normal = normalize(hitInfo.hitPoint - spherePos);
        }
    }
    return hitInfo;
}


vec3 getEnvironmentLight(Ray ray)
{
    return skyColor;
}
HitInfo rayTriangle(Ray ray, Triangle tri)
{
    HitInfo hitInfo = defaultHitInfo();

    vec3 v0 = tri.posA;
    vec3 v1 = tri.posB;
    vec3 v2 = tri.posC;

    vec3 dir = ray.dir;
    vec3 orig = ray.origin;
    
    vec3 v0v1 = v1 - v0;
    vec3 v0v2 = v2 - v0;

    vec3 N = cross(v0v1, v0v2);

    float kEpsilon = 1e-6;
    float NDotRayDirection = dot(N, dir);

    // check parallel
    if (abs(NDotRayDirection) < kEpsilon)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    // if this dot product is positive, the ray is entering through the back
    if (NDotRayDirection > 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    // solve plane equation and ignore triangles behind the orig
    float d = -dot(N, v0);
    float t = -(dot(N, orig) + d) / NDotRayDirection;

    if (t < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    vec3 P = orig + t * dir;

    vec3 C;

    float area = length(N) / 2.0;

    vec3 edge0 = v1 - v0;
    vec3 vp0 = P - v0;
    C = cross(edge0, vp0);
    float w = length(C) / 2.0 / area;
    if (dot(N, C) < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    vec3 edge1 = v2 - v1;
    vec3 vp1 = P - v1;
    C = cross(edge1, vp1);
    float u = length(C) / 2.0 / area;
    if (dot(N, C) < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    vec3 edge2 = v0 - v2;
    vec3 vp2 = P - v2;
    C = cross(edge2, vp2);
    float v = 1 - u - w;
    if (dot(N, C) < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    // hitInfo.normal = tri.normalA;
    // hitInfo.normal = N / float(length(N));
    hitInfo.normal = tri.normalA * u + tri.normalB * v + tri.normalC * w;
    hitInfo.didHit = true;
    // hitInfo.material = tri.material;
    hitInfo.dst = t;

    return hitInfo;

}

void swap(inout float a, inout float b) {
    float temp = a;
    a = b;
    b = temp;
}

bool boundingBoxIntersect(Ray ray, vec3 bboxMin, vec3 bboxMax)
{
    // calculates the boolean intersection between a aabb and a ray
    float tmin = (bboxMin.x - ray.origin.x) / ray.dir.x; 
    float tmax = (bboxMax.x - ray.origin.x) / ray.dir.x; 

    if (tmin > tmax) swap(tmin, tmax); 

    float tymin = (bboxMin.y - ray.origin.y) / ray.dir.y; 
    float tymax = (bboxMax.y - ray.origin.y) / ray.dir.y; 

    if (tymin > tymax) swap(tymin, tymax); 

    if ((tmin > tymax) || (tymin > tmax)) 
        return false; 

    if (tymin > tmin) tmin = tymin; 
    if (tymax < tmax) tmax = tymax; 

    float tzmin = (bboxMin.z - ray.origin.z) / ray.dir.z; 
    float tzmax = (bboxMax.z - ray.origin.z) / ray.dir.z; 

    if (tzmin > tzmax) swap(tzmin, tzmax); 

    if ((tmin > tzmax) || (tzmin > tmax)) 
        return false; 

    if (tzmin > tmin) tmin = tzmin; 
    if (tzmax < tmax) tmax = tzmax; 

    return true; 
}


HitInfo calculateRayCollision(Ray ray)
{
    // calculates the ray collisions for all 
    HitInfo closestHit = defaultHitInfo();

    // loop over all spheres in the scene
    for (int i = 0; i < spheresCount; i++)
    {
        Sphere sphere = spheres[i];
        HitInfo hitInfo = raySphere(
            ray,
            sphere.position,
            sphere.radius
        );

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst)
        {
            closestHit = hitInfo;
            closestHit.material = sphere.material;
        }
    }
    // loop over all triangles in the scene

    bool meshIntersectArray[MESHES_COUNT_MAX];

    // determine which mesh indices are good
    for (int j = 0; j < meshCount; j++)
    {
        // check if the ray intersects the bounding box
        Mesh mesh = meshes[j];
        bool intersectsBbox = boundingBoxIntersect(ray, mesh.bboxMin, mesh.bboxMax);
        meshIntersectArray[j] = intersectsBbox;
    }

    for (int i = 0; i < triCount; i++)
    {
        Triangle tri = getTriangle(i);

        int triMeshIndex = tri.meshIndex;

        for (int j = 0; j < meshCount; j++)
        {
            // if the ray isn't intersecting the bounding box of the triangle's mesh
            // we will not perform the calculation
            if (!meshIntersectArray[j])
            {
                break;
            }
        }


        HitInfo hitInfo = rayTriangle(
            ray,
            tri
        );
        
        if (hitInfo.didHit && hitInfo.dst < closestHit.dst)
        {
            closestHit = hitInfo;
            
            if (tri.meshIndex == selectedMeshId)
            {
                closestHit.material = highlightMaterials[0];
            }
            else
            {
                closestHit.material = tri.material;
            }
            
        }
    }

    return closestHit;
}

float randomValue(inout uint state) 
{
    state *= (state + uint(195439)) * (state + uint(124395)) * (state + uint(845921));
    return state / 4294967295.0;
}

vec3 randomDirection(inout uint rngState) 
{
    // calculates a random vector in a sphere
    float u = randomValue(rngState);
    float v = randomValue(rngState);

    float theta = 2.0 * pi * u;
    float phi = acos(2.0 * v - 1.0);

    return vec3(
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    );
}
vec3 randomDirectionHemisphere(vec3 normal, inout uint rngState) 
{
    vec3 randomDir = randomDirection(rngState);
    // return normal;
    return sign(dot(randomDir, normal)) * randomDir;
}

vec3 traceRay(Ray ray)
{

    vec3 incomingLight = vec3(0.0, 0.0, 0.0);
    vec3 rayColor = vec3(1.0,1.0,1.0);

    HitInfo hitInfo = calculateRayCollision(ray);
    if (hitInfo.didHit)
    {
        ray.origin = hitInfo.hitPoint;
        Material material = hitInfo.material;
        vec3 emittedLight = material.emissionColor * material.emissionStrength;
        incomingLight += material.color.xyz
        * mix(vec3(0.4), vec3(1.2), hitInfo.normal.y / 2.0 + 0.5)
        ;

    } else 
    {
        incomingLight += getEnvironmentLight(ray) * rayColor;
    }

    return incomingLight;

}


void main() 
{
    vec3 viewPointLocal = (vec3(fragPosition.xy / 2.0, 1) * ViewParams);
    vec3 viewPoint = (transpose(CamLocalToWorldMatrix) * vec4(viewPointLocal.xyz, 1.0)).xyz;

    Ray ray;
    ray.origin = CamGlobalPos;
    ray.dir = normalize(viewPoint);

    vec3 totalIncomingLight = traceRay(ray);

    color = vec4(totalIncomingLight, 1.0)
    ;
    
}



