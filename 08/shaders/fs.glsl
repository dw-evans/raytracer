#version 330 core

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
const int TRIANGLES_COUNT_MAX = 256;
float inf = 1.0 / 0.0;
float pi = 3.14159265359;
uniform int MAX_BOUNCES;
uniform int RAYS_PER_PIXEL;

uniform int spheresCount;
uniform int triCount;

uniform sampler2D previousFrame;

uniform uint frameNumber;

uniform bool STATIC_RENDER;

struct Material
{
    vec4 color;
    vec3 emissionColor;
    float emissionStrength;
};

struct Sphere
{
    vec3 position;
    float radius;
    Material material;
};

// layout(std140)
layout(std140) uniform sphereBuffer 
{
    // int spheresCount; // I cannot figure out how to get this to work
    Sphere spheres[SPHERES_COUNT_MAX];
};

struct Triangle
{
    vec3 posA;
    vec3 posB;
    vec3 posC;
    vec3 normalA;
    vec3 normalB;
    vec3 normalC;
    Material material;
};

layout(std140) uniform triBuffer 
{
    // int spheresCount; // I cannot figure out how to get this to work
    Triangle triangles[1];
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
    // Sphere sphere;
    Material material;
};


HitInfo defaultHitInfo() {
    HitInfo hitInfo;
    hitInfo.didHit = false;
    hitInfo.dst = 0.0;
    hitInfo.hitPoint = vec3(0.0);
    hitInfo.normal = vec3(0.0);
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
    if (abs(NDotRayDirection) < kEpsilon)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    float d = -dot(N, v0);

    float t = -(dot(N, orig) + d) / NDotRayDirection;

    if (t < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    vec3 P = orig + t * dir;

    vec3 C;

    vec3 edge0 = v1 - v0;
    vec3 vp0 = P - v0;
    C = cross(edge0, vp0);
    if (dot(N, C) < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    vec3 edge1 = v2 - v1;
    vec3 vp1 = P - v1;
    C = cross(edge1, vp1);
    if (dot(N, C) < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    vec3 edge2 = v0 - v2;
    vec3 vp2 = P - v2;
    C = cross(edge2, vp2);
    if (dot(N, C) < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    hitInfo.normal = tri.normalA;
    hitInfo.didHit = true;
    hitInfo.material = tri.material;

    return hitInfo;

}


HitInfo calculateRayCollision(Ray ray)
{
    // calculates the ray collisions for all 
    HitInfo closestHit = defaultHitInfo();

    closestHit.dst = inf;

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
            // closestHit.sphere = sphere;
        }
    }
    // loop over all triangles in the scene
    for (int i = 0; i < triCount; i++)
    {
        Triangle tri = triangles[i];
        HitInfo hitInfo = rayTriangle(
            ray,
            tri
        );

        if (hitInfo.didHit && hitInfo.dst < closestHit.dst)
        {
            closestHit = hitInfo;
            closestHit.material = tri.material;
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



// I am reading somethign wile about needing to flip the camera if triangles are modelled in a CW arrangement
// because this reverses the normals. Not sure how ti changes the position of the triangle...

// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html


vec3 traceRay(Ray ray, inout uint rngState)
{

    vec3 incomingLight = vec3(0.0, 0.0, 0.0);
    vec3 rayColor = vec3(1.0,1.0,1.0);

    for (int i = 0; i <= MAX_BOUNCES; i++)
    {
        HitInfo hitInfo = calculateRayCollision(ray);
        if (hitInfo.didHit)
        {
            ray.origin = hitInfo.hitPoint;
            // ray.dir = randomDirectionHemisphere(hitInfo.normal, rngState);
            // why on earth does this work?
            ray.dir = normalize(hitInfo.normal + randomDirection(rngState));
            
            // Material material = hitInfo.material;
            Material material = hitInfo.material;

            vec3 emittedLight = material.emissionColor * material.emissionStrength;

            incomingLight += emittedLight * rayColor;
            rayColor *= material.color.xyz;

        } else 
        {
            break;
        }
    }

    return incomingLight;

}


void main() 
{

    // calculate a random number generator state
    uint numPixels = screenWidth * screenHeight;
    vec4 pxCoord = gl_FragCoord;
    uint pxId = uint(pxCoord.x * screenWidth * screenHeight) + uint(pxCoord.y * screenHeight);
    uint rngState = pxId + frameNumber;

    // calculate the camere bits
    vec3 viewPointLocal = (vec3(fragPosition.xy / 2.0, 1) * ViewParams);
    vec3 viewPoint = (CamLocalToWorldMatrix * vec4(viewPointLocal.xyz, 1.0)).xyz;

    // Btw we can just interpolate the ray direction from the vertex shader.
    Ray ray;
    ray.origin = CamGlobalPos;
    ray.dir = normalize(viewPoint - ray.origin);

    HitInfo hit = calculateRayCollision(ray);

    vec3 totalIncomingLight = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < RAYS_PER_PIXEL; i++)
    {
        totalIncomingLight += traceRay(ray, rngState);
    }

    totalIncomingLight /= RAYS_PER_PIXEL;

    // Triangle myTriangle = triangles[0];

    // HitInfo hit2 = rayTriangle2(ray, myTriangl2e);
    // the normal seems to work, but the hit info is not correctly detecting hits
    // HitInfo hit2 = rayTriangle(ray, myTriangle);
    // HitInfo hit2 = rayTriangle(ray, myTriangle);

    if (STATIC_RENDER)
    {
        float weight = 1.0 / (float(frameNumber) + 2);
        color = 
        // texture(previousFrame, (fragPosition.xy + 1.0) / 2) * (1-weight) + 
        // vec4(totalIncomingLight, 1.0) * weight;

        texture(previousFrame, (fragPosition.xy + 1.0) / 2) * 0.01 
        + vec4(totalIncomingLight, 1.0) * 0.01 
        // + vec4(normalize(abs(hit2.normal)), 1.0)
        ;
    }
    else 
    {
        // color = vec4(0.5);
        color = vec4(totalIncomingLight, 1.0) * 0.5
        // + vec4(vec3(hit2.didHit), 1.0) * 0.5
        // + vec4(hit2.material.color) * 0.5
        ;
    }

    // HitInfo hit2 = rayTriangle2(ray, myTriangle);
    // color = vec4(hit2.didHit);


    // Triangle tri = triangles[0];


    // HitInfo hit2 = rayTriangle(ray, tri);

    // if (hit2.didHit)
    // {
    //     color = vec4(1.0);
    // }

    // // color = vec4(hit2.didHit);

    // tri = Triangle(
    //     vec3(0, 0, 10),
    //     vec3(5, 10, 10),
    //     vec3(10, 0, 10),
    //     vec3(0,0,-1),
    //     vec3(0,0,-1),
    //     vec3(0,0,-1)
    // );

    // // color = vec4(normalize(tri.posA), 1.0);
    // // color = vec4(normalize(tri.posB), 1.0);
    // // color = vec4(normalize(tri.posC), 1.0);
    // // color = vec4(abs(normalize(tri.normalA)), 1.0);
    // // color = vec4(abs(normalize(tri.normalB)), 1.0);
    // // color = vec4(abs(normalize(tri.normalC)), 1.0);
    // // position, normals all working. how is didhit doing?


    // color = vec4(hit2.didHit);


    
}



