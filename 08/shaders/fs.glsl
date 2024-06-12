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
// const int TRIANGLES_COUNT_MAX = 256;
float inf = 1.0 / 0.0;
float pi = 3.14159265359;
uniform int MAX_BOUNCES;
uniform int RAYS_PER_PIXEL;

uniform int spheresCount;
// uniform int triCount;

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

// struct Triangle
// {
//     vec3 posA;
//     vec3 posB;
//     vec3 posC;
//     vec3 normalA;
//     vec3 normalB;
//     vec3 normalC;
// };

// layout(std140) uniform triBuffer 
// {
//     // int spheresCount; // I cannot figure out how to get this to work
//     Triangle triangles[SPHERES_COUNT_MAX];
// };

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
    Sphere sphere;
    // Material material;
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

HitInfo calculateRayCollision(Ray ray)
// calculates the ray collisions for all 
{
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
            // closestHit.sphere.material = sphere.material;
            closestHit.sphere = sphere;
        }
    }
    // todo: loop over all triangles in the scene
    // ...

    return closestHit;
}

float randomValue(inout uint state) 
{
    state *= (state + uint(195439)) * (state + uint(124395)) * (state + uint(845921));
    return state / 4294967295.0;
}

vec3 randomDirection(inout uint rngState) 
// calculates a random vector in a sphere
{
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

// struct Triangle2 
// {
//     vec3 v0;
//     vec3 v1;
//     vec3 v2;
// };

// Triangle2 myTriangle = Triangle2(
//     vec3(1, -1, 0),
//     vec3(-1, -1, 0),
//     vec3(0, 1, 0)
// );

// I am reading somethign wile about needing to flip the camera if triangles are modelled in a CW arrangement
// because this reverses the normals. Not sure how ti changes the position of the triangle...

// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html

// HitInfo rayTriangle2(Ray ray, Triangle2 tri)
// {

//     HitInfo hitInfo = defaultHitInfo();

//     // edges
//     vec3 edgeAB = tri.v1 - tri.v0;
//     vec3 edgeAC = tri.v2 - tri.v0;
//     vec3 edgeBC = tri.v2 - tri.v1;
//     // normal
//     vec3 crossproduct = cross(edgeAB, edgeAC); // the not-normalized normal
    
//     // the normal vector (normalized)
//     vec3 N = normalize(crossproduct);

//     // solving the last part of teh plane equation
//     float D = -dot(N, tri.v0);


//     // but what if the ray and triangle are parallel
//     // dot product of direction and normal vector is close to zero


//     // check if the ray is parallel
//     float parallelCheck = dot(N, ray.dir);
//     // if the ray is parallel, there is no intersection therefore no hit.
//     if (abs(parallelCheck) <= 1e-6)
//     {
//         hitInfo.didHit = false;
//         return hitInfo;
//     }

//     // the ray distance travelled to the intersection
//     float t = -(dot(N, ray.origin) + D) / dot(N, ray.dir);
    
//     // the position of the hit
//     vec3 phit = ray.origin * t * ray.dir;

//     // check if the phit is within the triangle, we have only checked intersection with the plane
//     vec3 C0 = phit - tri.v0;
//     vec3 C1 = phit - tri.v1;
//     vec3 C2 = phit - tri.v2;

//     bool insideOutsideCheck = (
//         dot(N, cross(edgeAB, C0)) > 0.0 &&
//         dot(N, cross(edgeAC, C1)) > 0.0 &&
//         dot(N, cross(edgeBC, C2)) > 0.0 
//     );

//     if (insideOutsideCheck)
//     {
//         hitInfo.didHit = true;
//         hitInfo.dst = t;
//         hitInfo.hitPoint = phit;
//         hitInfo.normal = N;
//     } else
//     {
//         hitInfo.didHit = false;
//     }

//     return hitInfo;

// }


// HitInfo rayTriangle(Ray ray, Triangle tri)
// {
//     // nabbed this function directly from Sebastian Lague's video

//     vec3 edgeAB = tri.posB - tri.posA;
//     vec3 edgeAC = tri.posC - tri.posA;
//     vec3 normalVector = cross(edgeAB, edgeAC);
//     vec3 ao = ray.origin - tri.posA;
//     vec3 dao = cross(ao, ray.dir);

//     float determinant = -dot(ray.dir, normalVector);
//     float invDet = 1 / determinant;
    
//     // I would like to remove this since I don't see why it is necessary for low poly objects

//     // Calculate dst to triangle & barycentric coordinates of intersection point
//     float dst = dot(ao, normalVector) * invDet;
//     float u = dot(edgeAC, dao) * invDet;
//     float v = -dot(edgeAB, dao) * invDet;
//     float w = 1 - u - v;
    
//     // Initialize hit info
//     HitInfo hitInfo;
//     hitInfo.didHit = determinant >= 1E-6 && dst >= 0 && u >= 0 && v >= 0 && w >= 0;
//     hitInfo.hitPoint = ray.origin + ray.dir * dst;

//     // hitInfo.normal = normalize(tri.normalA * w + tri.normalB * u + tri.normalC * v);
//     hitInfo.normal = normalVector;
//     hitInfo.dst = dst;
//     return hitInfo;
// }

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
            Material material = hitInfo.sphere.material;

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

    // if (hit.didHit) 
    // {
    //     color = vec4(1.0);
    // }

    color = vec4(0.5);

    vec3 totalIncomingLight = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < RAYS_PER_PIXEL; i++)
    {
        totalIncomingLight += traceRay(ray, rngState);
    }

    totalIncomingLight /= RAYS_PER_PIXEL;


    if (STATIC_RENDER)
    {
        float weight = 1.0 / (float(frameNumber) + 2);
        color = 
        texture(previousFrame, (fragPosition.xy + 1.0) / 2) * (1-weight) + 
        vec4(totalIncomingLight, 1.0) * weight;
    }
    else 
    {
        color = vec4(totalIncomingLight, 1.0);
    }


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



