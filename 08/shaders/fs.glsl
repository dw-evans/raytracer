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


uniform int spheresCount;



const int SPHERES_COUNT_MAX = 256;
float inf = 1.0 / 0.0;
float pi = 3.14159265359;
const int MAX_BOUNCES = 30;
const int RAYS_PER_PIXEL = 500;

// layout(std140)
layout(std140) uniform sphereBuffer 
{
    // int spheresCount; // I cannot figure out how to get this to work
    Sphere spheres[SPHERES_COUNT_MAX];
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
            closestHit.sphere = sphere;
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
            ray.dir = randomDirectionHemisphere(hitInfo.normal, rngState);
            
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
    uint rngState = pxId; //* uint(time * 1000000);

    // calculate the camere bits
    vec3 viewPointLocal = (vec3(fragPosition.xy / 2, 1) * ViewParams);
    vec3 viewPoint = (CamLocalToWorldMatrix * vec4(viewPointLocal.xyz, 1.0)).xyz;


    Ray ray;
    ray.origin = CamGlobalPos;
    ray.dir = normalize(viewPoint - ray.origin);


    HitInfo hit = calculateRayCollision(ray);

    if (hit.didHit) 
    {
        color = vec4(1.0);
    }

    // color = vec4(randomDirectionHemisphere(hit.normal, rngState), 1.0);
    // color = vec4(
    //     randomDirectionHemisphere(vec3(0,0,1), rngState),
    //     1.0
    // );

    vec3 totalIncomingLight = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < RAYS_PER_PIXEL; i++)
    {
        totalIncomingLight += traceRay(ray, rngState);
    }

    color = vec4(totalIncomingLight, 1.0);



}



