#version 330 core

in vec3 fragPosition;
out vec4 color;

// view params includes width(x), height(y) and the plane depth(z)
uniform vec3 ViewParams;
uniform mat4x4 CamLocalToWorldMatrix;
uniform vec3 CamGlobalPos;
uniform float time;


struct Material
{
    vec4 color;
};

struct Sphere
{
    vec3 position;
    float radius;
    Material material;
};

const int SPHERES_COUNT_MAX = 2;

uniform int spheresCount;

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

float inf = 1.0 / 0.0;

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

float randomValue(uint seed) 
{
    state *= (state + 195439) * (state + 124395) * (state + 845921);
    return state / 4294967295.0;
}


void main() 
{
    vec3 viewPointLocal = (vec3(fragPosition.xy / 2, 1) * ViewParams);
    vec3 viewPoint = (CamLocalToWorldMatrix * vec4(viewPointLocal.xyz, 1.0)).xyz;

    Ray ray;
    ray.origin = CamGlobalPos;
    ray.dir = normalize(viewPoint - ray.origin);

    // HitInfo hit = raySphere(
    //     ray, 
    //     spheres[0].position,
    //     spheres[0].radius
    // );

    HitInfo hit = calculateRayCollision(ray);

    // color = vec4(abs(hit.normal), 1.0);
    color = hit.sphere.material.color;

}



