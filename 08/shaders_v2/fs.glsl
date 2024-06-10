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
    vec4 colour;
};

struct Sphere
{
    vec3 position;
    float radius;
    Material material;
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
};

HitInfo InitializeHitInfo() {
    HitInfo hitInfo;
    hitInfo.didHit = false;
    hitInfo.dst = 0.0;
    hitInfo.hitPoint = vec3(0.0);
    hitInfo.normal = vec3(0.0);
    return hitInfo;
}



// calculate the hit information between a ray and a sphere
HitInfo RaySphere(Ray ray, Sphere sphere) 
{
    HitInfo hitInfo = InitializeHitInfo();

    vec3 offsetRayOrigin = ray.origin - sphere.position;

    float a = dot(ray.dir, ray.dir); // a = 1 (assuming unit vector)
    float b = 2 * dot(offsetRayOrigin, ray.dir);
    float c = dot(offsetRayOrigin, offsetRayOrigin) - sphere.radius * sphere.radius;

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
            hitInfo.normal = normalize(hitInfo.hitPoint - sphere.position);
        }
    }
    return hitInfo;
}


void main() {

    vec3 viewPointLocal = (vec3(fragPosition.xy / 2, 1) * ViewParams);
    vec3 viewPoint = (CamLocalToWorldMatrix * vec4(viewPointLocal.xyz, 1.0)).xyz;

    Ray ray;
    ray.origin = CamGlobalPos;
    ray.dir = normalize(viewPoint - ray.origin);

    HitInfo hit = RaySphere(
        ray, 
        Sphere(
            vec3(0. + 6.0 * cos(time/1.5 + 0.5), 0. + 6.0 * sin(time/0.5), 20.), 
            3.0,
            Material(vec4(1.0))
        )
    );

    color = vec4(
        ray.dir,
        1.0
    );

    color = vec4(abs(hit.normal), 1.0);

}



