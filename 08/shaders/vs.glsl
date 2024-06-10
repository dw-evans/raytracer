
#version 330 core

precision highp float;

layout (location = 0) in vec3 position;

struct Ray 
{
    vec3 origin;
    vec3 dir;
};


// view params includes width(x), height(y) and the plane depth(z)
uniform vec3 ViewParams;
uniform mat4x4 CamLocalToWorldMatrix;
uniform vec3 CamGlobalPos;
uniform float time;

out vec4 vertexColour;

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
HitInfo RaySphere(Ray ray, vec3 sphereCentre, float sphereRadius) 
{
    HitInfo hitInfo = InitializeHitInfo();

    vec3 offsetRayOrigin = ray.origin - sphereCentre;

    // float a = dot(ray.dir, ray.dir);
    // float b = 2 * dot(offsetRayOrigin, ray.dir);
    // float c = dot(offsetRayOrigin, offsetRayOrigin) - sphereRadius * sphereRadius;

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
            hitInfo.normal = normalize(hitInfo.hitPoint - sphereCentre);
        }
    }
    return hitInfo;
}


// float my_abs(float fin) {
//     if (fin < 0.0) {
//         return -1.0 * fin;
//     }
//     return fin;
// }

void main() {
    gl_Position = vec4(position.x,  position.y, position.z, 1.0);

    vec3 viewPointLocal = (vec3(position.xy / 2, 1) * ViewParams);
    vec3 viewPoint = (CamLocalToWorldMatrix * vec4(viewPointLocal.xyz, 1.0)).xyz;

    Ray ray;
    ray.origin = CamGlobalPos;
    ray.dir = normalize(viewPoint - ray.origin);

    HitInfo hit = RaySphere(ray, vec3(0., 0., 5.0), 1.0);
    vertexColour = vec4(hit.didHit);

    // if (hit.didHit) {
    //     vertexColour = vec4(1.0);
    // } else {
    //     vertexColour = vec4(0.0);
    // }

    // vertexColour = vec4(
    //     position.xy, 0.0,
    //     1.0
    // );
    // vertexColour = vec4(
    //     viewPointLocal.xy, 0.0,
    //     1.0
    // );
    // vertexColour = vec4(
    //     viewPoint.xy, 0.0,
    //     1.0
    // );
    // vertexColour = vec4(
    //     ray.dir.x, 0.0, 0.0,
    //     1.0
    // );
    vertexColour = vec4(
        abs(position.x), 0.0, 0.0,
        1.0
    );

    // if (position.x < 0.0) {
    //     vertexColour = vec4(
    //         position.x * -1.0, 0.0, 0.0,
    //         1.0
    //     );
    // } else {
    //     vertexColour = vec4(
    //         position.x, 0.0, 0.0,
    //         1.0
    //     );
    // }


    // vertexColour = vec4(
    //     abs(hit.normal),
    //     1.0
    // );




}



