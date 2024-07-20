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

uniform int MAX_BOUNCES;
uniform int RAYS_PER_PIXEL;

uniform int spheresCount;
uniform int triCount;
uniform int meshCount;

uniform sampler2D previousFrame;

uniform uint frameNumber;

uniform bool STATIC_RENDER;

// uniform vec3 skyColor;
uniform vec3 groundColor;

uniform int selectedMeshId;

uniform float depthOfFieldStrength;
uniform float antialiasStrength;

// uniform bool oneSidedTris;

float inf = 1.0 / 0.0;
float pi = 3.14159265359;

struct Material
{
    vec4 color; // 16 
    vec3 emissionColor; // 12
    float emissionStrength; // 4
    float smoothness; // 4
    float transmission; // 4
    float ior; // 4 + 4x
};

// uniform Material highlightMaterial;#



// layout(std140) uniform materialBuffer
// {
//     Material highlightMaterials[4];
// };



layout(std140) uniform materialBuffer
{
    Material atmosphereMaterial;
};

vec3 skyColor = atmosphereMaterial.color.xyz;
// vec3 skyColor = vec3(0.4, 0.4, 0.4);


struct Sphere
{
    vec3 position; // 12 + 4
    float radius; // 4 + 12
    Material material; // ...
};

// layout(std140)
layout(std140) uniform sphereBuffer 
{
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
    // Material material; // ...
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
    Material material;
};

layout(std140) uniform meshBuffer 
{
    Mesh meshes[MESHES_COUNT_MAX];
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

struct Ray 
{
    vec3 origin;
    vec3 dir;
    HitInfo prevHit;
    // float ior;
    // bool inSolid;

};


HitInfo defaultHitInfo() {
    HitInfo hitInfo;
    hitInfo.didHit = false;
    hitInfo.dst = 1e20;
    hitInfo.hitPoint = vec3(0.0);
    hitInfo.normal = vec3(0.0);
    hitInfo.material = Material(
        vec4(0.0),
        vec3(0.0),
        0.0,
        0.0,
        0.0,
        0.0
    );
    hitInfo.facetId = -1;
    hitInfo.hitBackside = false;
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
    // float skyGradientT = pow(smoothstep(0, 0.4, ray.dir.y), 0.35)
    // vec3 skyGradient = mix(skyColor, )

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
    // if (NDotRayDirection > 0.0)
    // {
    //     hitInfo.didHit = false;
    //     return hitInfo;
    // }

    // set the flag if it hits the back of a facet
    if (NDotRayDirection > 0.0)
    {
        hitInfo.hitBackside = true;
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
    hitInfo.hitPoint = t * ray.dir + ray.origin;

    hitInfo.facetId = tri.triId;

    return hitInfo;

}


// not sure what is going on here
// seems like the triangle colliusion fails when dst is set to inf

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

// TODO find a better implementation of a random number generator.

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



HitInfo calculateRayCollision(Ray ray, inout uint rngState)
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

        if ((hitInfo.didHit) && (hitInfo.dst < closestHit.dst))
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

        // int triMeshIndex = tri.meshIndex;

        // the ray cannot reflect off the same one it reflected off!
        if (tri.triId == ray.prevHit.facetId)
        {
            continue;
        }


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

        if ((hitInfo.didHit) && (hitInfo.dst < closestHit.dst))
        {
            closestHit = hitInfo;
            // closestHit.material = tri.material;
            closestHit.material = meshes[tri.meshIndex].material;
            
            // if (tri.meshIndex == selectedMeshId)
            // {
            //     closestHit.material = highlightMaterials[0];
            // }
            // else
            // {
            //     closestHit.material = tri.material;
            // }
        }
    }

    return closestHit;
}



float schlickApproximation(float n1, float n2, float theta)
{
    // function that calculates the reflection coefficient
    // i.e. how much goes to reflection, and the rest going to refraction
    float R0 = pow((n1 - n2) / (n1 + n2), 2);
    return R0 + (1 - R0) * pow(1 - cos(theta), 5);
}

float angleRefraction(float n1, float n2, float thetai)
{
    return asin(n1 / n2 * sin(thetai));
}

float angleBetweenVectors(vec3 v1, vec3 v2) {
    vec3 norm_v1 = normalize(v1);
    vec3 norm_v2 = normalize(v2);

    float dot_product = dot(norm_v1, norm_v2);

    // clamp for fear of floating point errors
    dot_product = clamp(dot_product, -1.0, 1.0);

    float angle = acos(dot_product);
    return angle;
}


vec3 traceRay(Ray ray, inout uint rngState)
{
    vec3 incomingLight = vec3(0.0, 0.0, 0.0);
    vec3 rayColor = vec3(1.0,1.0,1.0);

    vec3 diffuseDir;
    vec3 specularDir;
    vec3 emittedLight;
    Material material;


    for (int i = 0; i <= MAX_BOUNCES; i++)
    {
        HitInfo hitInfo = calculateRayCollision(ray, rngState);
        if (hitInfo.didHit)
        {
            // vec3 diffuseDir = normalize(hitInfo.normal + randomDirection(rngState));
            // vec3 specularDir = reflect(ray.dir, hitInfo.normal);

            material = hitInfo.material;

            // deal with transparency by allowing some fraction of the rays to pass through
            // do no do any color modifications if it passes through.
            if ((hitInfo.didHit) && (hitInfo.material.color.w < 0.99999))
            {
                if (randomValue(rngState) > hitInfo.material.color.w)
                {
                    // move the ray origin to its hit point but dont modify direction
                    ray.origin = hitInfo.hitPoint;
                    ray.prevHit = hitInfo;
                    continue;
                }
            }   

            emittedLight = material.emissionColor * material.emissionStrength;
            incomingLight += emittedLight * rayColor;

            // -1 if hits back side
            // +1 if hits front side
            int normalFlip = -1 * (int(hitInfo.hitBackside) * 2 - 1);


            if (material.transmission > 1e-6)
            {

                float n1 = ray.prevHit.material.ior;
                float n2 = material.ior;
                float eta = n1 / n2;

                // technically there is a -1 * -1, I think? 
                specularDir = refract(ray.dir, normalFlip * hitInfo.normal, eta);
                // specularDir = refract(ray.dir, -1 * hitInfo.normal, eta);

                if (specularDir == vec3(0.0))
                {
                    // change to reflect
                    // it reflects and picks up the material color
                    specularDir = reflect(ray.dir, hitInfo.normal);
                    diffuseDir = normalize(normalFlip * hitInfo.normal + randomDirection(rngState));
                    rayColor *= material.color.xyz;
                    
                } else {
                    float theta = angleBetweenVectors(-1 * normalFlip * hitInfo.normal, specularDir);
                    // I think this is correct?
                    float shlickRatio = schlickApproximation(n1, n2, theta);
                    
                    // if greater than the shlickRatio, it refracts successfully
                    if (randomValue(rngState) > shlickRatio)
                    {
                        // it refracts successfully
                        vec3 transmissionColor;
                        Material transmissionMaterial = ray.prevHit.material;
                        float attenuationCoeff = -log(transmissionMaterial.transmission);
                        transmissionColor = 
                        transmissionMaterial.color.xyz * exp(-attenuationCoeff* hitInfo.dst)
                        ;
                        // multiply it by the transmission color before doing the other calculations
                        rayColor *= transmissionColor * transmissionMaterial.color.w;

                        
                        diffuseDir = normalize(-1 * normalFlip * hitInfo.normal + randomDirection(rngState));
                        // if it hits the back side, refract is entering the atmosphere materia
                        // if it hits the front side, it picks up the material colour
                        if (hitInfo.hitBackside)
                        {
                            // exiting the material
                            rayColor *= vec3(1.0);
                        } else
                        {
                            // entering the material
                            rayColor *= material.color.xyz;
                        }

                    } else
                    {
                        // it reflects and picks up the material color
                        specularDir = reflect(ray.dir, hitInfo.normal);
                        diffuseDir = normalize(normalFlip * hitInfo.normal + randomDirection(rngState));
                        rayColor *= material.color.xyz;
                    }
                }

            } else
            {
                // standard reflection off the material
                specularDir = reflect(ray.dir, hitInfo.normal);
                diffuseDir = normalize(normalFlip * hitInfo.normal + randomDirection(rngState));
                rayColor *= material.color.xyz;
            }

            // move the ray origin to its hit point
            ray.origin = hitInfo.hitPoint;
            ray.dir = mix(diffuseDir, specularDir, material.smoothness);
            ray.prevHit = hitInfo;

        } else 
        {
            // the ray misses all objects, pick up the environment color
            incomingLight += getEnvironmentLight(ray) * rayColor;
            break;
        }

    }
    
    return incomingLight;
}



vec4 toneMap(vec4 color) 
{
    return color / (color + vec4(1.0));
}
vec4 gammaCorrect(vec4 color, float gamma)
{
    return pow(color, vec4(1.0 / gamma));
}

void main() 
{

    uint numPixels = screenWidth * screenHeight;
    vec4 pxCoord = gl_FragCoord;
    uint pxId = uint(pxCoord.x * screenWidth * screenHeight) + uint(pxCoord.y * screenHeight);
    uint rngState = pxId + frameNumber;

    // calculate the camere bits
    vec3 viewPointLocal = (vec3(fragPosition.xy / 2.0, 1) * ViewParams);
    vec3 viewPoint = (transpose(CamLocalToWorldMatrix) * vec4(viewPointLocal.xyz, 1.0)).xyz;

    // Btw we can just interpolate the ray direction from the vertex shader.
    Ray ray;
    ray.origin = CamGlobalPos;
    ray.dir = normalize(viewPoint);



    float camNearPlane = ViewParams.z;

    vec3 totalIncomingLight = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < RAYS_PER_PIXEL; i++)
    {
        float planeWidth = ViewParams.x;
        Ray newRay;
        vec3 depthOfFieldOffset = (vec3(randomValue(rngState), randomValue(rngState), 0.0) * 2.0 - 1.0) * depthOfFieldStrength * camNearPlane;
        vec3 antialiasOffset = (vec3(randomValue(rngState), randomValue(rngState), 0.0) * 2.0 - 1.0) * antialiasStrength;

        // make a new ray with a slightly deviated angle. 2.0 * 0.3 looks pretty nice?
        // newRay.dir = normalize(rayVec + antialiasOffset - depthOfFieldOffset);
        // newRay.origin = ray.origin.xyz + depthOfFieldOffset;

        newRay.dir = normalize(viewPoint + depthOfFieldOffset + antialiasOffset);
        newRay.origin = ray.origin.xyz - depthOfFieldOffset;
        newRay.prevHit = defaultHitInfo();
        newRay.prevHit.material = atmosphereMaterial;

        totalIncomingLight += traceRay(newRay, rngState);
        // totalIncomingLight += traceRay(ray, rngState);
    }

    totalIncomingLight /= RAYS_PER_PIXEL;

    HitInfo hit0 = rayTriangle(ray, getTriangle(0));
    HitInfo hit1 = rayTriangle(ray, getTriangle(1));


    if (STATIC_RENDER)
    {
        float weight = 1.0 / (float(frameNumber) + 1);
        color = 
        texture(previousFrame, (fragPosition.xy + 1.0) / 2) * (1-weight)
        +
        // gammaCorrect(
        // toneMap(
        vec4(totalIncomingLight, 1.0) * weight
        // )
        // , 1.0)
        ;
    }
    else 
    {
        color = vec4(normalize(totalIncomingLight), 1.0)
        ;
    }

}



