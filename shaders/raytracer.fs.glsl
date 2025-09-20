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
const int TRIANGLES_COUNT_MAX = 500000;
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
// uniform vec3 groundColor;

uniform int selectedMeshId;

uniform float depthOfFieldStrength;
uniform float antialiasStrength;

uniform int chunkx;
uniform int chunky;

uniform int chunksx;
uniform int chunksy;

uniform int MAX_CYCLES;

// uniform bool oneSidedTris;

float inf = 1.0 / 0.0;
float pi = 3.14159265359;

struct Material
{
    vec4 color; // 16 
    vec3 specularColor; // 12
    
    vec3 emissionColor; // 12
    float emissionStrength; // 4

    float specularStrength; // 4
    float smoothness; // 4
    float transmission; // 4
    float ior; // 4
    
    float metallic; // 4
    bool transparentFromBehind; // 4
};


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
    Triangle triangles[];
};

Triangle getTriangle(int index)
// legacy for implementation with multiple uniform buffers
{
    Triangle ret;

    return triangles[index];
}


Triangle getTriangleFromId(int triId) {
    for (int i = 0; i < triCount; i++) {
        Triangle tri = triangles[i];  // fixed typo
        if (tri.triId == triId) {
            return tri;
        }
    }
    // Return a default/empty triangle if not found
    Triangle emptyTri;
    emptyTri.triId = -1;  // mark as invalid
    return emptyTri;
}

uniform int nodeCount;

struct GraphNode {
    vec3 aabbMin;
    vec3 aabbMax;
    int id;
    int childId1;
    int childId2;
    int childObjOffset; // index into global childObj array
    int childObjCount;  // number of children
};

// layout(std140, binding=9) buffer triBuffer
layout(std140, binding=12) buffer GraphNodeBuffer {
    GraphNode graphNodes[];
};

layout(std430, binding=13) buffer BVHTriIdsBuffer {
    int childObjIds[]; // all child IDs concatenated
};

GraphNode getNodeFromNodeId(int nodeId) {
    // returns a node from a node id
    for (int i = 0; i < triCount; i++) {
        GraphNode obj = graphNodes[i];  // fixed typo
        if (obj.id == nodeId) {
            return obj;
        }
    }
    // Return a default/empty triangle if not found
    GraphNode obj_null;
    obj_null.id = -1;  // mark as invalid
    return obj_null;
}


const int MAX_TRIS_REQUESTED = 100;

Triangle[MAX_TRIS_REQUESTED] getTrianglesFromNode(GraphNode obj) {
    // returns an array of triangles from the start offset and length params in the 
    // graph node data structure
    Triangle[MAX_TRIS_REQUESTED] ret;

    for (int i = 0; i < obj.childObjCount; i++) {
        ret[i] = getTriangleFromId(childObjIds[obj.childObjOffset + i]);
    }
    return ret;
}


struct Mesh
{
    int index;
    int node0Id;
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
    vec3 debugInfo;
};

struct Ray 
{
    vec3 origin;
    vec3 dir;
    HitInfo prevHit;
    float ior;
    bool inSolid;

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
        vec3(0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        false
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
    
    // vec3 skyColor = vec3(0.5, 0.5, 1.0);
    vec3 skyColor = vec3(3.9216e-03, 6.9020e-01, 8.6275e-01);
    vec3 skyColor2 = vec3(1.0, 1.0, 1.0);
    vec3 groundColor = vec3(0.4, 0.4, 0.4);
    // vec3 groundColor = vec3(1.0, 0.0, 1.0);
    vec3 sunlightDirection = normalize(vec3(-1.0, 0.2, -1.0));
    float sunFocus = 100.0;
    float sunIntensity = 20.0;

    float skyGradientT = pow(smoothstep(0.0, 0.8, ray.dir.y), 0.35);
    float groundToSkyT = smoothstep(-0.01, 0.0, ray.dir.y);

    vec3 skyGradient = mix(skyColor2, skyColor, skyGradientT);
    float sun = pow(max(0.0, dot(ray.dir, sunlightDirection)), sunFocus) * sunIntensity;
    bool sunMask = groundToSkyT >= 1.0;

    return mix(groundColor, skyGradient, groundToSkyT) + sun * float(sunMask);

}

HitInfo rayTriangle(Ray ray, Triangle tri, bool hitBehind)
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
    if ((NDotRayDirection > 0.0) && (!hitBehind))
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    // set the flag if it hits the back of a facet
    if (NDotRayDirection > 0.0)
    {
        hitInfo.hitBackside = true;
    }

    // solves plane equation and ignore triangles behind the orig
    float d = -dot(N, v0);
    float t = -(dot(N, orig) + d) / NDotRayDirection;

    if (t < 0.0)
    {
        hitInfo.didHit = false;
        return hitInfo;
    }

    if (t < 1e-6)
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
    hitInfo.meshIndex = tri.meshIndex;

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
    // return true;
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

// uint hash(uint x) {
//     x ^= x >> 16;
//     x *= 0x7feb352dU;
//     x ^= x >> 15;
//     x *= 0x846ca68bU;
//     x ^= x >> 16;
//     return x;
// }

// float randomValue(inout uint state) {
//     state = hash(state);
//     return float(state) / float(0xffffffffU);
// }


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


// void rayGraphNodeCollision(Ray ray, GraphNode node, inout GraphNode[1024] g_arr, inout int index, inout int collisionsLen) {
//     bool check;

//     while g_arr[index].childId1
//     if (node.childId1 != -1) {
//         GraphNode child1 = getNodeFromNodeId(node.childId1);
//         check = boundingBoxIntersect(ray, child1.bboxMin, child1.bboxMax);
//         if check {
//             g_arr[index] = node;
//         }
//     }
//     else {
//         GraphNode child2 = getNodeFromNodeId(node.childId2);
//         check = boundingBoxIntersect(ray, node.bboxMin, node.bboxMax);
//         if check {
//             return child2;
//         }
//     }



//     // bool check;
//     // if (node.childId1 != -1) {
//     //     GraphNode child1 = getNodeFromNodeId(node.childId1);
//     //     check = boundingBoxIntersect(ray, child1.bboxMin, child1.bboxMax);
//     //     if check {
//     //         return child1;
//     //     }
//     // }
//     // else {
//     //     GraphNode child2 = getNodeFromNodeId(node.childId2);
//     //     check = boundingBoxIntersect(ray, node.bboxMin, node.bboxMax);
//     //     if check {
//     //         return child2;
//     //     }
//     // }
//     // // if we miss both, return a null one 
//     // GraphNode ret_null;
//     // ret_null.id = -1;
//     // return ret_null;
// }



vec3 idToColor(int n) {
    // Use large primes to scramble bits
    uint x = uint(n);
    x = (x ^ (x >> 16u)) * 0x45d9f3bu;
    x = (x ^ (x >> 16u)) * 0x45d9f3bu;
    x = (x ^ (x >> 16u));

    // Split into R, G, B channels (0–255)
    float r = float((x >>  0u) & 0xFFu) / 255.0;
    float g = float((x >>  8u) & 0xFFu) / 255.0;
    float b = float((x >> 16u) & 0xFFu) / 255.0;

    return vec3(r, g, b);
}


HitInfo calculateRayCollision(Ray ray, inout uint rngState) {
    // calculates the ray collisions for all 
    HitInfo closestHit = defaultHitInfo();

    bool doHitBackside;

    // loop over all spheres in the scene
    for (int i = 0; i < spheresCount; i++) {
        Sphere sphere = spheres[i];
        HitInfo hitInfo = raySphere(
            ray,
            sphere.position,
            sphere.radius
        );

        if (hitInfo.dst < 1e-6) {
            continue;
        }

        if ((hitInfo.didHit) && (hitInfo.dst < closestHit.dst)) {
            closestHit = hitInfo;
            closestHit.material = sphere.material;
        }
    }

    // loop over all of the meshesand their BVHs
    int bbox_check_count = 0;
    for (int j = 0; j < meshCount; j++) {
        
        GraphNode graphNodeStack[16]; // stack of node ids to check

        Mesh mesh = meshes[j];

        int node0Id = mesh.node0Id;

        GraphNode g = getNodeFromNodeId(mesh.node0Id);

        graphNodeStack[0] = g;

        int stackIndex = 1;
        // int checks = 0;
        // int max_checks = 10;

        while (stackIndex > 0)  {

        // for (int j = 0; j < 5; j++) {

            GraphNode node = graphNodeStack[--stackIndex];

            bool isvalid1 = (node.childId1 >= 0);
            bool isvalid2 = (node.childId2 >= 0);

            // check for collision against child 1, add it to the stack if it hits
            if (isvalid1) {
                GraphNode child1 = getNodeFromNodeId(node.childId1);
                bool check1 = boundingBoxIntersect(ray, child1.aabbMin, child1.aabbMax);
                if (check1) {
                    graphNodeStack[stackIndex++] = child1;
                }
            }
            // check for collision against child 2, add it to the stack if it hits
            if (isvalid2) {
                GraphNode child2 = getNodeFromNodeId(node.childId2);
                bool check2 = boundingBoxIntersect(ray, child2.aabbMin, child2.aabbMax);
                if (check2) {
                    graphNodeStack[stackIndex++] = child2;
                }
            } 
            // if either child was valid, we can skip the comparison
            if (isvalid1 || isvalid2) {
                continue;
            }

            bbox_check_count += 1;
            bool check = boundingBoxIntersect(ray, node.aabbMin, node.aabbMax);
            if (check) {

                // closestHit.debugInfo = idToColor(node.id) ; closestHit.didHit = true;
                // return closestHit;

                // closestHit.debugInfo.x = 1.0 ; closestHit.didHit = true;
                // return closestHit;
                Triangle[MAX_TRIS_REQUESTED] nodeTriangles = getTrianglesFromNode(node);
                
                for (int i = 0; i < node.childObjCount; i++) {

                    Triangle tri = nodeTriangles[i];
                    // Triangle tri = triangles[i];

                    // hitInfo.debugInfo.x = len / 10.0;

                    // the ray cannot reflect off the same one it reflected off!
                    if (tri.triId == ray.prevHit.facetId) {
                        continue;
                    }

                    // if the ray is in a solid (refraction, force enable backside reflection for facets of the same mesh.)
                    if (ray.inSolid) {
                        // if allow the object to hit the backside of its own mesh if it came off the same 
                        if (ray.prevHit.meshIndex == tri.meshIndex) {
                            doHitBackside = true;
                        }
                    } 
                    // otherwise allow hitting the back side of a surface if specified in material 
                    // (edge case of a back side surface within a transparent material)
                    else {
                        doHitBackside = !meshes[tri.meshIndex].material.transparentFromBehind;
                    }

                    // calculate the rayTriangle intersection 
                    HitInfo triHitInfo = rayTriangle(ray, tri, doHitBackside);
                    // HitInfo triHitInfo = rayTriangle(ray, tri, true);


                    // skip if the distance is too low
                    if (triHitInfo.dst < 1e-6) {
                        continue;
                    }

                    if ((triHitInfo.didHit) && (triHitInfo.dst < closestHit.dst)) {
                        closestHit = triHitInfo;
                        closestHit.material = meshes[tri.meshIndex].material;
                    }
                }
            }
        }
    }
    int threshold = 100;
    closestHit.debugInfo = vec3(1.0, 1.0, 1.0) * bbox_check_count / float(threshold);
    if (bbox_check_count > threshold) {
        closestHit.debugInfo = vec3(1.0, 0.0, 1.0);
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
float cosSchlickApproximation(float n1, float n2, float costheta)
{
    // function that calculates the reflection coefficient
    // i.e. how much goes to reflection, and the rest going to refraction
    float R0 = pow((n1 - n2) / (n1 + n2), 2);
    return R0 + (1 - R0) * pow(1 - costheta, 5);
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

float cosAngleBetweenVectors(vec3 v1, vec3 v2) {
    vec3 norm_v1 = normalize(v1);
    vec3 norm_v2 = normalize(v2);

    float dot_product = dot(norm_v1, norm_v2);

    // clamp for fear of floating point errors
    float cosangle = clamp(dot_product, -1.0, 1.0);

    return cosangle;
}


vec3 traceRay(inout Ray ray, inout uint rngState)
{
    vec3 incomingLight = vec3(0.0, 0.0, 0.0);
    vec3 rayColor = vec3(1.0,1.0,1.0);

    vec3 diffuseDir;
    vec3 specularDir;
    vec3 emittedLight;
    Material material;
    Material prevmaterial;
    // bool isSpecularBounce;
    // bool isTransmission;
    // vec3 mixinColor = vec3(1.0);
    // vec3 specularMixinColor = vec3(1.0);

    vec3 mixinColor;
    vec3 specularColor;
    vec3 diffuseColor;

    for (int i = 0; i <= MAX_BOUNCES; i++)
    {
        bool isTransmission = false;
        bool isSpecularBounce = false;

        // if (i == 0) { break; }
        HitInfo hitInfo = calculateRayCollision(ray, rngState);
        if (hitInfo.didHit)
        {

            prevmaterial = ray.prevHit.material;
            material = hitInfo.material;

            // deal with transparency by allowing some fraction of the rays to pass through
            // do no do any color modifications if it passes through.
            if ((hitInfo.material.color.w < 0.99999))
            {
                if (randomValue(rngState) > hitInfo.material.color.w)
                {
                    // move the ray origin to its hit point but dont modify direction, do not change ior
                    if (i == MAX_BOUNCES) { break; }
                    
                    ray.origin = hitInfo.hitPoint;
                    ray.prevHit = hitInfo;
                    continue;
                }
            }   

            emittedLight = material.emissionColor * material.emissionStrength;
            incomingLight += emittedLight * rayColor;

            // compensate normal for internal reflection if it hit the backside
            // -1 if hits back side
            // +1 if hits front side
            int normalFlip = -1 * (int(hitInfo.hitBackside) * 2 - 1);

            float n1;
            float n2;
            float eta;

            // if we hit the backside and were in solid, we are interfacting with atmosphere.
            if (ray.inSolid && hitInfo.hitBackside) {
                n2 = atmosphereMaterial.ior;
            // if not, we are interfacing with the material hit
            } else {
                n2 = material.ior;
            }

            n1 = ray.ior; // ior of the material the ray is in
            eta = n1 / n2;

            // technically there is a -1 * -1, I think? 
            specularDir = refract(ray.dir, normalFlip * hitInfo.normal, eta);
            // diffuse direction into the material
            diffuseDir = normalize(-1 * normalFlip * hitInfo.normal + randomDirection(rngState));

            // if (specularDir == vec3(0.0))
            // {
            //     rayColor = vec3(1.0, 0.0, 1.0);
                // continue;
            // }

            float costheta = cosAngleBetweenVectors(normalFlip * hitInfo.normal, -ray.dir);
            float shlickRatio = cosSchlickApproximation(n1, n2, costheta);


            // divert fraction of energy to specular probabilisically
            isSpecularBounce = (randomValue(rngState) < material.specularStrength);

            if (isSpecularBounce)
            {
                // if the ray is already within a solid, reflecting within a solid keeps it within a solid do not change ior
                // Maintain inSolid and ray.ior as it were before.

                // specular direction is now a reflect
                specularDir = reflect(ray.dir, hitInfo.normal * normalFlip);
                // invert the diffuse direction
                diffuseDir *= -1;
            }

            else
            {
                // at equal iors, diffuse only!
                // if (abs(n1-n2) < 1e-5) 
                // {
                //     isSpecularBounce = false;
                //     specularDir = reflect(ray.dir, hitInfo.normal);
                //     diffuseDir *= -1;
                // }       
                // if greater than the shlickRatio, it refracts successfully and enters the material
                // refraction of non-transmission will be a diffuse reflection
                if (randomValue(rngState) > shlickRatio)
                {
                    isSpecularBounce = false;
                    // isSpecularBounce = true;

                    if (randomValue(rngState) < material.transmission)
                    {
                        // the ray transmits through the material, i.e. transmissive material refraction
                        isTransmission = true;
                    }
                    else
                    {
                        // the ray does not transmit, invert the normal in preparation for diffuse reflection
                        // normalFlip *= -1;
                        specularDir = reflect(ray.dir, hitInfo.normal);
                        diffuseDir *= -1;
                    }

                    // on transmissive refraction into the material, the diffuse dir will enter the material...?
                    // diffuseDir = normalize(normalFlip * hitInfo.normal + randomDirection(rngState));

                    if (isTransmission) 
                    {
                        // if it hits the back side, we must assume it is refracting out into atmosphere
                        // handle transmission of the last hit before performing color calculations

                        vec3 transmissionColor;
                        Material transmissionMaterial = prevmaterial;
                        float attenuationCoeff = -log(transmissionMaterial.transmission);
                        transmissionColor = transmissionMaterial.color.xyz * exp(-attenuationCoeff* hitInfo.dst);
                        rayColor *= transmissionColor * transmissionMaterial.color.w;

                        if (hitInfo.hitBackside)
                        {
                            // mixinColor = atmosphereMaterial.color.xyz;
                            // mixinColor = vec3(1.0);
                            ray.inSolid = false;
                            ray.ior = atmosphereMaterial.ior;
                        } 
                        // if it refracts off an outside surface, it is entering the material
                        else
                        { 
                            // mixinColor = material.color.xyz;
                            ray.inSolid = true;
                            ray.ior = material.ior;
                        }
                    }
                    else
                    {
                        // diffuse reflection off the back side of a surface
                        // if it hits the back side, we must assume it is diffusely reflecting internally
                        if (hitInfo.hitBackside)
                        {
                            // mixinColor = material.color.xyz;
                            ray.inSolid = true;
                            ray.ior = material.ior;
                        } 
                        // if it hits the front side, it is plain diffuse reflection. maintain ray.inSolid and ior.
                        else
                        { 
                            // mixinColor = material.color.xyz;
                        }
                    }
                } 
                else
                {
                    isSpecularBounce = true;
                    specularDir = reflect(ray.dir, hitInfo.normal * normalFlip);
                    diffuseDir *= -1;
                }
            }
    


            if (i == MAX_BOUNCES) { break; }

            // move the ray origin to its hit point
            ray.origin = hitInfo.hitPoint;
            
            // bias direction based on smoothness, or use specular
            // ray.dir = mix(diffuseDir, specularDir, bool(material.smoothness * float(isSpecularBounce)));
            // rayColor *= mix(material.color.xyz, material.specularColor, isSpecularBounce);

            // isSpecularBounce = true;

            if (isSpecularBounce)
            {
                ray.dir = mix(diffuseDir, specularDir, material.smoothness);
                rayColor *= material.specularColor;
            }
            else
            {
                ray.dir = diffuseDir;
                rayColor *= material.color.xyz;
            }


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

    if (frameNumber > MAX_CYCLES)
    {
        color = texture(previousFrame, (fragPosition.xy + 1.0) / 2);
        return;
    }


    vec2 fragAbsPos = (fragPosition.xy + 1.0) * 0.5;

    // bool inx = (fragAbsPos.x >= (chunkx/(chunksx))) && (fragAbsPos.x < ((chunkx+1)/(chunksx)));
    // bool iny = (fragAbsPos.y >= (chunky/(chunksy))) && (fragAbsPos.y < ((chunky+1)/(chunksy)));

    bool inx = (fragAbsPos.x >= (chunkx / float(chunksx))) && (fragAbsPos.x < ((chunkx + 1) / float(chunksx)));
    bool iny = (fragAbsPos.y >= (chunky / float(chunksy))) && (fragAbsPos.y < ((chunky + 1) / float(chunksy)));

    // bool inx =  (fragAbsPos.x >= (chunkx / float(chunksx))) &&
    //             (fragAbsPos.x <= (chunkx + 1) / float(chunksx));
    // bool iny =  (fragAbsPos.y >= (chunky / float(chunksy))) &&
    //             (fragAbsPos.y <= (chunky + 1) / float(chunksy));


    // if not in x or y, color is the sample of the previous texture (black or previous render.)
    // if (!(inx && iny)) {
    if (!(inx && iny)) {
        color =
        texture(previousFrame, (fragPosition.xy + 1.0) / 2)
        // vec4(1.0)
        ;
        return;
    }
    
    
    uint numPixels = screenWidth * screenHeight;
    vec4 pxCoord = gl_FragCoord;
    // uint pxId = uint(pxCoord.x * screenWidth * screenHeight) + uint(pxCoord.y * screenHeight);
    // uint rngState = pxId + frameNumber;
    uint pxId = uint(pxCoord.y) * uint(screenWidth) + uint(pxCoord.x);
    uint rngState = pxId + uint(frameNumber * 747796405u); // prime multiplier


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
        newRay.ior = atmosphereMaterial.ior;

        totalIncomingLight += traceRay(newRay, rngState);
        // totalIncomingLight += traceRay(ray, rngState);

        float weight = 0.99;
        if (i == (RAYS_PER_PIXEL - 1))
        {
            // totalIncomingLight = totalIncomingLight * (1-weight) + weight * newRay.dir;
            totalIncomingLight = totalIncomingLight * (1-weight) + weight * newRay.prevHit.debugInfo;
            // totalIncomingLight = totalIncomingLight * (1-weight) + weight * abs(newRay.prevHit.normal);
            // totalIncomingLight = totalIncomingLight * (1-weight) + weight * (-newRay.prevHits.normal);
            // totalIncomingLight = totalIncomingLight * (1-weight) + weight * abs(newRay.prevHit.dst);
            // totalIncomingLight = totalIncomingLight * (1-weight) + weight * newRay.prevHit.material.color.xyz;
            // totalIncomingLight = totalIncomingLight * (1-weight) + weight * newRay.prevHit.material.color.xyz;
            // totalIncomingLight = totalIncomingLight * (1-weight) + weight * float(newRay.inSolid) * vec3(1.0, 0.0, 1.0);
            // totalIncomingLight = totalIncomingLight * (1-weight) + weight * float(!newRay.prevHit.hitBackside) * vec3(1.0, 0.0, 1.0);
        }
    }

    totalIncomingLight /= RAYS_PER_PIXEL;

    // HitInfo hit0 = rayTriangle(ray, getTriangle(0));
    // HitInfo hit1 = rayTriangle(ray, getTriangle(1));

    if (STATIC_RENDER)
    {
        float weight = 1.0 / (float(frameNumber));
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
    // if (STATIC_RENDER)
    // {
    //     if (frameNumber == 0)
    //     {
    //         color = vec4(0.0);
    //         return;
    //     }
    //     float alpha = 0.01;
    //     vec4 prev = texture(previousFrame, (fragPosition.xy + 1.0) / 2);
    //     vec4 samp = vec4(totalIncomingLight, 1.0);
    //     color = mix(prev, samp, alpha);
    // }
    else 
    {
        color = vec4(totalIncomingLight, 1.0)
        ;
    }

}



