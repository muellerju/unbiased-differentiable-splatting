#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "cudaHelper.cuh"

template<typename real, unsigned int Dims>
struct alignas(16) AlignedBox
{
	real lower[Dims];
	real upper[Dims];
};


template<typename real, unsigned int Dims>
__host__ __device__
void mergeBoxes(AlignedBox<real, Dims>& c, const AlignedBox<real, Dims>& a, const AlignedBox<real, Dims>& b)
{
    for (unsigned int i = 0; i < Dims; i++)
        c.lower[i] = min(a.lower[i], b.lower[i]);
    for (unsigned int i = 0; i < Dims; i++)
        c.upper[i] = max(a.upper[i], b.upper[i]);
}

template<typename real, unsigned int Dims>
__host__ __device__
real maxNormFromBbox(const AlignedBox<real, Dims>& bbox, const real* point)
{
    real radius = real(0);
    for (unsigned int i = 0; i < Dims; i++)
    {
        real x = max(point[i] - bbox.lower[i], bbox.upper[i] - point[i]);
        radius = max(radius, x);
    }
    return radius;
}

template<typename real, unsigned int Dims>
__host__ __device__
real longestSidelengthBBox(const AlignedBox<real, Dims>& bbox)
{
    real radius = real(0);
    for (unsigned int i = 0; i < Dims; i++)
    {
        radius = max(radius, abs(bbox.upper[i] - bbox.lower[i]));
    }
    return radius;
}

template<typename real, unsigned int Dims>
__host__ __device__
bool intersectBboxes(const AlignedBox<real, Dims>& a, const AlignedBox<real, Dims>& b)
{
    bool intersect = false;
    for (unsigned int i = 0; i < Dims; i++)
    {
        // Side length and center of box A
        real sideLengthA = a.upper[i] - a.lower[i];
        real centerA = sideLengthA / 2 + a.lower[i];
        // Side length and center of box B
        real sideLengthB = b.upper[i] - b.lower[i];
        real centerB = sideLengthB / 2 + b.lower[i];
        // Check for intersection along the current axis
        if (abs(centerA - centerB) * 2 < (sideLengthA + sideLengthB))
            intersect = true;
    }
    return intersect;
}

template<typename real, unsigned int Dims>
__host__ __device__
real seperatingPlaneFromBboxes(const AlignedBox<real, Dims>& left, const AlignedBox<real, Dims>& right, unsigned int axis)
{
    return (right.lower[axis] - left.upper[axis]) / (real)2.0;
}


__host__ __device__
inline float clamp(float x, float minVal, float maxVal)
{
    return fminf(fmaxf(x, minVal), maxVal);
}

template<typename real, unsigned int Dims>
__host__ __device__
bool intersectsCircle(const AlignedBox<real, Dims>& box, const real* center, real radius)
{
    real distanceSquared = (real)0.0;
    for (unsigned int i = 0; i < Dims; i++)
    {
        real closest = clamp(center[i], box.lower[i], box.upper[i]);
        // Calculate the distance between the circle's center and this closest point
        real distance = center[i] - closest;
        distanceSquared += (distance * distance);
    }
    return distanceSquared < (radius* radius);
}
