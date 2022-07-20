#pragma once

#include <cuda_runtime.h>
#include "transformation.cuh"

template<typename scalar_t, typename indice_t>
__host__ __device__
inline void signedDistances(
    scalar_t* distances, 
    const BaseVector2<scalar_t>& pixel,
    scalar_t width, scalar_t height, scalar_t focallength,
    unsigned int numSamples,
    const indice_t* indices, 
    const scalar_t* points, 
    const scalar_t* normals, 
    scalar_t zmax = 10000.0)
{
    for (unsigned int k = 0; k < numSamples; k++)
    {
        distances[k] = zmax;

        indice_t idx = indices[k];
        if (idx < 0)
            continue;
        
        unsigned int offset = 3*idx; // Allows for broadcasting from shape (1,n,3) -> (bn,w,h,k) //3 * (idx + numSources * b);
        const ConstVector3<scalar_t> camPoint(&points[offset]);
        const ConstVector3<scalar_t> camNormal(&normals[offset]);

        // Compute the depth per pixel and store the intermediate result in the output tensor 
        Vector3<scalar_t> worldDirection = backproject(width, height, focallength, pixel);
        scalar_t distance = projectionDistance(worldDirection.normalized() * -1.0, camPoint, camNormal);
        if (distance > 0.0)
            distances[k] = distance;
    }
}

template<typename scalar_t, typename indice_t>
__host__ __device__
inline void argSort(indice_t* indices, const scalar_t* values, unsigned int length)
{
    for (unsigned int i = 1; i < length; i++)
    {
        indice_t index = indices[i];
        scalar_t value = values[index];
        unsigned int j = i;
        while ((j > 0) && (values[indices[j - 1]] > value))
        {
            indices[j] = indices[j - 1];
            j -= 1;
        }
        indices[j] = index;
    }
}

template<typename scalar_t, typename indice_t>
__host__ __device__
inline void blending(scalar_t* out, const indice_t* indices, const scalar_t* in, unsigned int length)
{
    scalar_t remainder = 1.0;
    for (unsigned int i = 0; i < length; i++)
    {
        indice_t idx = indices[i];
        scalar_t weight = in[idx];
        out[idx] = remainder * weight;
        remainder *= (1.0 - weight);
    }
}
