#pragma once

#include <cuda_runtime.h>
#include "alphacompositing.cuh"
#include "mathHelper.cuh"

template<typename scalar_t, typename indice_t>
__host__ __device__
void gradBlending(scalar_t* gradOut, const indice_t* indices, const scalar_t* weights, const scalar_t* gradIn, unsigned int length)
{
    const scalar_t eps = 1e-4;
    blending(gradOut, indices, weights, length);

    for(unsigned int k = 0; k < length; k++)
    {
        indice_t kdx = indices[k];
        scalar_t gradValue = 0.0;
        for (unsigned int i = k; i < length; i++)
        {
            indice_t idx = indices[i];
            scalar_t denominator = (i == k) ? weights[kdx] : -(1 - weights[kdx]);
            if (abs<scalar_t>(denominator) > eps)
                gradValue += gradIn[idx] * gradOut[idx] / denominator;
        }
        gradOut[kdx] = gradValue;
    }
}