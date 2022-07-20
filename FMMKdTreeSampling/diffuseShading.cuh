#pragma once

#include <cuda_runtime.h>
#include "linalg.cuh"

template<typename real>
__host__ __device__
inline Vector3<real> lambert(const BaseVector3<real>& direction, const BaseVector3<real>& color, const BaseVector3<real>& normal, const BaseVector3<real>& diffuse)
{
	real cos = real(0);
	for (unsigned int i = 0; i < 3; i++)
		cos += direction[i] * normal[i];
	cos = max(real(0), cos); //abs<real>(cos);

	Vector3<real> radiance;
	for(unsigned int i = 0; i < 3; i++)
		radiance[i] = diffuse[i] / M_PI * color[i] * cos;
	return radiance;
}

template<typename real>
__host__ __device__
inline Vector3<real> shading(const real* directions, const real* colors, unsigned int numLights, const BaseVector3<real>& normal, const BaseVector3<real>& diffuse)
{
	Vector3<real> radiance;
	for (unsigned int i = 0; i < 3; i++)
		radiance[i] = real(0);

	for (unsigned int i = 0; i < numLights; i++)
	{
		const ConstVector3<real> direction(&directions[3 * i]);
		const ConstVector3<real> color(&colors[3 * i]);
		radiance += lambert(direction, color, normal, diffuse);
	}

	return radiance;
}

template<typename real>
__host__ __device__
Matrix44<real> buildShMatrix(const real* shCoefficients)
{
	const real c1 = 0.429043;
	const real c2 = 0.511664;
	const real c3 = 0.743125;
	const real c4 = 0.886227;
	const real c5 = 0.247708;

	Matrix44<real> M;
	M(0, 0) = c1 * shCoefficients[8];
	M(1, 0) = c1 * shCoefficients[4];
	M(2, 0) = c1 * shCoefficients[7];
	M(3, 0) = c2 * shCoefficients[3];

	M(0, 1) = c1 * shCoefficients[4];
	M(1, 1) = -c1 * shCoefficients[8];
	M(2, 1) = c1 * shCoefficients[5];
	M(3, 1) = c2 * shCoefficients[1];

	M(0, 2) = c1 * shCoefficients[7];
	M(1, 2) = c1 * shCoefficients[5];
	M(2, 2) = c3 * shCoefficients[6];
	M(3, 2) = c2 * shCoefficients[2];

	M(0, 3) = c2 * shCoefficients[3];
	M(1, 3) = c2 * shCoefficients[1];
	M(2, 3) = c2 * shCoefficients[2];
	M(3, 3) = c4 * shCoefficients[0] - c5 * shCoefficients[6];

	return M;
}

template<typename real>
__host__ __device__
real shadingSh(const real* shCoefficients, const BaseVector3<real>& normal)
{
	const real eps = 1e-5;
	real norm = normal.norm();

	//Vector4<real> n(normal, 1.0);
	Vector4<real> n(0.0, 0.0, 0.0, 1.0);
	if ( norm > eps )
	{
		real invNorm = 1.0/norm;
		n = Vector4<real>(normal, norm)*invNorm;
	}
	
	Matrix44<real> M = buildShMatrix(shCoefficients);
	real res = (n * M).dot(n);
    return res;
}