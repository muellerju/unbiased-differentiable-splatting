#pragma once

#include <cuda_runtime.h>
#include "linalg.cuh"

template<typename real>
__host__ __device__
inline real gradLambert(
	Matrix33<real>& gradDirection, Matrix33<real>& gradColor,
	Matrix33<real>& gradNormal, Matrix33<real>& gradDiffuse,
	const BaseVector3<real>& direction, const BaseVector3<real>& color,
	const BaseVector3<real>& normal, const BaseVector3<real>& diffuse)
{
	real cos = real(0);
	Vector3<real> diffCo;
	for (unsigned int i = 0; i < 3; i++)
	{
		cos += direction[i]*normal[i];
		diffCo[i] = diffuse[i] * color[i];
	}
	real scaling = max(real(0), cos) / M_PI;

	if (cos > 0)
	{
		gradDirection = outer(diffCo, normal) / M_PI;
		gradColor = DiagMatrix33<real>(diffuse.data_ptr()) * scaling;

		gradNormal = outer(diffCo, direction) / M_PI;
		gradDiffuse = DiagMatrix33<real>(color.data_ptr()) * scaling;
	}

	return cos;
}

template<typename real>
__device__
inline void atomicVectorAdd(real* outPtr, const BaseVector3<real>& in)
{
	for (unsigned int i = 0; i < 3; i++)
		atomicAdd(&outPtr[i], in[i]);
}

template<typename real>
__device__
inline void gradShading(
	real* gradOutDirections, real* gradOutColors,
	PointerVector3<real>& gradOutNormal, PointerVector3<real>& gradOutDiffuse,
	const BaseVector3<real>& gradIn,
	const real* directions, const real* colors, unsigned int numLights, 
	const BaseVector3<real>& normal, const BaseVector3<real>& diffuse)
{		
	for (unsigned int i = 0; i < numLights; i++)
	{
		const ConstVector3<real> direction(&directions[3 * i]);
		const ConstVector3<real> color(&colors[3 * i]);

		real* gradOutDirection = &gradOutDirections[3 * i];
		real* gradOutColor = &gradOutColors[3 * i];

		Matrix33<real> gradDirection, gradColor, gradNormal, gradDiffuse;
		real cos = gradLambert(gradDirection, gradColor, gradNormal, gradDiffuse, direction, color, normal, diffuse);

		if (cos > 0)
		{
			Vector3<real> pixelGradDirection = gradIn * gradDirection;
			Vector3<real> pixelGradColor = gradIn * gradColor;
			atomicVectorAdd(gradOutDirection, pixelGradDirection);
			atomicVectorAdd(gradOutColor, pixelGradColor);

			gradOutNormal += gradIn * gradNormal;
			gradOutDiffuse += gradIn * gradDiffuse;
		}
	}

}

template<typename real>
__host__ __device__
void gradBuildShMatrix(real* gradShCoefficients, const Matrix44<real>& gradM)
{
	const real c1 = 0.429043;
	const real c2 = 0.511664;
	const real c3 = 0.743125;
	const real c4 = 0.886227;
	const real c5 = 0.247708;

	gradShCoefficients[0] = c4 * gradM(3, 3);
	gradShCoefficients[1] = c2 * (gradM(1, 3) + gradM(3, 1));
	gradShCoefficients[2] = c2 * (gradM(3, 2) + gradM(2, 3));
	gradShCoefficients[3] = c2 * (gradM(3, 0) + gradM(0, 3));
	gradShCoefficients[4] = c1 * (gradM(1, 0) + gradM(0, 1));
	gradShCoefficients[5] = c1 * (gradM(2, 1) + gradM(1, 2));
	gradShCoefficients[6] = c3 * gradM(2, 2) - c5 * gradM(3, 3);
	gradShCoefficients[7] = c1 * (gradM(2, 0) + gradM(0, 2));
	gradShCoefficients[8] = c1 * (gradM(0, 0) - gradM(1, 1));
}

template<typename real>
__host__ __device__
void gradShadingSh(real* gradShCoefficients, Vector3<real>& gradNormal, const real* shCoefficients, const BaseVector3<real>& normal)
{
	const real eps = 1e-5;

	// Partial forward pass
	Matrix44<real> M = buildShMatrix(shCoefficients);
	real norm = normal.norm();
	Vector4<real> n(0.0, 0.0, 0.0, 1.0);
	if ( norm > eps )
	{
		real invNorm = 1.0/norm;
		n = Vector4<real>(normal, norm)*invNorm;
	}

	// Compute derivative of the normalization operation
	Matrix33<real> gradNormalize = Matrix33<real>::zeros();
	if (norm > eps)
	{
		real invNorm = 1.0/norm;
		gradNormalize = DiagMatrix33<real>(invNorm) - outer(normal,normal)*power<real,3>(invNorm);
	}
	
	// Compute gradient of the spherical harmonics operation
	Vector4<real> gradN = M * n + M.transposed() * n;
	Matrix44<real> gradM = outer(n, n);

	// Compute final gradient 
	gradBuildShMatrix(gradShCoefficients, gradM);
	gradNormal = gradNormalize * gradN.xyz();
}