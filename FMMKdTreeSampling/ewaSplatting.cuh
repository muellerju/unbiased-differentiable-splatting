#pragma once
#include <cuda_runtime.h>

#include "mathHelper.cuh"
#include "linalg.cuh"
#include "transformation.cuh"

template<typename real>
__host__ __device__
Matrix22<real> constructInvJacobian(const Vector3<real>& x0Tilde, const Vector3<real>& x1Tilde, const BaseVector3<real>& camNormal)
{
	Vector3<real> u0 = x0Tilde.normalized();
	Vector3<real> u1 = u0.cross(camNormal);

	// Construct Jacobian
	Matrix22<real> invJk;
	invJk(0, 0) = x0Tilde.norm();
	invJk(0, 1) = x1Tilde.dot(u0);
	invJk(1, 0) = 0.0;
	invJk(1, 1) = x1Tilde.dot(u1);
	return invJk;
}

template<typename real>
__host__ __device__
Matrix22<real> buildInverseJacobian(
	real width, real height, real focallength, 
	const Vector2<real>& screenpoint, 
	const BaseVector3<real>& camPoint, 
	const BaseVector3<real>& camNormal)
{
	Vector2<real> y0 = screenpoint.copy();
	y0[0] += 1.0;
	Vector2<real> y1 = screenpoint.copy();
	y1[1] += 1.0;

	Vector3<real> x0 = backproject(width, height, focallength, y0);
	Vector3<real> x1 = backproject(width, height, focallength, y1);

	Vector3<real> x0Tilde = projectOntoPlane(x0, camPoint, camNormal) - camPoint;
	Vector3<real> x1Tilde = projectOntoPlane(x1, camPoint, camNormal) - camPoint;
	
	return constructInvJacobian(x0Tilde, x1Tilde, camNormal);
}

template<typename real>
__host__ __device__
Matrix22<real> buildCovariance(const Matrix22<real>& invJk, const BaseVector2<real>& stdDev, real smoothing)
{
	Matrix22<real> Jk = invJk.inv();
	DiagMatrix22<real> D(stdDev);
	DiagMatrix22<real> I(smoothing);
	return (Jk * D.squared() * Jk.transposed()) + I;
}

template<typename real>
__host__ __device__
real evaluteGaussian(const BaseVector2<real>& pixel, const Vector2<real>& screenpoint, const Matrix22<real>& invJk, const Matrix22<real>& Vk)
{
	Vector2<real> x = pixel - screenpoint;
	Matrix22<real> invVk = Vk.inv();

	real exponent = (x * invVk) * x;
	real normalization = 2.0 * M_PI * sqrt<real>(Vk.det()) * abs<real>(invJk.det());
	
	return exp<real>(-0.5 * exponent) / normalization; // TODO should be expf
}

template<typename real>
__host__ __device__
real evaluateEWAWeight(
	const BaseVector2<real>& pixel, 
	real width, real height, real focallength, 
	const BaseVector3<real>& camPoint, 
	const BaseVector3<real>& camNormal, 
	const BaseVector2<real>& stdDev, 
	real smoothing
)
{
	Vector2<real> screenpoint = toScreenspace(width, height, focallength, camPoint);

	// Build inverse jacobian
	Matrix22<real> invJk = buildInverseJacobian<real>(width, height, focallength, screenpoint, camPoint, camNormal);

	// Build covariance matrix
	Matrix22<real> Vk = buildCovariance<real>(invJk, stdDev, smoothing);

	// Evaluate gaussian weight
	real weight = evaluteGaussian<real>(pixel, screenpoint, invJk, Vk);
	return min(weight, 1.0);
}
