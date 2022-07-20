#pragma once

#include <cuda_runtime.h>

#include "linalg.cuh"
#include "ewaSplatting.cuh"
#include "transformationGrad.cuh"

template<typename real>
__host__ __device__
void gradConstructInvJacobian(Tensor322<real>& gradx0Tilde, Tensor322<real>& gradx1Tilde, Tensor322<real>& gradcamNormal, const Vector3<real>& x0Tilde, const Vector3<real>& x1Tilde, const BaseVector3<real>& camNormal)
{
	real invNorm = 1.0 / x0Tilde.norm();
	real invPowNorm = 1.0 / (x0Tilde.norm() * x0Tilde.dot(x0Tilde));

	gradx0Tilde(0, 0, 0) = x0Tilde[0] * invNorm;
	gradx0Tilde(0, 0, 1) = (-x0Tilde[0] * x0Tilde[1] * x1Tilde[1] - x0Tilde[0] * x0Tilde[2] * x1Tilde[2] + square(x0Tilde[1]) * x1Tilde[0] + square(x0Tilde[2]) * x1Tilde[0]) * invPowNorm;
	gradx0Tilde(0, 1, 0) = 0;
	gradx0Tilde(0, 1, 1) = (x0Tilde[0] * x1Tilde[0] * (camNormal[1] * x0Tilde[2] - camNormal[2] * x0Tilde[1]) - x1Tilde[1] * (camNormal[2] * (square(x0Tilde[0]) + square(x0Tilde[1]) + square(x0Tilde[2])) + x0Tilde[0] * (camNormal[0] * x0Tilde[2] - camNormal[2] * x0Tilde[0])) + x1Tilde[2] * (camNormal[1] * (square(x0Tilde[0]) + square(x0Tilde[1]) + square(x0Tilde[2])) + x0Tilde[0] * (camNormal[0] * x0Tilde[1] - camNormal[1] * x0Tilde[0]))) * invPowNorm;
	gradx0Tilde(1, 0, 0) = x0Tilde[1] * invNorm;
	gradx0Tilde(1, 0, 1) = (square(x0Tilde[0]) * x1Tilde[1] - x0Tilde[0] * x0Tilde[1] * x1Tilde[0] - x0Tilde[1] * x0Tilde[2] * x1Tilde[2] + square(x0Tilde[2]) * x1Tilde[1]) * invPowNorm;
	gradx0Tilde(1, 1, 0) = 0;
	gradx0Tilde(1, 1, 1) = (-x0Tilde[1] * x1Tilde[1] * (camNormal[0] * x0Tilde[2] - camNormal[2] * x0Tilde[0]) + x1Tilde[0] * (camNormal[2] * (square(x0Tilde[0]) + square(x0Tilde[1]) + square(x0Tilde[2])) + x0Tilde[1] * (camNormal[1] * x0Tilde[2] - camNormal[2] * x0Tilde[1])) - x1Tilde[2] * (camNormal[0] * (square(x0Tilde[0]) +
		square(x0Tilde[1]) + square(x0Tilde[2])) + x0Tilde[1] * (-camNormal[0] * x0Tilde[1] + camNormal[1] * x0Tilde[0]))) * invPowNorm;
	gradx0Tilde(2, 0, 0) = x0Tilde[2] * invNorm;
	gradx0Tilde(2, 0, 1) = (square(x0Tilde[0]) * x1Tilde[2] - x0Tilde[0] * x0Tilde[2] * x1Tilde[0] + square(x0Tilde[1]) * x1Tilde[2] - x0Tilde[1] * x0Tilde[2] * x1Tilde[1]) * invPowNorm;
	gradx0Tilde(2, 1, 0) = 0;
	gradx0Tilde(2, 1, 1) = (x0Tilde[2] * x1Tilde[2] * (camNormal[0] * x0Tilde[1] - camNormal[1] * x0Tilde[0]) - x1Tilde[0] * (camNormal[1] * (square(x0Tilde[0]) + square(x0Tilde[1]) + square(x0Tilde[2])) + x0Tilde[2] * (-camNormal[1] * x0Tilde[2] + camNormal[2] * x0Tilde[1])) + x1Tilde[1] * (camNormal[0] * (square(x0Tilde[0]) +
		square(x0Tilde[1]) + square(x0Tilde[2])) + x0Tilde[2] * (-camNormal[0] * x0Tilde[2] + camNormal[2] * x0Tilde[0]))) * invPowNorm;

	gradx1Tilde(0, 0, 0) = 0;
	gradx1Tilde(0, 0, 1) = x0Tilde[0] * invNorm;
	gradx1Tilde(0, 1, 0) = 0;
	gradx1Tilde(0, 1, 1) = x0Tilde[1] * camNormal[2] * invNorm - x0Tilde[2] * camNormal[1] * invNorm;
	gradx1Tilde(1, 0, 0) = 0;
	gradx1Tilde(1, 0, 1) = x0Tilde[1] * invNorm;
	gradx1Tilde(1, 1, 0) = 0;
	gradx1Tilde(1, 1, 1) = -x0Tilde[0] * camNormal[2] * invNorm + x0Tilde[2] * camNormal[0] * invNorm;
	gradx1Tilde(2, 0, 0) = 0;
	gradx1Tilde(2, 0, 1) = x0Tilde[2] * invNorm;
	gradx1Tilde(2, 1, 0) = 0;
	gradx1Tilde(2, 1, 1) = x0Tilde[0] * camNormal[1] * invNorm - x0Tilde[1] * camNormal[0] * invNorm;

	gradcamNormal(0, 0, 0) = 0;
	gradcamNormal(0, 0, 1) = 0;
	gradcamNormal(0, 1, 0) = 0;
	gradcamNormal(0, 1, 1) = -x0Tilde[1] * x1Tilde[2] * invNorm + x0Tilde[2] * x1Tilde[1] * invNorm;
	gradcamNormal(1, 0, 0) = 0;
	gradcamNormal(1, 0, 1) = 0;
	gradcamNormal(1, 1, 0) = 0;
	gradcamNormal(1, 1, 1) = x0Tilde[0] * x1Tilde[2] * invNorm - x0Tilde[2] * x1Tilde[0] * invNorm;
	gradcamNormal(2, 0, 0) = 0;
	gradcamNormal(2, 0, 1) = 0;
	gradcamNormal(2, 1, 0) = 0;
	gradcamNormal(2, 1, 1) = -x0Tilde[0] * x1Tilde[1] * invNorm + x0Tilde[1] * x1Tilde[0] * invNorm;
}

template<typename real>
__host__ __device__
void gradJacobian(
	Tensor222<real>& gradScreenpoint, 
	Tensor322<real>& gradCamPoint, 
	Tensor322<real>& gradCamNormal, 
	real width, real height, real focallength, 
	const Vector2<real>& screenpoint, 
	const BaseVector3<real>& camPoint, 
	const BaseVector3<real>& camNormal)
{
	// Perform forward step again
	Vector2<real> y0 = screenpoint.copy();
	y0[0] += 1.0;
	Vector2<real> y1 = screenpoint.copy();
	y1[1] += 1.0;

	Vector3<real> x0 = backproject(width, height, focallength, y0);
	Vector3<real> x1 = backproject(width, height, focallength, y1);

	Vector3<real> x0Tilde = projectOntoPlane(x0, camPoint, camNormal) - camPoint;
	Vector3<real> x1Tilde = projectOntoPlane(x1, camPoint, camNormal) - camPoint;

	Matrix22<real> invJk = constructInvJacobian(x0Tilde, x1Tilde, camNormal);

	// Compute gradient
	Tensor322<real> dInvJkdX0Tilde, dInvJkdX1Tilde, dInvJkdNormal;
	gradConstructInvJacobian(dInvJkdX0Tilde, dInvJkdX1Tilde, dInvJkdNormal, x0Tilde, x1Tilde, camNormal);

	Matrix33<real> dx0Tildedx0, dx0TildedcamPoint, dx0TildedcamNormal;
	gradProjectOntoPlane(dx0Tildedx0, dx0TildedcamPoint, dx0TildedcamNormal, x0, camPoint, camNormal);
	Matrix33<real> dx1Tildedx1, dx1TildedcamPoint, dx1TildedcamNormal;
	gradProjectOntoPlane(dx1Tildedx1, dx1TildedcamPoint, dx1TildedcamNormal, x1, camPoint, camNormal);

	Matrix23<real> dx0dy0 = gradBackproject(width, height, focallength, y0);
	Matrix23<real> dx1dy1 = gradBackproject(width, height, focallength, y1);

	gradScreenpoint = dInvJkdX0Tilde * dx0Tildedx0.transposed() * dx0dy0 + dInvJkdX1Tilde * dx1Tildedx1.transposed() * dx1dy1;
	gradCamPoint = dInvJkdX0Tilde * (dx0TildedcamPoint - DiagMatrix33<real>(1.0)).transposed() + dInvJkdX1Tilde * (dx1TildedcamPoint - DiagMatrix33<real>(1.0)).transposed();
	gradCamNormal = dInvJkdNormal + dInvJkdX0Tilde * dx0TildedcamNormal.transposed() + dInvJkdX1Tilde * dx1TildedcamNormal.transposed();

}

template<typename real>
__host__ __device__
void gradCovariance(Tensor2222<real>& gradInvJk, Tensor222<real>& gradStdDev, const Matrix22<real>& invJk, const BaseVector2<real>& stdDev, real smoothing)
{
	real cubedInvDet = 1.0 / power<real, 3>(invJk.det());
	gradInvJk(0, 0, 0, 0) = -2 * invJk(1, 1) * (square(invJk(0, 1)) * square(stdDev[1]) + square(invJk(1, 1)) * square(stdDev[0])) * cubedInvDet;
	gradInvJk(0, 0, 0, 1) = (-invJk(0, 1) * square(stdDev[1]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0)) + 2 * invJk(1, 1) * (invJk(0, 0) * invJk(0, 1) * square(stdDev[1]) + invJk(1, 0) * invJk(1, 1) * square(stdDev[0]))) * cubedInvDet;
	gradInvJk(0, 0, 1, 0) = (-invJk(0, 1) * square(stdDev[1]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0)) + 2 * invJk(1, 1) * (invJk(0, 0) * invJk(0, 1) * square(stdDev[1]) + invJk(1, 0) * invJk(1, 1) * square(stdDev[0]))) * cubedInvDet;
	gradInvJk(0, 0, 1, 1) = 2 * (invJk(0, 0) * square(stdDev[1]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0)) - invJk(1, 1) * (square(invJk(0, 0)) * square(stdDev[1]) + square(invJk(1, 0)) * square(stdDev[0]))) * cubedInvDet;
	gradInvJk(0, 1, 0, 0) = 2 * invJk(0, 0) * invJk(0, 1) * invJk(1, 1) * square(stdDev[1]) * cubedInvDet + 2 * invJk(1, 0) * square(invJk(1, 1)) * square(stdDev[0]) * cubedInvDet;
	gradInvJk(0, 1, 0, 1) = -(invJk(0, 0) * square(stdDev[1]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0)) + 2 * invJk(1, 0) * (invJk(0, 0) * invJk(0, 1) * square(stdDev[1]) + invJk(1, 0) * invJk(1, 1) * square(stdDev[0]))) * cubedInvDet;
	gradInvJk(0, 1, 1, 0) = -(invJk(0, 0) * square(stdDev[1]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0)) + 2 * invJk(1, 0) * (invJk(0, 0) * invJk(0, 1) * square(stdDev[1]) + invJk(1, 0) * invJk(1, 1) * square(stdDev[0]))) * cubedInvDet;
	gradInvJk(0, 1, 1, 1) = 2 * invJk(1, 0) * (square(invJk(0, 0)) * square(stdDev[1]) + square(invJk(1, 0)) * square(stdDev[0])) * cubedInvDet;
	gradInvJk(1, 0, 0, 0) = 2 * invJk(0, 1) * (square(invJk(0, 1)) * square(stdDev[1]) + square(invJk(1, 1)) * square(stdDev[0])) * cubedInvDet;
	gradInvJk(1, 0, 0, 1) = -(2 * invJk(0, 1) * (invJk(0, 0) * invJk(0, 1) * square(stdDev[1]) + invJk(1, 0) * invJk(1, 1) * square(stdDev[0])) + invJk(1, 1) * square(stdDev[0]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0))) * cubedInvDet;
	gradInvJk(1, 0, 1, 0) = -(2 * invJk(0, 1) * (invJk(0, 0) * invJk(0, 1) * square(stdDev[1]) + invJk(1, 0) * invJk(1, 1) * square(stdDev[0])) + invJk(1, 1) * square(stdDev[0]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0))) * cubedInvDet;
	gradInvJk(1, 0, 1, 1) = 2 * square(invJk(0, 0)) * invJk(0, 1) * square(stdDev[1]) * cubedInvDet + 2 * invJk(0, 0) * invJk(1, 0) * invJk(1, 1) * square(stdDev[0]) * cubedInvDet;
	gradInvJk(1, 1, 0, 0) = 2 * (-invJk(0, 0) * (square(invJk(0, 1)) * square(stdDev[1]) + square(invJk(1, 1)) * square(stdDev[0])) + invJk(1, 1) * square(stdDev[0]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0))) * cubedInvDet;
	gradInvJk(1, 1, 0, 1) = (2 * invJk(0, 0) * (invJk(0, 0) * invJk(0, 1) * square(stdDev[1]) + invJk(1, 0) * invJk(1, 1) * square(stdDev[0])) - invJk(1, 0) * square(stdDev[0]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0))) * cubedInvDet;
	gradInvJk(1, 1, 1, 0) = (2 * invJk(0, 0) * (invJk(0, 0) * invJk(0, 1) * square(stdDev[1]) + invJk(1, 0) * invJk(1, 1) * square(stdDev[0])) - invJk(1, 0) * square(stdDev[0]) * (invJk(0, 0) * invJk(1, 1) - invJk(0, 1) * invJk(1, 0))) * cubedInvDet;
	gradInvJk(1, 1, 1, 1) = -2 * invJk(0, 0) * (square(invJk(0, 0)) * square(stdDev[1]) + square(invJk(1, 0)) * square(stdDev[0])) * cubedInvDet;

	real squaredInvDet = 1.0 / square(invJk.det());
	gradStdDev(0, 0, 0) = 2 * square(invJk(1, 1)) * stdDev[0] * squaredInvDet;
	gradStdDev(0, 0, 1) = -2 * invJk(1, 0) * invJk(1, 1) * stdDev[0] * squaredInvDet;
	gradStdDev(0, 1, 0) = -2 * invJk(1, 0) * invJk(1, 1) * stdDev[0] * squaredInvDet;
	gradStdDev(0, 1, 1) = 2 * square(invJk(1, 0)) * stdDev[0] * squaredInvDet;
	gradStdDev(1, 0, 0) = 2 * square(invJk(0, 1)) * stdDev[1] * squaredInvDet;
	gradStdDev(1, 0, 1) = -2 * invJk(0, 0) * invJk(0, 1) * stdDev[1] * squaredInvDet;
	gradStdDev(1, 1, 0) = -2 * invJk(0, 0) * invJk(0, 1) * stdDev[1] * squaredInvDet;
	gradStdDev(1, 1, 1) = 2 * square(invJk(0, 0)) * stdDev[1] * squaredInvDet;
}

template<typename real>
__host__ __device__
void gradGaussian(Vector2<real>& gradScreenpoint, Matrix22<real>& gradInvJk, Matrix22<real>& gradVk, const BaseVector2<real>& pixel, const Vector2<real>& screenpoint, const Matrix22<real>& invJk, const Matrix22<real>& Vk)
{
	// Recompute forward weight
	Vector2<real> x = pixel - screenpoint;
	Matrix22<real> invVk = Vk.inv();

	real exponent = (x * invVk) * x;
	real normalization = 2.0 * M_PI * sqrt<real>(Vk.det()); 
	real gaussian = exp<real>(exponent * -0.5) / normalization;
	real invJkDet = abs<real>(invJk.det());
	real weight = gaussian / invJkDet;

	// Derivative w.r.t to the screenpoint
	gradScreenpoint = (invVk * x) * weight;

	// Derivative w.r.t to the invJk
	Matrix22<real> Jk = invJk.inv(); //inverse(invJk, detInvJk);
	gradInvJk = Jk.transposed() * -(gaussian / invJkDet);

	// Derivative w.r.t to Vk
	real Vkf = 1.0 / (normalization * invJkDet);
	real Vkg = exp<real>(-0.5 * exponent);
	Matrix22<real> gradVkf = invVk.transposed() * (-1.0 / (4.0 * M_PI * sqrt<real>(Vk.det()) * invJkDet));
	Matrix22<real> gradVkg = outer(invVk.transposed() * x, invVk * x) * exp(exponent * -0.5) * 0.5;
	gradVk = gradVkf * Vkg + gradVkg * Vkf;
}

template<typename real>
__host__ __device__
void gradEWAWeight(
	Vector3<real>& gradPoint, 
	Vector3<real>& gradNormal, 
	Vector2<real>& gradStdDev, 
	const BaseVector2<real>& pixel, 
	real width, real height, real focallength, 
	const BaseVector3<real>& camPoint, 
	const BaseVector3<real>& camNormal, 
	const BaseVector2<real>& stdDev, 
	real smoothing)
{
	Vector2<real> screenpoint = toScreenspace(width, height, focallength, camPoint);

	// Computer required results from the forward pass
	Matrix22<real> invJk = buildInverseJacobian<real>(width, height, focallength, screenpoint, camPoint, camNormal);
	Matrix22<real> Vk = buildCovariance<real>(invJk, stdDev, smoothing);
	
	// Compuate factors of the chain rule
	Matrix32<real> dScreenpointdCamPoint = gradScreenspace<real>(width, height, focallength, camPoint);

	Tensor222<real> dInvJkdScreen;
	Tensor322<real> dInvJkdCamPoint, dInvJkdCamNormal;
	gradJacobian(dInvJkdScreen, dInvJkdCamPoint, dInvJkdCamNormal, width, height, focallength, screenpoint, camPoint, camNormal);

	Tensor2222<real> dVkdInvJk;
	Tensor222<real> dVkdStdDev;
	gradCovariance<real>(dVkdInvJk, dVkdStdDev, invJk, stdDev, smoothing);

	Vector2<real> dwdScreenpoint;
	Matrix22<real> dwdInvJk, dwdVk;
	gradGaussian<real>(dwdScreenpoint, dwdInvJk, dwdVk, pixel, screenpoint, invJk, Vk);

	// Common factors of the chain rule
	Matrix22<real> gradInvJk = (dwdVk * dVkdInvJk + dwdInvJk);
	Vector3<real> dwdCamNormal = gradInvJk * dInvJkdCamNormal;
	Vector3<real> dwdCamPoint = gradInvJk * (dInvJkdCamPoint + dInvJkdScreen * dScreenpointdCamPoint) + dwdScreenpoint * dScreenpointdCamPoint;
	
	// Derivative w.r.t to the inputs
	gradStdDev = dwdVk * dVkdStdDev;
	gradNormal = dwdCamNormal;
	gradPoint = dwdCamPoint;
}

template<typename real>
__host__ __device__
void gradEWAWeightSimplified(
	Vector3<real>& gradPoint,
	const BaseVector2<real>& pixel, 
	real width, real height, real focallength, 
	const BaseVector3<real>& camPoint, 
	const BaseVector3<real>& camNormal, 
	const BaseVector2<real>& stdDev, 
	real smoothing
)
{
	// Computer required results from the forward pass
	Vector2<real> screenpoint = toScreenspace(width, height, focallength, camPoint);
	Matrix22<real> invJk = buildInverseJacobian<real>(width, height, focallength, screenpoint, camPoint, camNormal);
	Matrix22<real> Vk = buildCovariance<real>(invJk, stdDev, smoothing);

	// Compuate factors of the chain rule
	Matrix32<real> dScreenpointdCamPoint = gradScreenspace<real>(width, height, focallength, camPoint);
	Vector2<real> dwdScreenpoint;
	Matrix22<real> dwdInvJk, dwdVk;
	gradGaussian<real>(dwdScreenpoint, dwdInvJk, dwdVk, pixel, screenpoint, invJk, Vk);

	// Derivative w.r.t to the point
	gradPoint = dwdScreenpoint * dScreenpointdCamPoint;
}