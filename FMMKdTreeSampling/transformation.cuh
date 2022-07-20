#pragma once

#include <cuda_runtime.h>

#include "transformation.h"
#include "linalg.cuh"

template<typename real, transform_t transform>
__host__ __device__
Vector3<real> toCameraspace(const BaseMatrix44<real>& transformation, const BaseVector3<real>& point)
{
	Vector3<real> camPoint;
	for (unsigned int i = 0; i < 3; i++)
	{
		camPoint[i] = 0.0;
		for (unsigned int j = 0; j < 3; j++)
		{
			camPoint[i] += transformation(i, j) * point[j];
		}
	}
	if (transform == affine) // TODO: Add constexpr and switch to c++17 
	{
		for (unsigned int i = 0; i < 3; i++)
			camPoint[i] += transformation(i, 3);
	}
	return camPoint;
}

template<typename real>
__host__ __device__
Vector2<real> toScreenspace(real width, real height, real focallength, const BaseVector3<real>& camPoint)
{
	Vector2<real> screenpoint;
	screenpoint[0] = -(camPoint[0] * focallength) / camPoint[2] + width / 2.0;
	screenpoint[1] = (camPoint[1] * focallength) / camPoint[2] + height / 2.0;
	return screenpoint;
}

template<typename real>
__host__ __device__
Vector3<real> backproject(real width, real height, real focallength, const BaseVector2<real>& screenPoint)
{
	Vector3<real> camPoint;
	camPoint[0] = (width - 2.0 * screenPoint[0]) / (2.0 * focallength);
	camPoint[1] = -(height - 2.0 * screenPoint[1]) / (2.0 * focallength);
	camPoint[2] = 1.0;
	return camPoint;
}

template<typename real>
__host__ __device__
Vector3<real> projectOntoPlane(const Vector3<real>& x, const BaseVector3<real>& point, const BaseVector3<real>& normal)
{
	real normalAngle = point.dot(normal);
	real xts = normalAngle / x.dot(normal);
	Vector3<real> tilde = x * xts;
	return tilde;
}

template<typename real>
__host__ __device__
real projectionDistance(const Vector3<real>& x, const BaseVector3<real>& point, const BaseVector3<real>& normal)
{
	real normalAngle = point.dot(normal);
	real xts = normalAngle / x.dot(normal);
	return xts;
}
