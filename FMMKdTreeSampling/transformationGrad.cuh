#pragma once

#include "transformation.cuh"

template<typename real, transform_t transform>
__host__ __device__
void gradCameraspace(Tensor443<real>& gradTransformation, Matrix33<real>& gradPoint, const BaseMatrix44<real>& transformation, const BaseVector3<real>& point)
{
	// Grad transformation
	for (unsigned int i = 0; i < 48; i++)
		gradTransformation[i] = 0.0;

	for (unsigned int i = 0; i < 3; i++)
	{
		for (unsigned int j = 0; j < 3; j++)
		{
			for (unsigned int k = 0; k < 3; k++)
				gradTransformation(i, k, j) = (i == j) ? point[k] : 0.0;
		}
	}

	if constexpr (transform == affine)
	{
		for (unsigned int i = 0; i < 4; i++)
		{
			for (unsigned int k = 0; k < 3; k++)
				gradTransformation(i, 3, k) = (i == k) ? 1.0 : 0.0;
		}
	}


	// Grad point
	for (unsigned int i = 0; i < 3; i++)
	{
		for (unsigned int j = 0; j < 3; j++)
		{
			gradPoint(i, j) = transformation(i, j);
		}
	}
}

template<typename real>
__host__ __device__
Matrix32<real> gradScreenspace(real width, real height, real focallength, const BaseVector3<real>& camPoint)
{
	Matrix32<real> gradCamPoint;
	gradCamPoint(0, 0) = -focallength / camPoint[2];
	gradCamPoint(0, 1) = 0;
	gradCamPoint(1, 0) = 0;
	gradCamPoint(1, 1) = focallength / camPoint[2];
	gradCamPoint(2, 0) = camPoint[0] * focallength / square(camPoint[2]);
	gradCamPoint(2, 1) = -camPoint[1] * focallength / square(camPoint[2]);
	return gradCamPoint;
}


template<typename real>
__host__ __device__
Matrix23<real> gradBackproject(real width, real height, real focallength, const Vector2<real>& screenPoint)
{
	Matrix23<real> gradScreenpoint;
	gradScreenpoint(0, 0) = -1.0 / focallength;;
	gradScreenpoint(0, 1) = 0;
	gradScreenpoint(0, 2) = 0;
	gradScreenpoint(1, 0) = 0;
	gradScreenpoint(1, 1) = 1.0 / focallength;
	gradScreenpoint(1, 2) = 0;
	return gradScreenpoint;
}

template<typename real>
__host__ __device__
void gradProjectOntoPlane(
	Matrix33<real>& gradX, 
	Matrix33<real>& gradPoint, 
	Matrix33<real>& gradNormal, 
	const Vector3<real>& x, 
	const BaseVector3<real>& point, 
	const BaseVector3<real>& normal)
{
	real normalAngle = point.dot(normal);
	real invDot = 1.0 / x.dot(normal);
	real xts = normalAngle * invDot;

	gradX = DiagMatrix33<real>(xts) - outer(x, normal) * normalAngle * square(invDot);
	gradPoint = outer(x, normal) * invDot;
	gradNormal = outer(x, point) * invDot - outer(x, x) * normalAngle * square(invDot);

}