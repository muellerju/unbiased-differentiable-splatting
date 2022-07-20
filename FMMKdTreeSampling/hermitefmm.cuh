#pragma once

#include "cuda_runtime.h"

#include <cmath>

#ifndef M_PI
	#define M_PI 3.14159265359
#endif // !M_PI

#define HERMITEDEGREE 6
#define TAYLORDEGREE 6

template<typename real>
__host__ __device__ real fac(unsigned int n)
{
	real product = (real)1.0;
	for (unsigned int i = 1; i <= n; i++)
		product *= i;
	return product;
}

template<typename real>
__host__ __device__ real fac(unsigned int a, unsigned int b)
{
	return fac<real>(a) * fac<real>(b);
}

template<typename real>
__host__ __device__ real power(real x, real y, real a, real b)
{
	return pow(x, a) * pow(y, b);
}

template<typename real>
__host__ __device__ void hermiteFunction(real hermite[], real t, unsigned int ph)
{
	// Set recusion begin
	hermite[0] = exp(-(t * t));
	hermite[1] = (real)2.0 * t * exp(-(t * t));

	// Compute the hermite function for all remaining entries
	for (unsigned int n = 2; n <= ph; n++)
	{
		hermite[n] = ((real)2.0 * t * hermite[n - 1] - (real)2.0 * (n - 1) * hermite[n - 2]);
	}
}

/*
* Functions used to build a FMM expansion
*/

/*template<typename real>
__host__ __device__ void shiftHermiteExpansion(real shiftedCoeffs[], const real newXh[2], const real xh[2], const real coeffs[], unsigned int ph, real h)
{
	real shift[2];
	shift[0] = (xh[0] - newXh[0]) / std::sqrt(2 * h * h);
	shift[1] = (xh[1] - newXh[1]) / std::sqrt(2 * h * h);
	for (unsigned int x = 0; x <= ph; x++)
	{
		for (unsigned int y = 0; y <= ph; y++)
		{
			real coefficient = 0.0f;
			for (unsigned int a = 0; a <= x; a++)
			{
				for (unsigned int b = 0; b <= y; b++)
				{
					//Eigen::Vector2i alpha(a, b);
					unsigned int j = a * (ph + 1) + b;
					coefficient += coeffs[j] * power<real>(shift[0], shift[1], x - a, y - b) / fac<real>(x - a, y - b);
				}
			}
			unsigned int i = x * (ph + 1) + y;
#ifdef  __CUDA_ARCH__
			atomicAdd(&shiftedCoeffs[i], coefficient);
#else
			shiftedCoeffs[i] += coefficient;
#endif
		}
	}
}*/

template<typename real>
__host__ __device__
real shiftCoefficient(const real coeffs[], unsigned int ph, const real shift[2], unsigned x, unsigned int y)
{
	real coefficient = real(0);
	for (unsigned int a = 0; a < min(x, ph); a++)
	{
		for (unsigned int b = 0; a < min(x, ph); b++)
		{
			real coeff = coeffs[a * (ph + 1) + b];
			coefficient += coeff * power<real>(shift[0], shift[1], x - a, y - b) / fac<real>(x - a, y - b);
		}
	}
	return coefficient;
}


template<typename real>
__host__ __device__
void shiftHermiteExpansion(
	real shiftedCoeffs[], const real newXh[2], unsigned int newPh, 
	const real coeffs[], const real xh [], unsigned int ph, real h
)
{
	real shift[2];
	shift[0] = (xh[0] - newXh[0]) / std::sqrt(2 * h * h);
	shift[1] = (xh[1] - newXh[1]) / std::sqrt(2 * h * h);
	for (unsigned int x = 0; x <= newPh; x++)
	{
		for (unsigned int y = 0; y <= newPh; y++)
		{
			shiftedCoeffs[x * (newPh + 1) + y] += shiftCoefficient<real>(coeffs, ph, shift, x, y);
		}
	}
}

template<typename real>
__host__ __device__ void shiftLeafHermiteExpansion(real shiftedCoeffs[], const real newXh[2], const real xh[2], real coeff, unsigned int ph, real h)
{
	real shift[2];
	shift[0] = (xh[0] - newXh[0]) / std::sqrt(2 * h * h);
	shift[1] = (xh[1] - newXh[1]) / std::sqrt(2 * h * h);
	for (unsigned int x = 0; x <= ph; x++)
	{
		for (unsigned int y = 0; y <= ph; y++)
		{
			real coefficient = coeff * power<real>(shift[0], shift[1], x, y) / fac<real>(x, y);
			unsigned int i = x * (ph + 1) + y;
#ifdef  __CUDA_ARCH__
			atomicAdd(&shiftedCoeffs[i], coefficient);
#else
			shiftedCoeffs[i] += coefficient;
#endif
		}
	}
}

template<typename real>
__host__ __device__ real taylorCoefficient(unsigned int a, unsigned int b, const real hermiteX[], const real hermiteY[], const real coeffs[], unsigned int ph)
{
	real coefficient = 0.0;
	for (unsigned int k = 0; k <= ph; k++)
	{
		for (unsigned int l = 0; l <= ph; l++)
		{
			//real hn = hermite(a + k, b + l, x, y);
			real hn = hermiteX[a + k] * hermiteY[b + l];
			unsigned int index = k * (ph + 1) + l;
			coefficient += coeffs[index] * hn;
		}
	}
	return coefficient;
}

template<typename real, unsigned int maxdegree>
__host__ __device__ void convertToTaylor(
	real taylorCoeffs[], unsigned int pt, const real xt[2], 
	const real coeffs[], const real xh[2], unsigned int ph, 
	real h)
{

	// Pre-compute hermite functions
	real distX = (xh[0] - xt[0]) / std::sqrt(2 * h * h);
	real distY = (xh[1] - xt[1]) / std::sqrt(2 * h * h);
	real hermiteX[HERMITEDEGREE + TAYLORDEGREE + 2], hermiteY[HERMITEDEGREE + TAYLORDEGREE + 2];
	hermiteFunction(hermiteX, distX, HERMITEDEGREE + TAYLORDEGREE);
	hermiteFunction(hermiteY, distY, HERMITEDEGREE + TAYLORDEGREE);

	for (unsigned int a = 0; a <= pt; a++)
	{
		for (unsigned int b = 0; b <= pt; b++)
		{
			real coefficient = taylorCoefficient(a, b, hermiteX, hermiteY, coeffs, ph);
			real sign = (((a + b) % 2) == 0) ? 1.0f : -1.0f;
			unsigned int index = a * (pt + 1) + b;
			taylorCoeffs[index] += (sign * coefficient) / fac<real>(a, b);
		}
	}
}


template<typename real>
__host__ __device__ real hermiteError(real r, real nr, real ph)
{
	if (r >= 1.0f)
		return r;

	real a = nr / pow(1 - r, 2.0f);
	real b = pow(r, ph) / sqrt(fac<real>(ph));
	real c = b * b;
	real d = 2.0f * (1.0f - pow(r, ph)) * b;
	return a * (c + d);
}

template<typename real>
__host__ __device__
real taylorError(real r, unsigned int nr, unsigned int pt)
{
	real a = nr / power<real, 4>(1 - r);
	real b = pow(2 * r, pt);
	real c = (b * (2 - b)) / sqrt(fac<real>(pt));
	real d = square(1 - a);
	return a * (square(c) + 2 * d * c);
}

/*
* Functions used to evalute a FMM expansion
*/


template<typename real, unsigned int maxdegree>
__host__ __device__ real evluateHermite(const real xp[2], const real xh[2], const real coeffs[], unsigned int ph, real h)
{
	// Pre-compute hermite functions
	real distX = (xp[0] - xh[0]) / sqrt(2 * h * h);
	real distY = (xp[1] - xh[1]) / sqrt(2 * h * h);
	//real hermiteX[HERMITEDEGREE + 1], hermiteY[HERMITEDEGREE + 1];
	//hermiteFunction(hermiteX, distX, HERMITEDEGREE);
	//hermiteFunction(hermiteY, distY, HERMITEDEGREE);
	real hermiteX[maxdegree+1], hermiteY[maxdegree+1];
	hermiteFunction(hermiteX, distX, maxdegree);
	hermiteFunction(hermiteY, distY, maxdegree);

	// Evaluate hermite expansion
	real weight = real(0);
	for (unsigned int a = 0; a <= ph; a++)
	{
		for (unsigned int b = 0; b <= ph; b++)
		{
			unsigned int index = a * (ph + 1) + b;
			weight += coeffs[index] * hermiteX[a] * hermiteY[b];//devHermite(a, b, distX, distY);
		}
	}
	return weight;
}


template<typename real>
__host__ __device__ real evaluateTaylor(const real xp[2], const real xt[2], const real coeffs[], unsigned int pt, real h)
{
	real distX = (xp[0] - xt[0]) / sqrt(2 * h * h);
	real distY = (xp[1] - xt[1]) / sqrt(2 * h * h);
	real weight = 0.0f;
	for (unsigned int a = 0; a <= pt; a++)
	{
		real exponentA = static_cast<real>(a);
		for (unsigned int b = 0; b <= pt; b++)
		{
			unsigned int index = a * (pt + 1) + b;
			real exponentB = static_cast<real>(b);
			weight += coeffs[index] * power(distX, distY, exponentA, exponentB);
		}
	}
	return weight;
}