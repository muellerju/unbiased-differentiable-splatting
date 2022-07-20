#pragma once
#include <cuda_runtime.h>
#include <type_traits>


template<typename T, typename U>
struct is_same : std::false_type { };

template<typename T>
struct is_same<T, T> : std::true_type { };

template<typename T, typename U>
constexpr bool eqTypes() { return is_same<T, U>::value; }

template<typename real>
__host__ __device__
inline real sqrt(real x)
{
	if constexpr (eqTypes<real, float>())
	{
		return sqrtf(x);
	}
	else
	{
		return sqrt(x);
	}
}

template<typename real>
__host__ __device__
inline real exp(real x)
{
	if constexpr (eqTypes<real, float>())
	{
		return expf(x);
	}
	else
	{
		return exp(x);
	}
}

template<typename real>
__host__ __device__
inline real abs(real x)
{
	if constexpr (eqTypes<real, float>())
	{
		return fabsf(x);
	}
	else
	{
		return fabs(x);
	}
}

template<typename real>
__host__ __device__
inline real square(real a)
{
	return a * a;
}

template<typename real, unsigned int exponent>
__host__ __device__
inline real power(real x)
{
	real y = 1.0;
	for (unsigned int i = 0; i < exponent; i++)
		y *= x;
	return y;
}


template <typename real> 
__host__ __device__
int sign(real val) 
{
	return (real(0) < val) - (val < real(0));
}

