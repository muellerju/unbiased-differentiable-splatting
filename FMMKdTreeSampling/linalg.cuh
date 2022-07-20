#pragma once
#include <cuda_runtime.h>

#include "mathHelper.cuh"

template<typename real>
class BaseMatrix22
{
public:
	__host__ __device__ virtual real operator[](int i) const = 0;

	__host__ __device__ virtual real operator()(int i, int j) const = 0;
};

template<typename real>
class ConstMatrix22 : public BaseMatrix22<real>
{
private:
	const real* data;
public:
	__host__ __device__ inline ConstMatrix22(const real* _data) : data(_data) {}

	__host__ __device__ inline const real* data_ptr() const { return data; }

	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real operator()(int i, int j) const { return data[j + 2 * i]; }

};

template<typename real>
class Matrix22 : public BaseMatrix22<real>
{
protected:
	real data[4];
public:
	__host__ __device__ inline Matrix22() {}

	__host__ __device__ inline real* data_ptr() { return data; }
	
	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j) { return data[j + 2 * i]; }
	__host__ __device__ inline real operator()(int i, int j) const { return data[j + 2 * i]; }

	__host__ __device__ inline Matrix22<real> operator*(real scalar) const
	{
		Matrix22<real> M;
		M[0] = scalar * data[0];
		M[1] = scalar * data[1];
		M[2] = scalar * data[2];
		M[3] = scalar * data[3];
		return M;
	};

	__host__ __device__ inline Matrix22<real> operator/(real scalar) const
	{
		Matrix22<real> M;
		M[0] = data[0] / scalar;
		M[1] = data[1] / scalar;
		M[2] = data[2] / scalar;
		M[3] = data[3] / scalar;
		return M;
	};

	__host__ __device__ inline real det() const
	{
		return data[0] * data[3] - data[1] * data[2];
	}

	__host__ __device__ inline real trace() const
	{
		return data[0] + data[3];
	}

	__host__ __device__ inline Matrix22<real> inv() const
	{
		real invDet = 1.0 / this->det();
		Matrix22 invM;
		invM[0] = data[3] * invDet;
		invM[1] = -data[1] * invDet;
		invM[2] = -data[2] * invDet;
		invM[3] = data[0] * invDet;
		return invM;
	}

	__host__ __device__ inline Matrix22<real> transposed() const
	{
		Matrix22 M;
		M[0] = data[0];
		M[1] = data[2];
		M[2] = data[1];
		M[3] = data[3];
		return M;
	}

	__host__ __device__ inline Matrix22<real> squared() const
	{
		Matrix22 M;
		for (unsigned int i = 0; i < 4; i++)
			M[i] = data[i] * data[i];
		return M;
	}

	__host__ __device__ inline Matrix22<real> operator*(const Matrix22<real>& B) const
	{
		Matrix22<real> M;
		for (unsigned int i = 0; i < 2; i++)
		{
			for (unsigned int k = 0; k < 2; k++)
			{
				M(i, k) = real(0);
				for (unsigned int j = 0; j < 2; j++)
				{
					M(i, k) += this->operator()(i, j) * B(j, k);
				}
			}
		}
		return M;
	}

	__host__ __device__ inline friend Matrix22<real> operator+(const Matrix22<real>& a, const Matrix22<real>& b)
	{
		Matrix22 M;
		for (unsigned int i = 0; i < 4; i++)
			M[i] = a[i] + b[i];
		return M;
	}

	__host__ __device__ inline friend Matrix22<real> operator-(const Matrix22<real>& a, const Matrix22<real>& b)
	{
		Matrix22 M;
		for (unsigned int i = 0; i < 4; i++)
			M[i] = a[i] - b[i];
		return M;
	}

};

template<typename real>
class Matrix23
{
private:
	real data[6];
public:
	__host__ __device__ inline Matrix23() {}

	__host__ __device__ inline real* data_ptr() { return data; }

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j) { return data[j + 3 * i]; }
	__host__ __device__ inline real operator()(int i, int j) const { return data[j + 3 * i]; }

	__host__ __device__ inline Matrix23<real> operator*(real scalar) const
	{
		Matrix23<real> M;
		for (unsigned int i = 0; i < 6; i++)
			M[i] = data[i] * scalar;
		return M;
	};

	__host__ __device__ inline Matrix23<real> operator/(real scalar) const
	{
		Matrix23<real> M;
		for (unsigned int i = 0; i < 6; i++)
			M[i] = data[i] / scalar;
		return M;
	};
};

template<typename real>
class Matrix32
{
private:
	real data[6];
public:
	__host__ __device__ inline Matrix32() {}

	__host__ __device__ inline real* data_ptr() { return data; }

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j) { return data[j + 2 * i]; }
	__host__ __device__ inline real operator()(int i, int j) const { return data[j + 2 * i]; }

	__host__ __device__ inline Matrix32<real> operator*(real scalar) const
	{
		Matrix32<real> M;
		for (unsigned int i = 0; i < 6; i++)
			M[i] = data[i] * scalar;
		return M;
	};

	__host__ __device__ inline Matrix32<real> operator/(real scalar) const
	{
		Matrix32<real> M;
		for (unsigned int i = 0; i < 6; i++)
			M[i] = data[i] / scalar;
		return M;
	};
};

template<typename real>
class Matrix33
{
protected:
	real data[9];
public:
	__host__ __device__ inline Matrix33() {}

	__host__ __device__ inline real* data_ptr() { return data; }

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j) { return data[j + 3 * i]; }
	__host__ __device__ inline real operator()(int i, int j) const { return data[j + 3 * i]; }

	__host__ __device__ inline Matrix33<real> operator*(real scalar) const
	{
		Matrix33<real> M;
		for (unsigned int i = 0; i < 9; i++)
			M[i] = data[i] * scalar;
		return M;
	};

	__host__ __device__ inline Matrix33<real> operator/(real scalar) const
	{
		Matrix33<real> M;
		for (unsigned int i = 0; i < 9; i++)
			M[i] = data[i] / scalar;
		return M;
	};

	__host__ __device__ inline void operator=(const Matrix33<real>& M)
	{
		for (unsigned int i = 0; i < 9; i++)
			data[i] = M[i];
	}

	__host__ __device__ inline Matrix33 operator-(const Matrix33<real>& M)
	{
		Matrix33 A;
		for (unsigned int i = 0; i < 9; i++)
			A[i] = data[i] - M[i];
		return A;
	}

	__host__ __device__ inline Matrix33 transposed() const
	{
		Matrix33<real> M;
		for (unsigned int i = 0; i < 3; i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				M(i, j) = this->operator()(j, i);
			}
		}
		return M;
	}

	__host__ __device__ static Matrix33 zeros()
	{
		Matrix33<real> M;
		for(unsigned int i = 0; i < 3; i++)
		{
			for(unsigned int j = 0; j < 3; j++)
			{
				M(i,j) = real(0);
			}
		}
		return M;
	}

};

template<typename real>
class DiagMatrix33 : public Matrix33<real>
{
public:
	__host__ __device__ inline DiagMatrix33<real>(real value)
	{
		for (unsigned int i = 0; i < 3; i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				this->operator()(i, j) = (i == j) ? value : real(0);
				
			}				 
		}
	}

	__host__ __device__ inline DiagMatrix33<real>(const real* vec)
	{
		for (unsigned int i = 0; i < 3; i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				this->operator()(i, j) = (i == j) ? vec[i] : real(0);
			}				 
		}
	}
};

template<typename real>
class BaseMatrix44
{
public:
	__host__ __device__ virtual real operator[](int i) const = 0;

	__host__ __device__ virtual real operator()(int i, int j) const = 0;
};

template<typename real>
class ConstMatrix44 : public BaseMatrix44<real>
{
private:
	const real* data;
public:
	__host__ __device__ ConstMatrix44<real>(const real* _data) : data(_data) {}

	__host__ __device__ real operator[](int i) const { return data[i]; }

	__host__ __device__ real operator()(int i, int j) const { return data[j + 4 * i]; }
};

template<typename real>
class Matrix44 : public BaseMatrix44<real>
{
private:
	real data[4*4];
public:
	__host__ __device__ inline Matrix44() {}

	__host__ __device__ inline real* data_ptr() { return data; }

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j) { return data[j + 4 * i]; }
	__host__ __device__ inline real operator()(int i, int j) const { return data[j + 4 * i]; }

	__host__ __device__ inline Matrix44<real> operator*(real scalar) const
	{
		Matrix44<real> M;
		for (unsigned int i = 0; i < 4 * 4; i++)
			M[i] = data[i] * scalar;
		return M;
	};

	__host__ __device__ inline Matrix44<real> operator/(real scalar) const
	{
		Matrix44<real> M;
		for (unsigned int i = 0; i < 4 * 4; i++)
			M[i] = data[i] / scalar;
		return M;
	};

	__host__ __device__ inline friend Matrix44<real> operator+(const Matrix44<real>& a, const Matrix44<real>& b)
	{
		Matrix44 M;
		for (unsigned int i = 0; i < 16; i++)
			M[i] = a[i] + b[i];
		return M;
	}

	__host__ __device__ inline friend Matrix44<real> operator-(const Matrix44<real>& a, const Matrix44<real>& b)
	{
		Matrix44 M;
		for (unsigned int i = 0; i < 16; i++)
			M[i] = a[i] - b[i];
		return M;
	}

	__host__ __device__ inline real cwiseDot(const Matrix44<real>& M)
	{
		real c = real(0);
		for (unsigned int i = 0; i < 16; i++)
			c += data[i] * M[i];
		return c;
	}

	__host__ __device__ inline Matrix44 transposed()
	{
		Matrix44 M;
		for (unsigned int i = 0; i < 4; i++)
		{
			for (unsigned int j = 0; j < 4; j++)
				M(j, i) = this->operator()(i, j);
		}
		return M;
	}
};

template<typename real>
class BaseVector2
{
public:
	__host__ __device__ virtual real operator[](int i) const = 0;

	__host__ __device__ virtual real operator()(int i) const = 0;

};

template<typename real>
class ConstVector2 : public BaseVector2<real>
{
private:
	const real* data;
public:
	__host__ __device__ inline ConstVector2(const real* _data) : data(_data) {}

	__host__ __device__ inline const real* data_ptr() const { return data; }

	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real operator()(int i) const { return data[i]; }
};

template<typename real>
class Vector2 : public BaseVector2<real>
{
private:
	real data[2];
public:
	__host__ __device__ inline Vector2() {}
	
	__host__ __device__ inline real* data_ptr() { return data; }

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i) { return data[i]; }
	__host__ __device__ inline real operator()(int i) const { return data[i]; }

	__host__ __device__ inline Vector2<real> operator/(real scalar) const
	{
		Vector2<real> v;
		v[0] = data[0] / scalar;
		v[1] = data[1] / scalar;
		return v;
	}

	__host__ __device__ inline Vector2<real> operator*(real scalar) const
	{
		Vector2<real> v;
		v[0] = data[0] * scalar;
		v[1] = data[1] * scalar;
		return v;
	}

	__host__ __device__ inline Vector2<real> operator*(Matrix22<real> M) const
	{
		Vector2<real> y;
		y[0] = data[0] * M(0, 0) + data[1] * M(1, 0);
		y[1] = data[0] * M(0, 1) + data[1] * M(1, 1);
		return y;
	}

	__host__ __device__ inline real operator*(Vector2<real> v) const
	{
		return data[0] * v[0] + data[1] * v[1];
	}

	__host__ __device__ inline void operator=(const Vector2<real>& a)
	{
		data[0] = a[0];
		data[1] = a[1];
	}

	__host__ __device__ inline friend Vector2<real> operator-(const BaseVector2<real>& a, const BaseVector2<real>& b)
	{
		Vector2<real> x;
		for (unsigned int i = 0; i < 2; i++)
			x[i] = a[i] - b[i];
		return x;
	}

	__host__ __device__ inline friend Vector2<real> operator+(const Vector2<real>& a, const Vector2<real>& b)
	{
		Vector2<real> x;
		for (unsigned int i = 0; i < 2; i++)
			x[i] = a[i] + b[i];
		return x;
	}

	__host__ __device__ inline Vector2<real> copy() const
	{
		Vector2 v;
		v[0] = data[0];
		v[1] = data[1];
		return v;
	}

	__host__ __device__ inline real dot(const Vector2<real>& v) const
	{
		return data[0] * v[0] + data[1] * v[1];
	}
};

template<typename real>
class BaseVector3
{
public:
	__host__ __device__ virtual real operator[](int i) const = 0;

	__host__ __device__ virtual real operator()(int i) const = 0;

	__host__ __device__ virtual real* data_ptr() const = 0;

	__host__ __device__ virtual real dot(const BaseVector3<real>& b) const = 0;

	__host__ __device__ virtual real norm() const = 0;
};

template<typename real>
class ConstVector3 :public BaseVector3<real>
{
private:
	const real* data;
public:
	__host__ __device__ ConstVector3<real>(const real* _data) : data(_data) {}

	__host__ __device__ real operator[](int i) const { return data[i]; }

	__host__ __device__ real operator()(int i) const { return data[i]; }

	__host__ __device__ inline real* data_ptr() const { return (real*)data; } 

	__host__ __device__ inline real dot(const BaseVector3<real>& b) const
	{
		real x = real(0);
		for (unsigned int i = 0; i < 3; i++)
			x += data[i] * b[i];
		return x;
	}

	__host__ __device__ inline real norm() const

	{
		real x = real(0);
		for (unsigned int i = 0; i < 3; i++)
			x += data[i] * data[i];
		return sqrt<real>(x);
	}
};

template<typename real>
class PointerVector3 : public BaseVector3<real>
{
private:
	real* data;
public:
	__host__ __device__ PointerVector3<real>(real* _data) : data(_data) {}

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i) { return data[i]; }
	__host__ __device__ inline real operator()(int i) const { return data[i]; }

	__host__ __device__ inline real* data_ptr() const { return (real*)data; }

	__host__ __device__ inline void operator=(const BaseVector3<real>& a)
	{
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
	}

	__host__ __device__ inline void operator+=(const BaseVector3<real>& a)
	{
		data[0] += a[0];
		data[1] += a[1];
		data[2] += a[2];
	}

	__host__ __device__ inline real dot(const BaseVector3<real>& b) const
	{
		real x = real(0);
		for (unsigned int i = 0; i < 3; i++)
			x += data[i] * b[i];
		return x;
	}

	__host__ __device__ inline real norm() const

	{
		real x = real(0);
		for (unsigned int i = 0; i < 3; i++)
			x += data[i] * data[i];
		return sqrt<real>(x);
	}
};

template<typename real>
class Vector3 : public BaseVector3<real>
{
private:
	real data[3];
public:
	__host__ __device__ inline Vector3() {}

	__host__ __device__ inline real* data_ptr() const { return (real*)data; }
	__host__ __device__ inline real* data_ptr() { return data; }
	
	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i) { return data[i]; }
	__host__ __device__ inline real operator()(int i) const { return data[i]; }

	__host__ __device__ inline Vector3<real> operator/(real scalar) const
	{
		Vector3<real> M;
		for (unsigned int i = 0; i < 3; i++)
			M[i] = data[i] / scalar;
		return M;
	};

	__host__ __device__ inline Vector3<real> operator*(real scalar) const
	{
		Vector3<real> M;
		for (unsigned int i = 0; i < 3; i++)
			M[i] = data[i] * scalar;
		return M;
	};

	__host__ __device__ inline friend Vector3<real> operator-(const BaseVector3<real>& a, const BaseVector3<real>& b)
	{
		Vector3<real> x;
		for (unsigned int i = 0; i < 3; i++)
			x[i] = a[i] - b[i];
		return x;
	}

	__host__ __device__ inline friend Vector3<real> operator+(const Vector3<real>& a, const Vector3<real>& b)
	{
		Vector3<real> x;
		for (unsigned int i = 0; i < 3; i++)
			x[i] = a[i] + b[i];
		return x;
	}

	__host__ __device__ inline void operator=(const Vector3<real>& a)
	{
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
	}

	__host__ __device__ inline void operator-=(const Vector3<real>& a)
	{
		data[0] -= a[0];
		data[1] -= a[1];
		data[2] -= a[2];
	}

	__host__ __device__ inline void operator+=(const Vector3<real>& a)
	{
		data[0] += a[0];
		data[1] += a[1];
		data[2] += a[2];
	}

	__host__ __device__ inline real norm() const

	{
		real x = real(0);
		for (unsigned int i = 0; i < 3; i++)
			x += data[i] * data[i];
		return sqrt<real>(x);
	}

	__host__ __device__ inline Vector3 normalized() const
	{
		Vector3 n;
		real invNorm = 1.0 / this->norm();
		for (unsigned int i = 0; i < 3; i++)
			n[i] = data[i]*invNorm;
		return n;
	}

	__host__ __device__ inline real dot(const BaseVector3<real>& b) const
	{
		real x = real(0);
		for (unsigned int i = 0; i < 3; i++)
			x += data[i] * b[i];
		return x;
	}

	__host__ __device__ inline Vector3 cross(const BaseVector3<real>& b) const
	{
		Vector3<real> x;
		x[0] = data[1] * b[2] - data[2] * b[1];
		x[1] = data[2] * b[0] - data[0] * b[2];
		x[2] = data[0] * b[1] - data[1] * b[0];
		return x;
	}
};

template<typename real>
class Vector4
{
private:
	real data[4];
public:
	__host__ __device__ inline Vector4() {}
	__host__ __device__ inline Vector4(const BaseVector3<real>& vec, real w) 
	{
		data[0] = vec[0];
		data[1] = vec[1];
		data[2] = vec[2];
		data[3] = w;
	}

	__host__ __device__ inline Vector4(real x, real y, real z, real w)
	{
		data[0] = x;
		data[1] = y;
		data[2] = z;
		data[3] = w;
	}

	__host__ __device__ inline real* data_ptr() { return data; }

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i) { return data[i]; }
	__host__ __device__ inline real operator()(int i) const { return data[i]; }

	__host__ __device__ inline Vector3<real> xyz()
	{
		Vector3<real> v;
		for (unsigned int i = 0; i < 3; i++)
			v[i] = data[i];
		return v;
	}

	__host__ __device__ inline real dot(const Vector4<real>& x)
	{
		real y = real(0);
		for (unsigned int i = 0; i < 4; i++)
			y += data[i] * x[i];
		return y;
	} 

	__host__ __device__ inline Vector4<real> operator/(real scalar) const
	{
		Vector4<real> M;
		for (unsigned int i = 0; i < 4; i++)
			M[i] = data[i] / scalar;
		return M;
	}

	__host__ __device__ inline Vector4<real> operator+(const Vector4<real>& v) const
	{
		Vector4<real> y;
		for (unsigned int i = 0; i < 4; i++)
			y[i] = data[i] + v[i];
		return y;
	}

	__host__ __device__ inline Vector4<real> operator*(real scalar) const
	{
		Vector4<real> v;
		for (unsigned int i = 0; i < 4; i++)
			v[i] = data[i] * scalar;
		return v;
	};

	__host__ __device__ inline friend Vector4<real> operator-(const Vector4<real>& a, const Vector4<real>& b)
	{
		Vector4<real> x;
		for (unsigned int i = 0; i < 4; i++)
			x[i] = a[i] - b[i];
		return x;
	}
};

template<typename real>
class DiagMatrix22 : public Matrix22<real>
{
public:
	__host__ __device__ inline DiagMatrix22(const BaseVector2<real>& values)
	{
		this->data[0] = values[0];
		this->data[1] = real(0);
		this->data[2] = real(0);
		this->data[3] = values[1];
	}
	__host__ __device__ inline DiagMatrix22(real value)
	{
		this->data[0] = value;
		this->data[1] = real(0);
		this->data[2] = real(0);
		this->data[3] = value;
	}
};

template<typename real>
class Tensor222
{
private:
	real data[8];
public:
	__host__ __device__ inline Tensor222() {}

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j, int k) { return data[k + 2 * (j + 2 * i)]; }
	__host__ __device__ inline real operator()(int i, int j, int k) const { return data[k + 2 * (j + 2 * i)]; }

	__host__ __device__ inline Tensor222<real> operator+(const Tensor222<real>& T)
	{
		Tensor222<real> A;
		for (unsigned int i = 0; i < 8; i++)
			A[i] = data[i] + T[i];
		return A;
	}

	__host__ __device__ inline Tensor222<real> transposed() const
	{
		Tensor222<real> T;
		for (unsigned int i = 0; i < 2; i++)
		{
			for (unsigned int j = 0; j < 2; j++)
			{
				for (unsigned int k = 0; k < 2; k++)
					T(k,i,j) = this->operator()(i,j,k);
			}
		}
		return T;
	}
};

template<typename real>
class Tensor322
{
private:
	real data[12];
public:
	__host__ __device__ inline Tensor322() {}

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j, int k) { return data[k + 2*(j + 2*i)]; }
	__host__ __device__ inline real operator()(int i, int j, int k) const { return data[k + 2 * (j + 2 * i)]; }

	__host__ __device__
		inline Tensor322<real> operator+(const Tensor322<real>& T)
	{
		Tensor322<real> A;
		for (unsigned int i = 0; i < 12; i++)
			A[i] = data[i] + T[i];
		return A;
	}
};

template<typename real>
class Tensor443
{
private:
	real data[48];
public:
	__host__ __device__ inline Tensor443() {}

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j, int k) { return data[k + 3 * (j + 4 * i)]; }
	__host__ __device__ inline real operator()(int i, int j, int k) const { return data[k + 3 * (j + 4 * i)]; }
};

template<typename real>
class Tensor444
{
private:
	real data[64];
public:
	__host__ __device__ inline Tensor444() {}

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j, int k) { return data[k + 4 * (j + 4 * i)]; }
	__host__ __device__ inline real operator()(int i, int j, int k) const { return data[k + 4 * (j + 4 * i)]; }
};

template<typename real>
class Tensor2222
{
private:
	real data[16];
public:
	__host__ __device__ inline Tensor2222() {}

	__host__ __device__ inline real& operator[](int i) { return data[i]; }
	__host__ __device__ inline real operator[](int i) const { return data[i]; }

	__host__ __device__ inline real& operator()(int i, int j, int k, int l) { return data[l + 2 * (k + 2 * (j + 2 * i))]; }
	__host__ __device__ inline real operator()(int i, int j, int k, int l) const { return data[l + 2 * (k + 2 * (j + 2 * i))]; }
};


template<typename real>
__host__ __device__
inline Vector2<real> matvec(const BaseMatrix22<real>& M, const BaseVector2<real>& x)
{
	Vector2<real> y;
	y[0] = M[0] * x[0] + M[1] * x[1];
	y[1] = M[2] * x[0] + M[3] * x[1];
	return y;
}

template<typename real>
__host__ __device__
inline Vector3<real> matvec(const Matrix33<real>& M, const Vector3<real>& x)
{
	Vector3<real> y;
	for (unsigned int i = 0; i < 3; i++)
	{
		y[i] = real(0);
		for (unsigned int j = 0; j < 3; j++)
		{
			y[i] += M(i, j)*x[j];
		}
	}
	return y;
}

template<typename real>
__host__ __device__
inline Vector2<real> matvec(const Matrix23<real>& M, const Vector3<real>& x)
{
	Vector2<real> y;
	for (unsigned int i = 0; i < 2; i++)
	{
		y[i] = real(0);
		for (unsigned int j = 0; j < 3; j++)
		{
			y[i] += M(i, j) * x[j];
		}
	}
	return y;
}

template<typename real>
__host__ __device__
inline Vector2<real> matvec(const BaseVector2<real>& x, const BaseMatrix22<real>& M)
{
	Vector2<real> y;
	y[0] = M[0] * x[0] + M[2] * x[1];
	y[1] = M[1] * x[0] + M[3] * x[1];
	return y;
}

template<typename real>
__host__ __device__
inline Vector2<real> matvec(const Matrix32<real>& M, const Vector3<real>& v)
{
	Vector2<real> y;
	for (unsigned int j = 0; j < 2; j++)
	{
		y[j] = real(0);
		for (unsigned int i = 0; i < 3; i++)
		{
			y[j] += M(i, j) * v[i];
		}
	}
	return y;
}

template<typename real>
__host__ __device__
inline Matrix22<real> outer(const BaseVector2<real>& x, const BaseVector2<real>& y)
{
	Matrix22<real> M;
	M[0] = x[0] * y[0];
	M[1] = x[0] * y[1];
	M[2] = x[1] * y[0];
	M[3] = x[1] * y[1];
	return M;
}

template<typename real>
__host__ __device__
inline Matrix33<real> outer(const BaseVector3<real>& x, const BaseVector3<real>& y)
{
	Matrix33<real> M;
	for (unsigned int i = 0; i < 3; i++)
	{
		for (unsigned int j = 0; j < 3; j++)
			M(i, j) = x[i] * y[j];
	}
	return M;
}

template<typename real>
__host__ __device__
inline Matrix44<real> outer(const Vector4<real>& x, const Vector4<real>& y)
{
	Matrix44<real> M;
	for (unsigned int i = 0; i < 4; i++)
	{
		for (unsigned int j = 0; j < 4; j++)
			M(i, j) = x[i] * y[j];
	}
	return M;
}

template<typename real>
__host__ __device__
inline real det(const BaseMatrix22<real>& M)
{
	return M[0] * M[3] - M[1] * M[2];
}

template<typename real>
__host__ __device__
inline Matrix22<real> inverse(const BaseMatrix22<real>& M, real& detValue)
{
	Matrix22<real> invM;
	detValue = det(M);
	invM[0] = M[3] / detValue;
	invM[1] = -M[1] / detValue;
	invM[2] = -M[2] / detValue;
	invM[3] = M[0] / detValue;
	return invM;
}

template<typename real>
__host__ __device__
inline real dot(const BaseVector2<real>& x, const BaseVector2<real>& y)
{
	return x[0] * y[0] + x[1] * y[1];
}

template<typename real>
__host__ __device__
inline Matrix22<real> transpose(const BaseMatrix22<real>& M)
{
	Matrix22<real> MT;
	MT[0] = M[0];
	MT[1] = M[2];
	MT[2] = M[1];
	MT[3] = M[3];
	return MT;
}

template<typename real>
__host__ __device__ 
inline Vector3<real> operator*(const Matrix33<real>& M, const Vector3<real>& v)
{
	Vector3<real> a;
	for (unsigned int i = 0; i < 3; i++)
	{
		a[i] = real(0);
		for (unsigned int j = 0; j < 3; j++)
			a[i] += M(i, j) * v[j];
	}
	return a;
}

template<typename real>
__host__ __device__
inline Vector3<real> tensordot(const Tensor443<real>& T, const Matrix44<real>& M)
{
	Vector3<real> v;
	for (unsigned int i = 0; i < 3; i++)
	{
		v[i] = real(0);
		for (unsigned int k = 0; k < 4; k++)
		{
			for (unsigned int l = 0; l < 4; l++)
			{
				v[i] += T(k, l, i) * M(k, l);
			}
		}
	}
	return v;
}

template<typename real>
__host__ __device__
inline Matrix22<real> tensordot(const Tensor2222<real>& T, const Matrix22<real>& M)
{
	Matrix22<real> A;
	for (unsigned int k = 0; k < 2; k++)
	{
		for (unsigned int l = 0; l < 2; l++)
		{
			A(k, l) = real(0);
			for (unsigned int i = 0; i < 2; i++)
			{
				for (unsigned int j = 0; j < 2; j++)
					A(k, l) += T(i, j, k, l) * M(i, j);
			}
		}
	}
	return A;
}

template<typename real>
__host__ __device__
inline Matrix22<real> tensordot(const Tensor222<real>& T, const Vector2<real>& v)
{
	Matrix22<real> A;
	for (unsigned int k = 0; k < 2; k++)
	{
		for (unsigned int l = 0; l < 2; l++)
		{
			A(k, l) = real(0);
			for (unsigned int i = 0; i < 2; i++)
				A(k, l) += T(i, k, l) * v[i];
		}
	}		
	return A;
}

template<typename real>
__host__ __device__
inline Matrix22<real> tensordot(const Tensor322<real>& T, const Vector3<real>& v)
{
	Matrix22<real> A;
	for (unsigned int i = 0; i < 2; i++)
	{
		for (unsigned int j = 0; j < 2; j++)
		{
			A(i, j) = real(0);
			for (unsigned int k = 0; k < 3; k++)
				A(i, j) += T(k, i, j) * v(k);
		}
	}
	return A;
}

template<typename real>
__host__ __device__
inline real tensordot(const Matrix22<real>& M, const Matrix22<real>& N)
{
	real a = real(0);
	for (unsigned i = 0; i < 2; i++)
	{
		for (unsigned j = 0; j < 2; j++)
		{
			a += M(i, j) * N(i, j);
		}
	}
	return a;
}

template<typename real>
__host__ __device__
inline Vector3<real> tensordot(const Matrix23<real>& M, const Vector2<real>& v)
{
	Vector3<real> a;
	
	for (unsigned int j = 0; j < 3; j++)
	{
		a[j] = real(0);
		for (unsigned int i = 0; i < 2; i++)
			a[j] += M(i, j) * v[i];
	}
	return a;
}

template<typename real>
__host__ __device__
inline Tensor322<real> operator*(const Tensor322<real>& T, const Matrix33<real>& M)
{
	Tensor322<real> A;
	for (unsigned int i = 0; i < 3; i++)
	{
		for (unsigned int k = 0; k < 2; k++)
		{
			for (unsigned int l = 0; l < 2; l++)
			{
				A(i, k, l) = real(0);
				for (unsigned int j = 0; j < 3; j++)
					A(i, k, l) += T(j, k, l) * M(i, j);
			}
		}
	}
	return A;
}

template<typename real>
__host__ __device__
inline Tensor222<real> operator*(const Tensor322<real>& T, const Matrix23<real>& M)
{
	Tensor222<real> A;
	for (unsigned int i = 0; i < 2; i++)
	{
		for (unsigned int k = 0; k < 2; k++)
		{
			for (unsigned int l = 0; l < 2; l++)
			{
				A(i, k, l) = real(0);
				for (unsigned int j = 0; j < 3; j++)
					A(i, k, l) += T(j, k, l) * M(i, j);
			}
		}
	}
	return A;
}

template<typename real>
__host__ __device__ 
inline Vector2<real> operator*(const Matrix22<real>& M, const Vector2<real>& v)
{
	Vector2<real> y;
	y[0] = M(0, 0) * v[0] + M(0, 1) * v[1];
	y[1] = M(1, 0) * v[0] + M(1, 1) * v[1];
	return y;
}

///
// Operators used in computing the final gradient
///

template<typename real>
__host__ __device__
inline Vector2<real> operator*(const Matrix22<real>& M, const Tensor222<real>& T)
{
	Vector2<real> v;
	for (unsigned int k = 0; k < 2; k++)
	{
		v[k] = real(0);
		for (unsigned int i = 0; i < 2; i++)
		{
			for (unsigned int j = 0; j < 2; j++)
				v[k] += M(i, j) * T(k, i, j);
		}
	}
	return v;
}

template<typename real>
__host__ __device__
inline Matrix22<real> operator*(const Matrix22<real>& M, const Tensor2222<real>& T)
{
	Matrix22<real> A;
	for (unsigned int k = 0; k < 2; k++)
	{
		for (unsigned int l = 0; l < 2; l++)
		{
			A(k, l) = real(0);
			for (unsigned int i = 0; i < 2; i++)
			{
				for (unsigned int j = 0; j < 2; j++)
				{
					A(k,l) += M(i,j) * T(k, l, i, j);
				}
			}
		}
	}
	return A;
}

template<typename real>
__host__ __device__
inline Vector3<real> operator*(const Matrix22<real>& M, const Tensor322<real>& T)
{
	Vector3<real> v;
	for (unsigned int i = 0; i < 3; i++)
	{
		v[i] = real(0);
		for (unsigned int k = 0; k < 2; k++)
		{
			for (unsigned int l = 0; l < 2; l++)
			{
				v[i] += M(k, l) * T(i, k, l);
			}
		}
		
	}
	return v;
}

template<typename real>
__host__ __device__
inline Vector3<real> operator*(const BaseVector3<real>& u, const Matrix33<real>& M)
{
	Vector3<real> v;
	for (unsigned int i = 0; i < 3; i++)
	{
		v[i] = real(0);
		for (unsigned int j = 0; j < 3; j++)
		{
			v[i] += u[j] * M(j, i); // Why do i have to transpose here for the correct gradient?
		}

	}
	return v;
}

template<typename real>
__host__ __device__
inline Tensor322<real> operator*(const Tensor222<real>& T, const Matrix32<real> M)
{
	Tensor322<real> A;
	for (unsigned int k = 0; k < 3; k++)
	{
		for (unsigned int i = 0; i < 2; i++)
		{
			for (unsigned int j = 0; j < 2; j++)
			{
				A(k, i, j) = real(0);
				for (unsigned int l = 0; l < 2; l++)
					A(k, i, j) += M(k, l) * T(l, i, j);
			}
		}
	}
	return A;
}

template<typename real>
__host__ __device__
inline Vector3<real> operator*(const Vector2<real>& u, const Matrix32<real>& M)
{
	Vector3<real> v;
	for (unsigned int i = 0; i < 3; i++)
	{
		v[i] = real(0);
		for (unsigned int j = 0; j < 2; j++)
		{
			v[i] += u[j] * M(i, j);
		}
	}
	return v;
}

template<typename real>
__host__ __device__
inline Matrix44<real> operator*(const Vector3<real>& v, const Tensor443<real>& T)
{
	Matrix44<real> M;
	for (unsigned int i = 0; i < 4; i++)
	{
		for (unsigned int j = 0; j < 4; j++)
		{
			M(i, j) = real(0);
			for (unsigned int k = 0; k < 3; k++)
				M(i, j) += v[k] * T(i, j, k);
		}
	}
	return M;
}

template<typename real>
__host__ __device__
inline Matrix44<real> operator*(const BaseVector3<real>& v, const Tensor443<real>& T)
{
	Matrix44<real> M;
	
	for (unsigned int i = 0; i < 4; i++)
	{
		for (unsigned int j = 0; j < 4; j++)
		{
			M(i, j) = real(0);
			for (unsigned int k = 0; k < 3; k++)
				M(i, j) += v[k] * T(i, j, k);
		}
	}

	return M;
}

template<typename real>
__host__ __device__
inline Vector4<real> operator*(const Vector4<real>& v, const Matrix44<real>& M)
{
	Vector4<real> y;
	for (unsigned int i = 0; i < 4; i++)
	{
		y[i] = real(0);
		for (unsigned int j = 0; j < 4; j++)
		{
			y[i] += v[j] * M(j, i);
		}
	}
	return y;
}

template<typename real>
__host__ __device__
inline Vector4<real> operator*(const Matrix44<real>& M, const Vector4<real>& v)
{
	Vector4<real> y;
	for (unsigned int i = 0; i < 4; i++)
	{
		y[i] = real(0);
		for (unsigned int j = 0; j < 4; j++)
		{
			y[i] += M(i, j) * v[j];
		}
	}
	return y;
}

template<typename real>
__host__ __device__
inline Vector3<real> operator-(ConstVector3<real> const& x, ConstVector3<real> const& y)
{
	Vector3<real> z;
	for (unsigned int i = 0; i < 3; i++)
		z[i] = x[i] - y[i];
	return z;
}