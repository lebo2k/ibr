#ifndef Interpolation_H
#define Interpolation_H

#include "Globals.h"

#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <limits> // add this to use numeric_limits class

class Interpolation {

private:

	// function for correct rounding
	template<class U, class T> static inline U saturate_cast(T val)
	{
		if (numeric_limits<U>::is_integer && !numeric_limits<T>::is_integer) {
			if (numeric_limits<U>::is_signed)
				return val > 0 ? (val > (T)numeric_limits<U>::max() ? numeric_limits<U>::max() : static_cast<U>(val + 0.5)) : (val < (T)numeric_limits<U>::min() ? numeric_limits<U>::min() : static_cast<U>(val - 0.5));
			else
				return val > 0 ? (val > (T)numeric_limits<U>::max() ? numeric_limits<U>::max() : static_cast<U>(val + 0.5)) : 0;
		}
		return static_cast<U>(val);
	}

	static Eigen::Matrix<int, Dynamic, 3> InterpLinear(const Eigen::Matrix<uchar, Dynamic, 3> A, const Eigen::Matrix<double, Dynamic, 1> X, const Eigen::Matrix<double, Dynamic, 1> Y, const int w, const int h, const int oobv); // Linear interpolation


public:
	
	static Eigen::Matrix<int, Dynamic, 3> Interpolate(int width, int height, Eigen::Matrix<uchar, Dynamic, 3> A, Eigen::Matrix<double, Dynamic, 1> X, Eigen::Matrix<double, Dynamic, 1> Y, int oobv);


};

#endif