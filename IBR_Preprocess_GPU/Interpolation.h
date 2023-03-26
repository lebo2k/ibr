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

	// Linear interpolation
	// updates matrix B of size n x 3 BGR channels of interpolated colors
	// A is the BGR image of size w*h pixels x 3 color channels
	// Mask_mat is the mask that corresponds to A
	// X is the nx1 set of pixel horizontal offsets into A
	// Y is the nx1 set of pixel vertical offsets into A
	// points masked out are given oobv; points interpolated between some pixels masked-in and some masked-out are given colors influenced only by masked-in pixels
	// note that w*h does not have to equal n
	// oobv is the out of bounds value
	// updates values in arg B
	// type T must be float or double
	// ML added an additional feature: assigns oobv not only when pixel outside of screen space bounds, but also when pixel in a masked out coordinate according to Mask; true denotes masked-in pixel, false denotes masked-out pixel
	template<class T> inline static void InterpLinear(const Eigen::Matrix<T, Dynamic, 3> *A_mat, const Eigen::Matrix<T, Dynamic, 1> *X_mat, const Eigen::Matrix<T, Dynamic, 1> *Y_mat, const Matrix<bool, Dynamic, 1> *Mask_mat, const int w, const int h, const int oobv, Eigen::Matrix<T, Dynamic, 3> *B_mat) {
		assert(X_mat->rows() == Y_mat->rows() && B_mat->rows() == X_mat->rows());
		assert(A_mat->rows() == w*h && Mask_mat->rows() == A_mat->rows());

		int num_points = X_mat->rows();
		int col = 3; // # channels in BGR image
		int end = num_points * col;
		int step = h * w;

		T dw = static_cast<T>(w);
		T dh = static_cast<T>(h);

		const T *A = (const T *)A_mat->data();
		const T *X = (const T *)X_mat->data();
		const T *Y = (const T *)Y_mat->data();
		const bool *Mask = (const bool *)Mask_mat->data();
		T *B = (T *)B_mat->data();

		// For each of the interpolation points
		int i, j, k, x, y;
		double u, v;
		double u1, u2, v1, v2; // u1 is split to upper right from upper-left, u2 to lower-right from lower-left, v1 to lower-left from upper-left, v2 from lower-right from upper-right
#pragma omp parallel for if (num_points > 300) num_threads(omp_get_num_procs()) default(shared) private(i,j,k,u,v,x,y,u1,u2,v1,v2)
		for (i = 0; i < num_points; i++) {

			if (X[i] >= 0 && Y[i] >= 0) {
				if (X[i] < (dw - 1)) {
					if (Y[i] < (dh - 1)) {
						// Linearly interpolate
						x = (int)X[i]; // floor integer x
						y = (int)Y[i]; // floor integer y
						k = h * x + y; // col major index of current position

						if ((!Mask[k]) && (!Mask[k + h]) && (!Mask[k + 1]) && (!Mask[k + h + 1])) { // all 4 containing pixels are masked-out, so we are out of bounds
							for (j = i; j < end; j += num_points)
								B[j] = oobv;
						}
						else {
							u1 = X[i] - x; // decimal portion of x
							v1 = Y[i] - y; // decimal portion of y
							u2 = u1;
							v2 = v1;

							if (!Mask[k]) { // ul masked out
								u1 = 1;
								v1 = 1;
							}
							if (!Mask[k + h]) { // ur masked out
								u1 = 0;
								v2 = 1;
							}
							if (!Mask[k + 1]) { // ll masked out
								u2 = 1;
								v1 = 0;
							}
							if (!Mask[k + h + 1]) { // lr masked out
								u2 = 0;
								v2 = 0;
							}

							for (j = i; j < end; j += num_points, k += step) {
								B[j] = (A[k] * (1 - u1) * (1 - v1)) + (A[k + h] * u1 * (1 - v2)) + (A[k + 1] * v1 * (1 - u2)) + (A[k + h + 1] * u2 * v2);
							}
						}
					}
					else if (approx_equal(Y[i], (dh - 1))) {
						// The Y coordinate is on the boundary
						// Avoid reading outside the buffer to avoid crashes
						// Linearly interpolate along X
						x = (int)X[i];
						k = h * x; // col major index of current position
						if ((!Mask[k]) &&
							(!Mask[k + h])) {
							for (j = i; j < end; j += num_points)
								B[j] = oobv;
						}
						else if (!Mask[k]) {
							for (j = i; j < end; j += num_points, k += step)
								B[j] = A[k + h];
						}
						else if (!Mask[k + h]) {
							for (j = i; j < end; j += num_points, k += step)
								B[j] = A[k];
						}
						else {
							u = X[i] - x;
							for (j = i; j < end; j += num_points, k += step)
								B[j] = A[k] + (A[k + h] - A[k]) * u;
						}
					}
					else {
						// Out of bounds
						for (j = i; j < end; j += num_points)
							B[j] = oobv;
					}
				}
				else if (approx_equal(X[i], (dw - 1))) {
					if (Y[i] < dh) {
						// The X coordinate is on the boundary
						// Avoid reading outside the buffer to avoid crashes
						// Linearly interpolate along Y
						y = (int)Y[i];
						k = h * (w - 1) + y; // col major index of current position
						if ((!Mask[k]) &&
							(!Mask[k + 1])) {
							for (j = i; j < end; j += num_points)
								B[j] = oobv;
						}
						else if (!Mask[k]) {
							for (j = i; j < end; j += num_points, k += step)
								B[j] = A[k + 1];
						}
						else if (!Mask[k + 1]) {
							for (j = i; j < end; j += num_points, k += step)
								B[j] = A[k];
						}
						else {
							v = Y[i] - y;
							for (j = i; j < end; j += num_points, k += step)
								B[j] = A[k] + (A[k + 1] - A[k]) * v;
						}
					}
					else if (approx_equal(Y[i], (dh - 1))) {
						// The X and Y coordinates are on the boundary
						// Avoid reading outside the buffer to avoid crashes
						// Output the last value in the array
						k = (h * w) - 1; // col major index of current position
						if (!Mask[k]) {
							for (j = i; j < end; j += num_points)
								B[j] = oobv;
						}
						else {
							for (j = i; j < end; j += num_points, k += step)
								B[j] = A[k];
						}
					}
					else {
						// Out of bounds
						for (j = i; j < end; j += num_points)
							B[j] = oobv;
					}
				}
				else {
					// Out of bounds
					for (j = i; j < end; j += num_points)
						B[j] = oobv;
				}
			}
			else {
				// Out of bounds
				for (j = i; j < end; j += num_points)
					B[j] = oobv;
			}
		}
	}

	// Linear interpolation
	// updates matrix B of size n x 1 of interpolated values
	// A is the BGR image of size w*h pixels x 1
	// Mask_mat is the mask that corresponds to A
	// X is the nx1 set of pixel horizontal offsets into A
	// Y is the nx1 set of pixel vertical offsets into A
	// points masked out are given oobv; points interpolated between some pixels masked-in and some masked-out are given colors influenced only by masked-in pixels
	// note that w*h does not have to equal n
	// oobv is the out of bounds value
	// updates values in arg B
	// type T must be float or double
	// ML added an additional feature: assigns oobv not only when pixel outside of screen space bounds, but also when pixel in a masked out coordinate according to Mask; true denotes masked-in pixel, false denotes masked-out pixel
	template<class T> inline static void InterpLinear(const Eigen::Matrix<T, Dynamic, 1> *A_mat, const Eigen::Matrix<T, Dynamic, 1> *X_mat, const Eigen::Matrix<T, Dynamic, 1> *Y_mat, const Matrix<bool, Dynamic, 1> *Mask_mat, const int w, const int h, const int oobv, Eigen::Matrix<T, Dynamic, 1> *B_mat) {
		assert(X_mat->rows() == Y_mat->rows() && B_mat->rows() == X_mat->rows());
		assert(A_mat->rows() == w*h && Mask_mat->rows() == A_mat->rows());

		int num_points = X_mat->rows();
		int end = num_points;
		int step = h * w;

		T dw = static_cast<T>(w);
		T dh = static_cast<T>(h);

		const T *A = (const T *)A_mat->data();
		const T *X = (const T *)X_mat->data();
		const T *Y = (const T *)Y_mat->data();
		const bool *Mask = (const bool *)Mask_mat->data();
		T *B = (T *)B_mat->data();

		// For each of the interpolation points
		int i, k, x, y;
		double u, v;
		//double u1, u2, v1, v2; // u1 is split to upper right from upper-left, u2 to lower-right from lower-left, v1 to lower-left from upper-left, v2 from lower-right from upper-right
		//double u10, u11, u20, u21, v10, v11, v20, v21;
#pragma omp parallel for if (num_points > 300) num_threads(omp_get_num_procs()) default(shared) private(i,k,u,v,x,y)
		for (i = 0; i < num_points; i++) {

			if (X[i] >= 0 && Y[i] >= 0) {
				if (X[i] < (dw - 1)) {
					if (Y[i] < (dh - 1)) {
						// Linearly interpolate
						x = (int)X[i]; // floor integer x
						y = (int)Y[i]; // floor integer y
						k = h * x + y; // col major index of current position
						


						if ((!Mask[k]) && (!Mask[k + h]) && (!Mask[k + 1]) && (!Mask[k + h + 1])) // all 4 containing pixels are masked-out, so we are out of bounds
							B[i] = oobv;
						else if (Mask[k])
							B[i] = A[k];
						else if (Mask[k + h])
							B[i] = A[k + h];
						else if (Mask[k + 1])
							B[i] = A[k + 1];
						else if (Mask[k + h + 1])
							B[i] = A[k + h + 1];
						/*
						else {
							u11 = X[i] - x; // decimal portion of x
							v11 = Y[i] - y; // decimal portion of y
							u10 = 1 - u11;
							v10 = 1 - v11;
							u21 = u11;
							v21 = v11;
							u20 = 1 - u21;
							v20 = 1 - v21;

							if (!Mask[k]) { // ul masked out
								u10 = 0;
								v10 = 0;
								if (Mask[k + h])
									u11 = 1;
								if (Mask[k + 1])
									v11 = 1;
							}
							if (!Mask[k + h + 1]) { // lr masked out
								u21 = 0;
								v21 = 0;
								if (Mask[k + h])
									v20 = 1;
								if (Mask[k + 1])
									u20 = 1;
							}
							if (!Mask[k + h]) { // ul masked out
								u11 = 0;
								v20 = 0;
								if (Mask[k])
									u10 = 1;
								if (Mask[k + h + 1])
									v21 = 1;
							}
							if (!Mask[k + 1]) { // ul masked out
								u20 = 0;
								v11 = 0;
								if (Mask[k])
									v10 = 1;
								if (Mask[k + h + 1])
									u21 = 1;
							}

							B[i] = (A[k] * u10 * v10) + (A[k + h] * u11 * v20) + (A[k + 1] * v11 * u20) + (A[k + h + 1] * u21 * v21);

							//out = A[k] + (A[k + h] - A[k]) * u; // out = curr_val + (val_right - curr_val) * u
							//out += ((A[k + 1] - out) + (A[k + h + 1] - A[k + 1]) * u) * v; // out += ((val_down - out) + (val_down_and_right - val_down) * u) * v
							//= LL - UL*(1-u)*v - UR*u*v + LR*u*v - LL*u*v
							//= UL*(1 - u)*(1 - v) + UR*u*(1 - v) + LL*(1 - u)*v + LR*u*v
						}
						*/
					}
					else if (approx_equal(Y[i], (dh - 1))) {
						// The Y coordinate is on the boundary
						// Avoid reading outside the buffer to avoid crashes
						// Linearly interpolate along X
						x = (int)X[i];
						k = h * x; // col major index of current position
						if ((!Mask[k]) &&
							(!Mask[k + h]))
							B[i] = oobv;
						else if (!Mask[k])
							B[i] = A[k + h];
						else if (!Mask[k + h])
							B[i] = A[k];
						else {
							u = X[i] - x;
							B[i] = A[k] + (A[k + h] - A[k]) * u;
						}
					}
					else {
						// Out of bounds
						B[i] = oobv;
					}
				}
				else if (approx_equal(X[i], (dw - 1))) {
					if (Y[i] < dh) {
						// The X coordinate is on the boundary
						// Avoid reading outside the buffer to avoid crashes
						// Linearly interpolate along Y
						y = (int)Y[i];
						k = h * (w - 1) + y; // col major index of current position
						if ((!Mask[k]) &&
							(!Mask[k + 1]))
							B[i] = oobv;
						else if (!Mask[k])
							B[i] = A[k + 1];
						else if (!Mask[k + 1])
							B[i] = A[k];
						else {
							v = Y[i] - y;
							B[i] = A[k] + (A[k + 1] - A[k]) * v;
						}
					}
					else if (approx_equal(Y[i], (dh - 1))) {
						// The X and Y coordinates are on the boundary
						// Avoid reading outside the buffer to avoid crashes
						// Output the last value in the array
						k = (h * w) - 1; // col major index of current position
						if (!Mask[k])
							B[i] = oobv;
						else
							B[i] = A[k];
					}
					else {
						// Out of bounds
						B[i] = oobv;
					}
				}
				else {
					// Out of bounds
					B[i] = oobv;
				}
			}
			else {
				// Out of bounds
				B[i] = oobv;
			}
		}
	}


public:

	// updates matrix B of size n x 3 channels of interpolated values
	// A is the image of size w*h pixels x 3 channels
	// X is the nx1 set of pixel horizontal offsets into A
	// Y is the nx1 set of pixel vertical offsets into A
	// static is the mask that corresponds to A
	// points masked out are given oobv; points interpolated between some pixels masked-in and some masked-out are given colors influenced only by masked-in pixels
	// note that w*h does not have to equal n
	// oobv is the out of bounds value
	// type T must be float or double
	// ML added an additional feature: assigns oobv not only when pixel outside of screen space bounds, but also when pixel in a masked out coordinate according to Mask; true denotes masked-in pixel, false denotes masked-out pixel
	template<class T> inline static void Interpolate(int width, int height, const Eigen::Matrix<T, Dynamic, 3> *A, const Eigen::Matrix<T, Dynamic, 1> *X, const Eigen::Matrix<T, Dynamic, 1> *Y, const Matrix<bool, Dynamic, 1> *Mask, int oobv, Eigen::Matrix<T, Dynamic, 3> *B) {
		bool timing = false; double t;
		if (timing) t = (double)getTickCount();

		assert(X->rows() == Y->rows() && B->rows() == X->rows());
		assert(A->rows() == width*height && Mask->rows() == A->rows());
		assert(typeid(T) == typeid(float) || typeid(T) == typeid(double));

		T oobv_val = static_cast<T>(oobv);

		InterpLinear<T>(A, X, Y, Mask, width, height, oobv_val, B);

		if (timing) {
			t = (double)getTickCount() - t;
			cout << "Interpolation::Interpolate() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
		}
	}

	// updates matrix B of size n x 1 channel of interpolated values
	// A is the image of size w*h pixels x 1 channel
	// X is the nx1 set of pixel horizontal offsets into A
	// Y is the nx1 set of pixel vertical offsets into A
	// static is the mask that corresponds to A
	// points masked out are given oobv; points interpolated between some pixels masked-in and some masked-out are given colors influenced only by masked-in pixels
	// note that w*h does not have to equal n
	// oobv is the out of bounds value
	// type T must be float or double
	// ML added an additional feature: assigns oobv not only when pixel outside of screen space bounds, but also when pixel in a masked out coordinate according to Mask; true denotes masked-in pixel, false denotes masked-out pixel
	template<class T> inline static void Interpolate(int width, int height, const Eigen::Matrix<T, Dynamic, 1> *A, const Eigen::Matrix<T, Dynamic, 1> *X, const Eigen::Matrix<T, Dynamic, 1> *Y, const Matrix<bool, Dynamic, 1> *Mask, int oobv, Eigen::Matrix<T, Dynamic, 1> *B) {
		bool timing = false; double t;
		if (timing) t = (double)getTickCount();

		assert(X->rows() == Y->rows() && B->rows() == X->rows());
		assert(A->rows() == width*height && Mask->rows() == A->rows());
		assert(typeid(T) == typeid(float) || typeid(T) == typeid(double));

		T oobv_val = static_cast<T>(oobv);

		InterpLinear<T>(A, X, Y, Mask, width, height, oobv_val, B);

		if (timing) {
			t = (double)getTickCount() - t;
			cout << "Interpolation::Interpolate() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
		}
	}

	// updates Inbound_mat to be false when interpolated X_mat, Y_mat screen space coordinates are masked out by Mask_mat but are fully inside the screen space (so can accomodate images that don't have cropped views of the product)
	// leaves other mask values unchanged
	// if any pixel of containing box of pixels is masked-in, the floating point coordinates are considered masked-in, otherwise masked-out; the reason for this is that Interpolate() uses any of the four surrounding pixels available, so can work with as few as one masked-in
	// w and h are sizes of mask if were laid out in screen space
	// masking is only effective if a projected point is within a mask's screen space.  Since an camera may frame a scene such that the object of interest is not fully contained, we can't simply call projected points outside of the screen space "masked-out."  closeup_xmin, closeup_xmax, closeup_ymin, closeup_ymax should be true in cases where photo is a close-up that doesn't fully capture the object within the screen space on the indicated side (value assigned by testing for valid masked-in pixels along the appropriate screen space side's edge) and false otherwise.  If one is true, that side of the screen space is not considered a limiting factor on being masked-in, so points falling outside of it are not masked-out; if it's false, points falling outside of that bound are considered masked out.
	template<class T> inline static void InterpolateAgainstMask(const int w, const int h, const Eigen::Matrix<T, Dynamic, 1> *X_mat, const Eigen::Matrix<T, Dynamic, 1> *Y_mat, const Matrix<bool, Dynamic, 1> *Mask_mat, Eigen::Matrix<bool, Dynamic, 1> *Inbound_mat, bool closeup_xmin, bool closeup_xmax, bool closeup_ymin, bool closeup_ymax) {
		assert(X_mat->rows() == Y_mat->rows() && Inbound_mat->rows() == X_mat->rows());

		//bool debug = true;
		//bool timing = true;
		//double t;
		//if (timing) t = (double)getTickCount();

		int num_points = X_mat->rows();
		int step = h * w;

		//if (debug) cout << "Interpolation::InterpolateAgainstMask() with num_points " << num_points << endl;

		T dw = static_cast<T>(w - 1);
		T dh = static_cast<T>(h - 1);

		const T *X = (const T *)X_mat->data();
		const T *Y = (const T *)Y_mat->data();
		const bool *Mask = (const bool *)Mask_mat->data();
		bool *Inbound = (bool *)Inbound_mat->data();
		T xval, yval;

		int num_groups = GLOBAL_NUM_PROCESSORS; // should equal the number of processors
		int num_points_per_group = floor(static_cast<float>(num_points) / static_cast<float>(num_groups));
		
		// For each of the interpolation points
		int group, i, k, x, y, start_point, end_point;
#pragma omp parallel for if (num_points > 300) num_threads(omp_get_num_procs()) default(shared) private(i,k,x,y,xval,yval,start_point,end_point,group,X,Y,Inbound)
		for (group = 0; group < num_groups; group++) { // split into groups by processor so can have separate incrementing data pointers
			start_point = group * num_points_per_group;
			if (group < (num_groups - 1)) end_point = start_point + num_points_per_group;
			else end_point = num_points;
			//if (debug) cout << "processing group " << group << " with start_point " << start_point << " and end_point " << end_point << endl;
			X = (const T *)X_mat->data() + start_point;
			Y = (const T *)Y_mat->data() + start_point;
			Inbound = (bool *)Inbound_mat->data() + start_point;
			for (i = start_point; i < end_point; i++) {
				xval = (*X);
				yval = (*Y);

				// check against screen space bounds in cases where photo doesn't crop object of interest on one or more sides
				if ((xval < 0) && (!closeup_xmin))
					(*Inbound) = false;
				else if ((xval >= w) && (!closeup_xmax))
					(*Inbound) = false;
				else if ((yval < 0) && (!closeup_ymin))
					(*Inbound) = false;
				else if ((yval >= h) && (!closeup_ymax))
					(*Inbound) = false;
				else if (xval >= 0 && yval >= 0) {
					if (xval < dw) {
						if (yval < dh) {
							// Linearly interpolate
							x = (int)xval; // rounded integer x
							y = (int)yval; // rounded integer y
							k = h * x + y; // col major index of current position
							if ((!Mask[k]) && (!Mask[k + h]) && (!Mask[k + 1]) && (!Mask[k + h + 1])) // all 4 containing pixels are masked-out, so we are out of bounds
								(*Inbound) = false;
						}
						else if (approx_equal(yval, dh)) {
							// The Y coordinate is on the boundary
							// Avoid reading outside the buffer to avoid crashes
							// Linearly interpolate along X
							x = (int)xval;
							k = h * x; // col major index of current position
							if ((!Mask[k]) && (!Mask[k + h])) // both valid containing pixels are masked-out, so we are out of bounds
								(*Inbound) = false;
						}
					}
					else if (approx_equal(xval, dw)) {
						if (yval < dh) {
							// The X coordinate is on the boundary
							// Avoid reading outside the buffer to avoid crashes
							// Linearly interpolate along Y
							y = (int)yval;
							k = h * (w - 1) + y; // col major index of current position
							if ((!Mask[k]) && (!Mask[k + 1])) // both valid containing pixels are masked-out, so we are out of bounds
								(*Inbound) = false;
						}
						else if (approx_equal(yval, dh)) {
							// The X and Y coordinates are on the boundary
							// Avoid reading outside the buffer to avoid crashes
							// Output the last value in the array
							k = (h * w) - 1; // col major index of current position
							if (!Mask[k])
								(*Inbound) = false;
						}
					}
				}

				X++;
				Y++;
				Inbound++;
			} // points
		} // groups

		//if (timing) {
		//	t = (double)getTickCount() - t;
		//	cout << "Interpolation::InterpolateAgainstMask() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
		//}

		//if (debug) cin.ignore();
	}

};

#endif