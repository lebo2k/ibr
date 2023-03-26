#include "Interpolation.h"

// n pixels = width*height
// A is the BGR image of size n pixels x 3 BGR channels and type uchar
// X is the nx1 set of pixel horizontal offsets
// Y is the nx1 set of pixel vertical offsets
// oobv is the out of bounds value
Eigen::Matrix<int, Dynamic, 3> Interpolation::Interpolate(int width, int height, Eigen::Matrix<uchar, Dynamic, 3> A, Eigen::Matrix<double, Dynamic, 1> X, Eigen::Matrix<double, Dynamic, 1> Y, int oobv) {
	assert(X.rows() == Y.rows() && X.cols() == Y.cols(), "Interpolation::Interp() X and Y must be of same size");
	assert(A.rows() == width*height && X.rows() == width*height && Y.rows() == width*height, "Interpolation::Interp() A, X, and Y must have the same number of rows corresponding to the number of points");
	int num_points = width * height;


	Eigen::Matrix<int, Dynamic, 3> B = InterpLinear(A, X, Y, width, height, oobv);

	return B;
}

// Linear interpolation
// n points = w*h where w is width and h is height
// returns output matrix B of size n x 3 BGR channels and type int
// A is the BGR image of size n pixels x 3 color channels and type uchar
// X is the nx1 set of pixel horizontal offsets
// Y is the nx1 set of pixel vertical offsets
// oobv is the out of bounds value
Eigen::Matrix<int, Dynamic, 3> Interpolation::InterpLinear(const Eigen::Matrix<uchar, Dynamic, 3> A, const Eigen::Matrix<double, Dynamic, 1> X, const Eigen::Matrix<double, Dynamic, 1> Y, const const int w, const int h, const int oobv) {
	assert(X.rows() == Y.rows() && X.cols() == Y.cols(), "Interpolation::InterpLinear() X and Y must be of same size");
	assert(A.rows() == w*h && X.rows() == w*h && Y.rows() == w*h, "Interpolation::InterpLinear() A, X, and Y must have the same number of rows corresponding to the number of points");

	int num_points = w*h;
	int col = 3; // # channels in BGR image
	int end = num_points * col;
	int step = h * w;

	double dw = (double)w;
	double dh = (double)h;

	Eigen::Matrix<int, Dynamic, 3> B(num_points, 3);
	B = A.cast<int>();

	// For each of the interpolation points
	int i, j, k, x, y;
	double u, v, out;
#pragma omp parallel for if (num_points > 300) num_threads(omp_get_num_procs()) default(shared) private(i,j,k,u,v,x,y,out)
	for (i = 0; i < num_points; i++) {

		if (X(i) >= 0 && Y(i) >= 0) {
			if (X(i) < (dw - 1)) {
				if (Y(i) < (dh - 1)) {
					// Linearly interpolate
					x = (int)X(i); // rounded integer x
					y = (int)Y(i); // rounded integer y
					u = X(i) - x; // decimal portion of x
					v = Y(i) - y; // decimal portion of y
					k = h * x + y; // row major index
					for (int j = 0; j < 3; j++) {
						out = A(k, j) + (A(k + h, j) - A(k, j)) * u;
						out += ((A(k + 1, j) - out) + (A(k + h + 1, j) - A(k + 1, j)) * u) * v;
						B(k,j) = saturate_cast<int, double>(out);
					}
				}
				else if (Y(i) == (dh - 1)) {
					// The Y coordinate is on the boundary
					// Avoid reading outside the buffer to avoid crashes
					// Linearly interpolate along X
					x = (int)X(i);
					u = X(i) - x;
					k = h * x; // row major index
					for (int j = 0; j < 3; j++) 
						B(k,j) = saturate_cast<int, double>(A(k,j) + (A(k + h,j) - A(k,j)) * u);
				}
				else {
					// Out of bounds
					for (int j = 0; j < 3; j++)
						B(i,j) = oobv;
				}
			}
			else if (X(i) == (dw - 1)) {
				if (Y(i) < (dh - 1)) {
					// The X coordinate is on the boundary
					// Avoid reading outside the buffer to avoid crashes
					// Linearly interpolate along Y
					y = (int)Y(i);
					v = Y(i) - y;
					k = h * w + y; // row major index
					for (int j = 0; j < 3; j++)
						B(k, j) = saturate_cast<int, double>(A(k, j) + (A(k + 1, j) - A(k, j)) * v);
				}
				else if (Y(i) == (dh - 1)) {
					// The X and Y coordinates are on the boundary
					// Avoid reading outside the buffer to avoid crashes
					// Output the last value in the array
					k = h * w; // row major index
					for (int j = 0; j < 3; j++)
						B(k,j) = saturate_cast<int, double>(A(k,j));
				}
				else {
					// Out of bounds
					for (int j = 0; j < 3; j++)
						B(i,j) = oobv;
				}
			}
			else {
				// Out of bounds
				for (int j = 0; j < 3; j++)
					B(i, j) = oobv;
			}
		}
		else {
			// Out of bounds
			for (int j = 0; j < 3; j++)
				B(i, j) = oobv;
		}
	}

	return B;
}