#ifndef HelperFunctions_H
#define HelperFunctions_H

#include "Globals.h"

using namespace std;
using namespace cv;
using namespace cvb;

/*
	// Timing
	double t = (double)getTickCount();
    watershed( img0, markers ); // do the work
    t = (double)getTickCount() - t;
    printf( "execution time = %gms\n", t*1000./getTickFrequency() );
	// or cout << "execution time = " << t*1000. / getTickFrequency() << " ms" << endl;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	if (timing) {
		t = (double)getTickCount() - t;
		cout << "execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
*/

/*
	// Setting bgr colors for a pixel at once
	Mat mymat(myothermat.size(), CV_8UC3);
	mymat.at<Vec3b>(i,j) = Vec3b(0,0,0);
	// or set color ahead of time with Vec3b color(0,0,0)
*/

/*
Efficient OpenCV loop over matrix elements
void access_example()
{
	Mat m = Mat::zeros(2400, 2400, CV_32SC1);
	int rows = m.rows;
	int cols = m.cols;
	unsigned int* p;
	for (int r=0; r<rows; r++) {
		p = m.ptr<unsigned int>(r);
		for (int c=0; c<cols; c++) {
			p[c] = 5;
		}
	}
}
*/

/*
Passing an Eigen::Matrix arg of unknown type, rows, and columns using templates

template<typename Derived>
typename Derived::PlainObject bar(const Eigen::MatrixBase<Derived>& v)
{
typename Derived::PlainObject ret(v.rows(), v.cols());

std::cout << "v size  : " << v.rows() << ", " << v.cols() << std::endl;
std::cout << "ret size: " << ret.rows() << ", " << ret.cols() << std::endl;
std::cout << "v" << endl << v << endl;

return ret;
}
*/

/*
Converting images between OpenCV and Eigen
Note: use RowMajor for Eigen matrices since OpenCV is stored row major

// update this section!! using http://stackoverflow.com/questions/14783329/opencv-cvmat-and-eigenmatrix
// OpenCV to Eigen
cv::Mat A = cv::Mat::zeros(height, width, CV_8UC3);
MatrixXBGR B;
int n_pixels=A.rows*A.cols;
Mat Am=A.reshape(0,n_pixels).clone(); // 0 for channel number leaves it the same - could use 3 here instead
cv2eigen(Am,B);

// Eigen to OpenCV
int n_pixels=height*width;
cv::Mat A = cv::Mat::zeros(1, n_pixels, CV_8UC3);
MatrixXBGR B;
eigen2cv(B,A);
Mat Am=A.reshape(0, height).clone();
*/

/*
Efficient Eigen loop over matrix elements

note that eigen matrices are column-first ordered by default
usage:
Matrix4d M;
loop_matrix<double, 4, 4>(M);

also note:
size_t is the result type of the sizeof operator.  Use size_t for variables that model size or index in an array. size_t conveys semantics: you immediately know it represents a size in bytes or an index, rather than just another integer.  Also, using size_t to represent a size in bytes helps making the code portable

template <typename T, int R, int C>
inline T loop_matrix(const Eigen::Matrix<T,R,C>& M) {
	if (M.size() == 0) return 0;

	// to track where we are in rows and columns of the matrix during the loop without using division/mod operators on the index i
	int r = 0;
	int c = 0;

	for (size_t i = 0, size = M.size(); i < size; i++) {
		T val = (*(M.data() + i));

		// increment c, r; column-first order
		if (r > R) {
			r = 0;
			c++;
		} else r++;
	}
}
*/

// Math
int round(float num, int num_decimal_places);
int array_sum_int(int* ary, int count);
int array_max_index_int(int* ary, int count);

// String conversions
double convert_string_to_double(std::string s);
void convert_double_to_string(double dval, std::string &strval);
int convert_string_to_int(std::string s);
void convert_int_to_string(int intval, std::string* strval);
void convert_long_to_string(long longval, std::string &strval);
char* convert_string_to_chars(std::string strval);
std::string* convert_chars_to_string(char* charval);
char* gen_string_serialized(std::string str, int index);

// I/O
char* build_filename(std::string filepath, char* filename);
void save_image_mat(Mat* img, char* filename, char* extension);

// Vectors
double veclength(Point2d p);
double veclength(Point3d p);
double vecdist(Point3d p1, Point3d p2);
void normalize(Point3d &V);

// Matrices
double ScaleFromExtrinsics(Mat *P); // P is a 4x4 extrinsics matrix of type CV_64F; returns the scale factor being applied by the matrix

// Angles
double AngleBetweenVectors(Point3d p1, Point3d p2);
double FindAngleDegrees(Point2f pt1, Point2f pt2, Point2f pt3, bool clockwise);

// Error checking
bool IsFiniteNumber(double x);

// Display
void display_mat(Mat* img, char* winName);
void mark_point_circle_mat(Mat &img, int x, int y, int radius, Scalar color);
void draw_line_mat(Mat &img, Point p1, Point p2, Scalar color);

// Parsing
std::vector<double> ParseString_Doubles(std::string s);
Mat ParseString_Matrix64F(std::string s, int cols, int rows);

// Sorting
inline bool pairCompare(const std::pair<int, double>& firstElem, const std::pair<int, double>& secondElem) { return firstElem.second < secondElem.second; };

// Conversion
Eigen::Matrix3d ConvertOpenCVMatToEigenMatrix3d(cv::Mat *m); // converts 3x3 OpenCV matrix of type CV_64F to 3x3 eigen matrix of doubles
Eigen::Matrix4d ConvertOpenCVMatToEigenMatrix4d(cv::Mat *m); // converts 4x4 OpenCV matrix of type CV_64F to 4x4 eigen matrix of doubles
Eigen::Matrix<double, 3, 4> ConvertOpenCVMatToEigenMatrix3x4d(cv::Mat *m); // converts 3x4 OpenCV matrix of type CV_64F to 3x4 eigen matrix of doubles
Eigen::Matrix<double, 3, 4> Convert4x4OpenCVExtrinsicsMatTo3x4EigenExtrinsicsMatrixd(cv::Mat *m); // converts 4x4 OpenCV camera extrinsics matrix of type CV_64F to 3x4 eigen camera extrinsics matrix of doubles, dropping the last row of [0 0 0 1]
void ConvertOpenCVBGRMatToEigenMatrices(cv::Mat *cv_m, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_b, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_g, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_r); // updates Eigen matrices to be of size given by cv_m and updates their values to match BGR channels of cv_m

#endif
