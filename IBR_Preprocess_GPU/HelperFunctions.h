#ifndef HelperFunctions_H
#define HelperFunctions_H

#include "Globals.h"
#include "decomposition.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// put here not Globals.h since the files include each other
const enum GLOBAL_AGI_CAMERA_ORIENTATION { AGO_ORIGINAL = 1, AGO_ROTATED_RIGHT = 8, AGO_ROTATED_LEFT = 6 }; // Agisoft camera orientation codes; original means matches sensor and side means on its side so width and height are switched


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
bool approx_equal(float x, float y);
bool approx_equal(float x, double y);
bool approx_equal(double x, float y);
bool approx_equal(double x, double y);

// String conversions
double convert_string_to_double(std::string s);
void convert_double_to_string(double dval, std::string &strval);
int convert_string_to_int(std::string s);
void convert_int_to_string(int intval, std::string* strval);
void convert_long_to_string(long longval, std::string &strval);
char* convert_string_to_chars(std::string strval);
std::string convert_chars_to_string(char* charval);
char* gen_string_serialized(std::string str, int index);
void StringToWString(std::wstring &ws, const std::string &s);

// I/O
char* build_filename(std::string filepath, char* filename);
void save_image_mat(Mat* img, const char* filename, char* extension);
cv::Mat readPPM(const char *filename); // reads a .ppm image file into a cv::Mat BGR image
Eigen::MatrixXf readPGM(const char *filename); // reads a .pgm image file into a cv::Mat GRAY image
__int64 FileSize(std::wstring name);


// only takes rows and cols into account (ignores maxRows and maxCols)
// requires m is column-major order
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
inline void SaveEigenMatrix(std::string name, int type_name, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& m) {
	assert((!(m.Flags & Eigen::RowMajorBit)));
	
	std::string fn = GLOBAL_FILEPATH_DATA + "matrix_" + name + ".adf";
	FILE* pFile = fopen(fn.c_str(), "wb"); // write binary mode

	// write 50 character (byte) asset name that is binary-zero-terminated (char \0)
	char name_chars[50];
	std::size_t length = name.copy(name_chars, name.length(), 0);
	name_chars[length] = '\0';
	fwrite((void*)name_chars, sizeof(char), 50, pFile);

	// write matrix metadata
	int rows = m.rows();
	int cols = m.cols();
	fwrite((void*)&rows, sizeof(int), 1, pFile);
	fwrite((void*)&cols, sizeof(int), 1, pFile);
	fwrite((void*)&type_name, sizeof(int), 1, pFile);

	// write matrix data
	fwrite((void*)m.data(), sizeof(_Tp), m.rows() * m.cols(), pFile);

	fclose(pFile);
}

// requires m is column-major order
// type_name must be one of GLOBAL_TYPE_NAME options - just can't seem to require that in the arg list without generating a compile error
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
inline void LoadEigenMatrix(std::string name, int type_name, Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& m) {
	assert((!(m.Flags & Eigen::RowMajorBit)));

	bool debug = false;

	std::string fn = GLOBAL_FILEPATH_DATA + "matrix_" + name + ".adf";
	FILE* pFile = fopen(fn.c_str(), "rb"); // read binary mode

	if (pFile == NULL) {
		cerr << "LoadEigenMatrix() file not found" << endl;
		return;
	}

	// read asset name
	char name_chars[50];
	fread(name_chars, sizeof(char), 50, pFile);
	char* pch = strchr(name_chars, '\0');
	if (pch == NULL) pch = name_chars + 49;
	char *name_chars_tmp = new char[pch - name_chars + 1];
	memcpy(name_chars_tmp, &name_chars[0], pch - name_chars + 1);
	string s = convert_chars_to_string(name_chars_tmp);
	delete[] name_chars_tmp;
	if (debug) cout << "Name found was " << s << endl;

	// read # rows and columns
	int rows_read, cols_read;
	fread((void*)&rows_read, sizeof(int), 1, pFile);
	fread((void*)&cols_read, sizeof(int), 1, pFile);
	if (debug) cout << rows_read << " rows and " << cols_read << " cols" << endl;
	GLOBAL_TYPE_NAME type_name_read;
	fread((void*)&type_name_read, sizeof(int), 1, pFile);

	// check metadata against data provided in args
	assert(type_name == type_name_read);
	assert(m.rows() == rows_read);
	assert(m.cols() == cols_read);
	if ((type_name != type_name_read) ||
		(m.rows() != rows_read) ||
		(m.cols() != cols_read))
		return;

	// read data
	_Tp *m_data = m.data();
	fread((void*)m_data, sizeof(_Tp), rows_read * cols_read, pFile);

	fclose(pFile);
}

// requires m is correct size for contents of file
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
inline void LoadEigenMatrixBasic(std::string filename, Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> &m) {
	std::string fn = GLOBAL_FILEPATH_DATA + filename;
	FILE* pFile = fopen(fn.c_str(), "rb"); // read binary mode

	if (pFile == NULL) {
		cerr << "LoadEigenMatrixBasic() file not found" << endl;
		return;
	}

	// read data
	_Tp *m_data = m.data();
	fread((void*)m_data, sizeof(_Tp), m.rows() * m.cols(), pFile);

	fclose(pFile);
}


// Vectors
double veclength(Point2d p);
double veclength(Point3d p);
float veclength(Point3f p);
double vecdist(Point3d p1, Point3d p2);
float vecdist(Point3f p1, Point3f p2);
float vecdist(Point2f p1, Point2f p2);
void normalize(Point3d &V);
void normalize(Point3f &V);

// Matrices
double ScaleFromExtrinsics(Matrix4d P); // P is a 4x4 extrinsics matrix; returns the scale factor being applied by the matrix

// Angles
double RadiansToDegrees(double rads);
double AngleRadiansBetweenVectors(Point3d p1, Point3d p2);
double AngleDegreesBetweenVectors(Point3d p1, Point3d p2);
double FindAngleDegrees(Point2f pt1, Point2f pt2, Point2f pt3, bool clockwise);

// Error checking
bool IsFiniteNumber(double x);

// Display
void display_mat(Mat* img, char* winName, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL);
void display_mat_existingwindow(Mat* img, string winname, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL);
void mark_point_circle_mat(Mat &img, int x, int y, int radius, Scalar color);
void draw_line_mat(Mat &img, Point p1, Point p2, Scalar color);

// Parsing
std::vector<double> ParseString_Doubles(std::string s);
Eigen::MatrixXd ParseString_Matrixd(std::string s, int cols, int rows);

// Sorting
inline bool pairCompare(const std::pair<int, double>& firstElem, const std::pair<int, double>& secondElem) { return firstElem.second < secondElem.second; };

// Images
inline int PixIndexFwdRM(Point pt, int width) { return (pt.y * width) + pt.x; }; // forward computation of index into pixel position data structures from pixel pt in image img_; assumes indices are stored in row-major order
inline Point PixIndexBwdRM(int idx, int width) { Point pt; pt.y = std::floor(static_cast<float>(idx) / static_cast<float>(width)); pt.x = idx - pt.y*width; return pt; }; // backward computation of pixel pt in image img_ from index into pixel position data structures; assumes indices are stored in row-major order
inline int PixIndexFwdCM(Point pt, int height) { return (pt.x * height) + pt.y; }; // forward computation of index into pixel position data structures from pixel pt in image img_; assumes indices are stored in column-major order
inline Point PixIndexBwdCM(int idx, int height) { Point pt; pt.x = std::floor(idx / height); pt.y = idx - pt.x*height; return pt; }; // backward computation of pixel pt in image img_ from index into pixel position data structures; assumes indices are stored in column-major order


// Debug
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
static void DebugPrintMatrix(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *X, std::string name) {
	cout << endl << name << endl;
	cout << X->rows() << " rows x " << X->cols() << " cols" << endl;
	cout << "min " << X->minCoeff() << ", max " << X->maxCoeff() << endl;

	Eigen::Matrix<double, _rows, _cols, _options, _maxRows, _maxCols> tmp = X->cast<double>();
	cout << "sum " << tmp.sum() << ", count " << X->count() << endl;

	int numel = X->rows() * X->cols();
	
	if (numel < 100) {
		cout << (*X) << endl;
	}
	else if (X->rows() < 10) {
		cout << "	" << "First 10 cols of " << name << endl;
		for (int idx = 0; idx < 10; idx++) {
			cout << "Column " << idx << ":" << endl << X->col(idx) << endl;
		}
		cout << "	" << "Last 10 cols of " << name << endl;
		for (int idx = X->cols() - 10; idx < X->cols(); idx++) {
			cout << "Column " << idx << ":" << endl << X->col(idx) << endl;
		}
	}
	else if (X->cols() < 10) {
		cout << "	" << "First 10 rows of " << name << endl;
		for (int idx = 0; idx < 10; idx++) {
			cout << "Row " << idx << ":" << endl << X->row(idx) << endl;
		}
		cout << "	" << "Last 10 rows of " << name << endl;
		for (int idx = X->rows() - 10; idx < X->rows(); idx++) {
			cout << "Row " << idx << ":" << endl << X->row(idx) << endl;
		}
	}
	else {
		cout << "	" << "First 10 elements of " << name << endl;
		for (int idx = 0; idx < 10; idx++) {
			cout << "Element " << idx << ":" << endl << (*X)(idx) << endl;
		}
		cout << "	" << "Last 10 elements of " << name << endl;
		for (int idx = numel-10; idx < numel; idx++) {
			cout << "Element " << idx << ":" << endl << (*X)(idx) << endl;
		}
	}

	Matrix<bool, _rows, _cols> Xzeros = tmp.cwiseAbs().array() < GLOBAL_FLOAT_ERROR;
	cout << "number of coefficients that are zero: " << Xzeros.count() << endl;

	cin.ignore();
}

inline bool sort_eigenvalues(const double& l, const double& r) { return l > r; }; // sort in decreasing order
Mat DetermineEigenTransform(Mat* imgGray, Matrix<bool, Dynamic, 1> *mask);

map<unsigned int, int> GetLabelCounts(Matrix<unsigned int, Dynamic, Dynamic> *seg); // returns map of label => count of coefficients

#endif
