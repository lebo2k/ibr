#include "HelperFunctions.h"

using namespace std;
using namespace cv;
using namespace cvb;


/*
	Math
*/

int round(float num, int num_decimal_places)
{
	return ceil(num*pow(10.0,num_decimal_places))/pow(10.0,num_decimal_places);
}

int array_sum_int(int* ary, int count)
{
	int sum = 0;
	for ( int i=0; i<count; i++ )
	{
		sum += ary[i];
	}
	return sum;
}

int array_max_index_int(int* ary, int count)
{
	int max_index = -1;
	int max = 0;
	for ( int i=0; i<count; i++ )
	{
		if ( ary[i]>max )
		{
			max = ary[i];
			max_index = i;
		}
	}
	return max_index;
}

/*
	String conversions
*/

int convert_string_to_int(std::string s)
{
	return atoi(s.c_str());
}

double convert_string_to_double(std::string s)
{
	return atof(s.c_str());
}

void convert_int_to_string(int intval, std::string* strval)
{
	ostringstream convert;   // stream used for the conversion
	convert << intval;      // insert the textual representation of 'Number' in the characters in the stream
	*strval = convert.str(); // set 'Result' to the contents of the stream
}

void convert_long_to_string(long longval, std::string &strval)
{
	ostringstream convert;   // stream used for the conversion
	convert << longval;      // insert the textual representation of 'Number' in the characters in the stream
	strval = convert.str(); // set 'Result' to the contents of the stream
}

void convert_double_to_string(double dval, std::string &strval)
{
	ostringstream convert;   // stream used for the conversion
	convert << dval;      // insert the textual representation of 'Number' in the characters in the stream
	strval = convert.str(); // set 'Result' to the contents of the stream
}

char* convert_string_to_chars(std::string strval)
{
	char * str_chars = new char[strval.length() + 1];
	std::strcpy(str_chars,strval.c_str());
	return str_chars;
}

std::string* convert_chars_to_string(char* charval)
{
	stringstream ss;
	std::string *s = new std::string;
	ss << charval;
	ss >> *s;
	return s;
}

/*
	IO
*/

void save_image_mat(Mat* img, char* filename, char* extension)
{
	std::string fn_img = GLOBAL_FILEPATH_DATA + filename + "." + extension;
	imwrite(fn_img, *img);
}

char* build_filename(std::string filepath, char* filename)
{
	std::string fn = filepath + filename;
	char* fn_chars = convert_string_to_chars(fn);
	return fn_chars;
}

/*
	Vectors
*/

double veclength(Point2d p)
{
	double l = sqrt(pow(p.x, 2) + pow(p.y, 2));
	return l;
}

double veclength(Point3d p)
{
	double l = sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
	return l;
}

double vecdist(Point3d p1, Point3d p2)
{
	double l = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
	return l;
}

void normalize(Point3d &V)
{
	double lenV = veclength(V);
	V.x = V.x / lenV;
	V.y = V.y / lenV;
	V.z = V.z / lenV;
}

/*
	Matrices
*/

// P is a 4x4 extrinsics matrix of type CV_64F
// returns the scale factor being applied by the matrix
double ScaleFromExtrinsics(Mat *P) {
	assert(P->rows == 4 && P->cols == 4, "ScaleFromExtrinsics() P must be a 4x4 matrix");
	assert(P->type() == CV_64F, "ScaleFromExtrinsics() P must be of type CV_64F");
	double *p;
	double scale_factor, sf;
	for (int r = 0; r < 3; r++) { // last row doesn't contain the scale factor
		p = P->ptr<double>(r);
		sf = powf(p[0], 2.) + powf(p[1], 2.) + powf(p[2], 2.);
		if (r > 0) assert(abs(sf-scale_factor)<0.0001, "ScaleFromExtrinsics() scale factor must be consistent across rows of P (within an epsilon for double values)");
		else scale_factor = sf;
	}
	return sqrt(scale_factor);
}

/*
	Angles
*/

double RadiansToDegrees(double rads) {
	return rads * 180.0 / CV_PI;
}

double AngleRadiansBetweenVectors(Point3d p1, Point3d p2) {
	double dot = p1.ddot(p2);
	double mag1 = veclength(p1);
	double mag2 = veclength(p2);
	dot /= mag1*mag2;
	return acos(dot);
}

double FindAngleDegrees(Point2f pt1, Point2f pt2, Point2f pt3, bool clockwise) {
	double angle1 = RadiansToDegrees(atan2(pt2.y - pt1.y, pt2.x - pt1.x));
	double angle2 = RadiansToDegrees(atan2(pt2.y - pt3.y, pt2.x - pt3.x));
	double angle = angle1 - angle2;
	if (angle < 0) angle += 360;
	else if (angle>360) angle -= 360;
	if (clockwise) angle = 360 - angle;
	return angle;
}


/*
	Display
*/

void display_mat(Mat* img, char* winName)
{
	namedWindow(winName, 1);
	
	int long_limit = GLOBAL_MAX_IMAGE_DISPLAY_SIDE_LENGTH;
	int long_side = std::max(img->cols, img->rows);
	double scale_factor;
	if (long_side > long_limit) scale_factor = long_side / 1024;
	else scale_factor = 1.0;

	if (scale_factor != 1.0) {
		int new_width = img->cols / scale_factor;
		int new_height = img->rows / scale_factor;
		Mat imgResized = cv::Mat::zeros(new_height, new_width, CV_8UC1);
		resize(*img, imgResized, imgResized.size(), 0.0, 0.0, CV_INTER_AREA);
		imshow(winName, imgResized);
	}
	else imshow(winName, *img);
	
	waitKey(0); // image is not drawn until the waitKey() command is given
	destroyWindow(winName);
}

void mark_point_circle_mat(Mat &img, int x, int y, int radius, Scalar color)
{
	if ((x<0) ||
		(y<0) ||
		(x >= img.cols) ||
		(y >= img.rows)) return;
	circle(img, Point((int)(x + 0.5f), (int)(y + 0.5f)), radius, color);
}

void draw_line_mat(Mat &img, Point p1, Point p2, Scalar color)
{
	if ((p1.x<0) ||
		(p1.y<0) ||
		(p1.x >= img.cols) ||
		(p1.y >= img.rows)) return;
	if ((p2.x<0) ||
		(p2.y<0) ||
		(p2.x >= img.cols) ||
		(p2.y >= img.rows)) return;
	line(img, p1, p2, color);
}


/*
	Error checking
*/

// Test for NAN (-1.#IND in Windows) and INF
bool IsFiniteNumber(double x)
{
	return (x <= DBL_MAX && x >= -DBL_MAX);
}


// Parsing
std::vector<double> ParseString_Doubles(std::string s) {
	std::string delimiter = " ";
	double val;
	std::vector<double> vals;
	size_t pos = 0;
	std::string sval;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		sval = s.substr(0, pos);
		val = convert_string_to_double(sval);
		vals.push_back(val);
		s.erase(0, pos + delimiter.length());
	}
	val = convert_string_to_double(s); // get the last entry after the last delimiter
	vals.push_back(val);
	return vals;
}

Mat ParseString_Matrix64F(std::string s, int cols, int rows) {
	std::vector<double> vals = ParseString_Doubles(s);
	Mat m = cv::Mat::zeros(rows, cols, CV_64F);
	int c = 0;
	int r = 0;
	for (std::vector<double>::iterator it = vals.begin(); it != vals.end(); ++it) {
		m.at<double>(r, c) = (*it);
		if (c >= (cols-1)) {
			if (r >= (rows-1)) break;
			else {
				r++;
				c = 0;
			}
		}
		else c++;
	}
	return m;
}

// Conversion

// converts 3x3 OpenCV matrix of type CV_64F to 3x3 eigen matrix of doubles
Matrix3d ConvertOpenCVMatToEigenMatrix3d(Mat *m) {
	assert(m->type() == CV_64F, "ConvertOpenCVMatToEigenMatrix3d() m must have type CV_64F");
	Matrix3d em;

	double *p;
	for (int r = 0; r < m->rows; r++) {
		p = m->ptr<double>(r);
		for (int c = 0; c < m->cols; c++) {
			em(r, c) = p[c];
		}
	}

	return em;
}

// converts 4x4 OpenCV matrix of type CV_64F to 4x4 eigen matrix of doubles
Matrix4d ConvertOpenCVMatToEigenMatrix4d(Mat *m) {
	assert(m->type() == CV_64F, "ConvertOpenCVMatToEigenMatrix3d() m must have type CV_64F");
	Matrix4d em;

	double *p;
	for (int r = 0; r < m->rows; r++) {
		p = m->ptr<double>(r);
		for (int c = 0; c < m->cols; c++) {
			em(r, c) = p[c];
		}
	}

	return em;
}

// converts 3x4 OpenCV matrix of type CV_64F to 3x4 eigen matrix of doubles
Matrix<double, 3, 4> ConvertOpenCVMatToEigenMatrix3x4d(Mat *m) {
	assert(m->type() == CV_64F, "ConvertOpenCVMatToEigenMatrix3d() m must have type CV_64F");
	Matrix<double, 3, 4> em;

	double *p;
	for (int r = 0; r < m->rows; r++) {
		p = m->ptr<double>(r);
		for (int c = 0; c < m->cols; c++) {
			em(r, c) = p[c];
		}
	}

	return em;
}

// converts 4x4 OpenCV camera extrinsics matrix of type CV_64F to 3x4 eigen camera extrinsics matrix of doubles, dropping the last row of [0 0 0 1]
Eigen::Matrix<double, 3, 4> Convert4x4OpenCVExtrinsicsMatTo3x4EigenExtrinsicsMatrixd(cv::Mat *m) {
	assert(m->type() == CV_64F, "ConvertOpenCVMatToEigenMatrix3d() m must have type CV_64F");
	Matrix<double, 3, 4> em;

	double *p;
	for (int r = 0; r < 3; r++) {
		p = m->ptr<double>(r);
		for (int c = 0; c < 4; c++) {
			em(r, c) = p[c];
		}
	}

	return em;
}

// updates Eigen matrices to be of size given by cv_m and updates their values to match BGR channels of cv_m
void ConvertOpenCVBGRMatToEigenMatrices(cv::Mat *cv_m, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_b, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_g, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_r) {
	assert(cv_m->rows == eig_m_b.rows() && cv_m->cols == eig_m_b.cols(), "ConvertOpenCVBGRMatToEigenMatrices() size of eig_m_b does not match cv_m");
	assert(cv_m->rows == eig_m_g.rows() && cv_m->cols == eig_m_g.cols(), "ConvertOpenCVBGRMatToEigenMatrices() size of eig_m_g does not match cv_m");
	assert(cv_m->rows == eig_m_r.rows() && cv_m->cols == eig_m_r.cols(), "ConvertOpenCVBGRMatToEigenMatrices() size of eig_m_b does not match cv_m");
	uchar* pT;
	for (int r = 0; r < cv_m->rows; r++) {
		pT = cv_m->ptr<uchar>(r);
		for (int c = 0; c < cv_m->cols; c++) {
			for (int ch = 0; ch < 3; ch++) {
				eig_m_b(r, c) = pT[c * 3];
				eig_m_g(r, c) = pT[c * 3 + 1];
				eig_m_r(r, c) = pT[c * 3 + 2];
			}
		}
	}
}