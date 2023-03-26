#include "HelperFunctions.h"


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

bool approx_equal(float x, float y) { return (abs(x - y) < GLOBAL_FLOAT_ERROR); }
bool approx_equal(float x, double y) { return (abs(x - y) < GLOBAL_FLOAT_ERROR); }
bool approx_equal(double x, float y) { return (abs(x - y) < GLOBAL_FLOAT_ERROR); }
bool approx_equal(double x, double y) { return (abs(x - y) < GLOBAL_FLOAT_ERROR); }

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

std::string convert_chars_to_string(char* charval)
{
	std::string s(charval);
	return s;
}

/*
	IO
*/

void save_image_mat(Mat* img, const char* filename, char* extension)
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

float veclength(Point3f p)
{
	float l = sqrt(pow(p.x, 2) + pow(p.y, 2) + pow(p.z, 2));
	return l;
}

double vecdist(Point3d p1, Point3d p2)
{
	double l = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
	return l;
}

float vecdist(Point3f p1, Point3f p2)
{
	float l = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
	return l;
}

float vecdist(Point2f p1, Point2f p2) {
	float l = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
	return l;
}

void normalize(Point3d &V)
{
	double lenV = veclength(V);
	V.x = V.x / lenV;
	V.y = V.y / lenV;
	V.z = V.z / lenV;
}

void normalize(Point3f &V)
{
	float lenV = veclength(V);
	V.x = V.x / lenV;
	V.y = V.y / lenV;
	V.z = V.z / lenV;
}

/*
	Matrices
*/

// P is a 4x4 extrinsics matrix
// returns the scale factor being applied by the matrix
double ScaleFromExtrinsics(Matrix4d P) {
	double scale_factor, sf;
	for (int r = 0; r < 3; r++) { // last row doesn't contain the scale factor
		sf = powf(P(r,0), 2.) + powf(P(r,1), 2.) + powf(P(r,2), 2.);
		if (r > 0) assert(abs(sf-scale_factor)<0.0001);
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

double AngleDegreesBetweenVectors(Point3d p1, Point3d p2) {
	double rads = AngleRadiansBetweenVectors(p1, p2);
	return RadiansToDegrees(rads);
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

void display_mat(Mat* img, char* winName, GLOBAL_AGI_CAMERA_ORIENTATION orientation)
{
	Mat i = Mat::zeros(img->rows, img->cols, img->type());
	img->copyTo(i);

	switch (orientation) {
	case 8: // must be rotated left (transpose with vertical flip)
		transpose(i, i);
		flip(i, i, 0);
		break;
	case 6: // must be rotated right (transpose with horizontal flip)
		transpose(i, i);
		flip(i, i, 1);
		break;
	case 1: // fine as is - same as default
		break;
	default:
		break;
	}

	namedWindow(winName, WINDOW_AUTOSIZE);
	
	int long_limit = GLOBAL_MAX_IMAGE_DISPLAY_SIDE_LENGTH;
	int long_side = std::max(i.cols, i.rows);
	double scale_factor;
	if (long_side > long_limit) scale_factor = long_side / 1024;
	else scale_factor = 1.0;

	if (scale_factor != 1.0) {
		int new_width = i.cols / scale_factor;
		int new_height = i.rows / scale_factor;
		Mat imgResized = cv::Mat::zeros(new_height, new_width, CV_8UC1);
		resize(i, imgResized, imgResized.size(), 0.0, 0.0, CV_INTER_AREA);
		imshow(winName, imgResized);
	}
	else imshow(winName, i);
	
	waitKey(0); // image is not drawn until the waitKey() command is given
	destroyWindow(winName);
}

void display_mat_existingwindow(Mat* img, string winname, GLOBAL_AGI_CAMERA_ORIENTATION orientation)
{
	Mat i = Mat::zeros(img->rows, img->cols, img->type());
	img->copyTo(i);

	switch (orientation) {
	case 8: // must be rotated left (transpose with vertical flip)
		transpose(i, i);
		flip(i, i, 0);
		break;
	case 6: // must be rotated right (transpose with horizontal flip)
		transpose(i, i);
		flip(i, i, 1);
		break;
	case 1: // fine as is - same as default
		break;
	default:
		break;
	}

	int long_limit = GLOBAL_MAX_IMAGE_DISPLAY_SIDE_LENGTH;
	int long_side = std::max(i.cols, i.rows);
	double scale_factor;
	if (long_side > long_limit) scale_factor = long_side / 1024;
	else scale_factor = 1.0;

	if (scale_factor != 1.0) {
		int new_width = i.cols / scale_factor;
		int new_height = i.rows / scale_factor;
		Mat imgResized = cv::Mat::zeros(new_height, new_width, CV_8UC1);
		resize(i, imgResized, imgResized.size(), 0.0, 0.0, CV_INTER_AREA);
		imshow(winname, imgResized);
	}
	else imshow(winname, i);
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

MatrixXd ParseString_Matrixd(std::string s, int cols, int rows) {
	std::vector<double> vals = ParseString_Doubles(s);
	Matrix<double, Dynamic, Dynamic> m(rows, cols);
	m.setZero();
	int c = 0;
	int r = 0;
	for (std::vector<double>::iterator it = vals.begin(); it != vals.end(); ++it) {
		m(r, c) = (*it);
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



// reads a .ppm image file into a cv::Mat BGR image
cv::Mat readPPM(const char *filename) {
	bool debug = false;

	char buff[16];
	FILE *fp;
	int c, rgb_comp_color;
	//open PPM file for reading
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//read image format
	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	//check the image format
	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	//check for comments
	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n');
		c = getc(fp);
	}

	ungetc(c, fp);
	//read image size information
	int height, width;
	if (fscanf(fp, "%d %d", &width, &height) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color != 255) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
	cvtColor(img, img, CV_BGR2RGB);

	while (fgetc(fp) != '\n');

	//read pixel data from file
	if (fread((void*)img.ptr<Vec3b>(0), 3 * width, height, fp) != height) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	cvtColor(img, img, CV_RGB2BGR);

	if (debug) display_mat(&img, "image from ppm");

	fclose(fp);

	return img;
}

// reads a .pgm image file into a cv::Mat GRAY image
Eigen::MatrixXf readPGM(const char *filename) {
	bool debug = false;

	char buff[16];
	FILE *fp;
	int c, rgb_comp_color;
	//open PPM file for reading
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//read image format
	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	//check the image format
	if (buff[0] != 'P' || buff[1] != '5') {
		fprintf(stderr, "Invalid image format (must be 'P5')\n");
		exit(1);
	}

	//check for comments
	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n');
		c = getc(fp);
	}

	ungetc(c, fp);
	//read image size information
	int height, width;
	if (fscanf(fp, "%d %d", &width, &height) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color != 255) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}
	
	cv::Mat imgcv = cv::Mat::zeros(height, width, CV_8UC1); // read using opencv to match rowmajor order of data, then convert to eigen

	while (fgetc(fp) != '\n');

	//read pixel data from file
	if (fread((void*)imgcv.ptr<uchar>(0), width, height, fp) != height) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	Eigen::Matrix<unsigned char, Dynamic, Dynamic> img_uchar(height, width);
	EigenOpenCV::cv2eigen(imgcv, img_uchar);

	Eigen::MatrixXf img_float = img_uchar.cast<float>();

	if (debug) {
		img_uchar = img_float.cast<uchar>();
		cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
		EigenOpenCV::eigen2cv(img_uchar, img);
		display_mat(&img, "image from ppm");
	}

	fclose(fp);

	return img_float;
}

Mat DetermineEigenTransform(Mat* imgGray, Matrix<bool, Dynamic, 1> *mask) {
	bool debug = false;

	if (debug) display_mat(imgGray, "imgGray");

	Mat ET = cv::Mat(imgGray->size(), CV_64F, Scalar(0));

	int blockSize = 3; // must be odd
	// Find square neighborhood blocksize x blocksize around pixel, bounded by image size
	int sideSize = (blockSize - 1) / 2;

	std::vector<double> evs; // vector of eigenvalues
	double ev; // eigenvalue
	double eigenTransformVal, minEigenTransformVal; // eigen-transform value
	double sum; // temporary variable used for summation
	float dropPerc; // percentage of eigenvalues to drop (floored to neareset lower integer and cannot be below 2; ignored if evs.size()<number to be dropped
	int dropNum; // number of largst eigenvalues to drop from the analysis; should be in the range [2,w/3] where w is the total number of eigenvalues; ignored if evs.size()<number to be dropped
	Point2f blockCenter(blockSize / 2.0F, blockSize / 2.0F); // center point of block - since block is square, it's the same for rows and cols
	Mat rot_mat_30 = getRotationMatrix2D(blockCenter, 30, 1.0);
	Mat rot_mat_60 = getRotationMatrix2D(blockCenter, 60, 1.0);

	int idx_full;
	int height = imgGray->rows;

	// For each pixel in imgGray
	for (int r = sideSize; r < (imgGray->rows - sideSize); r++)
	{
		for (int c = sideSize; c < (imgGray->cols - sideSize); c++)
		{
			idx_full = PixIndexFwdCM(Point(c, r), height);
			if (!(*mask)(idx_full, 0)) continue; // enters infinite loop when all pixels are black - besides, no reason to compute masked-out pixel transforms

			Rect rectNH(c - sideSize, r - sideSize, blockSize, blockSize);

			for (int rot_idx = 0; rot_idx < 3; rot_idx++) // to compute minimum-response eigen-transform
			{
				Mat block_ref(*imgGray, rectNH);
				Mat block = block_ref.clone();

				if (rot_idx == 1) warpAffine(block, block, rot_mat_30, block.size());
				else if (rot_idx == 2) warpAffine(block, block, rot_mat_60, block.size());

				EigenvalueDecomposition *e = new EigenvalueDecomposition(block);
				block.release();
				Mat evsMat = e->eigenvalues();
				//cout << "EVs from EVD code:" << endl;
				for (int i = 0; i < evsMat.cols; i++)
				{
					ev = evsMat.at<double>(0, i);
					evs.push_back(abs(ev));
					//cout << "\t" << ev << endl;
				}
				delete e;
				evsMat.release();

				dropPerc = 0.25;
				dropNum = max(2, (int)floor(evs.size() * dropPerc));
				std::sort(evs.begin(), evs.end(), sort_eigenvalues);

				sum = 0;
				for (std::vector<double>::iterator it = evs.begin() + dropNum; it != evs.end(); ++it)
				{
					sum += *it;
				}
				eigenTransformVal = sum / (evs.size() - dropNum);

				if (rot_idx == 0) minEigenTransformVal = eigenTransformVal;
				else if (eigenTransformVal < minEigenTransformVal)
					minEigenTransformVal = eigenTransformVal;

			}

			ET.at<double>(r, c) = minEigenTransformVal;
			evs.erase(evs.begin(), evs.end());
		}
	}

	// Rescale matrix ET so that highest response is white and display the result
	double minVal, maxVal;
	Point minLoc, maxLoc;
	cv::minMaxLoc(ET, &minVal, &maxVal, &minLoc, &maxLoc);
	double beta = 255.0f / maxVal;
	cv::Mat ETS = ET * beta;
	Mat ETS_uchar = cv::Mat(imgGray->size(), CV_8U, Scalar(0));
	ETS.convertTo(ETS_uchar, CV_8U);
	
	if (debug) display_mat(&ETS_uchar, "ETS_uchar");

	return ETS_uchar;

	/*
	// note: use MeanShift for segmenting the result
	Mat ETS_ucharC3 = cv::Mat(imgGray->size(), CV_8UC3, Scalar(0));
	cvtColor(ETS_uchar, ETS_ucharC3, CV_GRAY2BGR);
	pyrMeanShiftFiltering(ETS_ucharC3, ETS_ucharC3, 20, 40, 1);
	display_mat(&ETS_ucharC3, "ETS_ucharC3 mean-shifted");
	*/
}

// returns map of label => count of coefficients
map<unsigned int, int> GetLabelCounts(Matrix<unsigned int, Dynamic, Dynamic> *seg) {
	// find all segment labels
	unsigned int label;
	map<unsigned int, int> label_counts;
	unsigned int *pS = seg->data();
	for (int c = 0; c < seg->cols(); c++) {
		for (int r = 0; r < seg->rows(); r++) {
			label = *pS++;
			if (label_counts.find(label) == label_counts.end())
				label_counts[label] = 1;
			else label_counts[label] = label_counts[label] + 1;
		}
	}

	return label_counts;
}

// return the size of a file in a 64-bit integer
__int64 FileSize(std::wstring name) {
	__stat64 buf;
	if (_wstat64(name.c_str(), &buf) != 0)
		return -1; // error, could use errno to find out more

	return buf.st_size;
}

void StringToWString(std::wstring &ws, const std::string &s) {
	std::wstring wsTmp(s.begin(), s.end());
	ws = wsTmp;
}