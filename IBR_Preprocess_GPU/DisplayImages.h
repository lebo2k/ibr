#ifndef DisplayImages_H
#define DisplayImages_H

#include "Globals.h"
#include "DepthMap.h"

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;


// image segmentation
class DisplayImages {

private:

public:

	template<int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static void DisplaySegmentedImage(Matrix<unsigned int, _rows, _cols, _options, _maxRows, _maxCols> *labels, int height, int width, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL, std::string winname = "") {
		assert(labels->rows()*labels->cols() == height*width);

		Mat img = Mat(height, width, CV_8UC3, Scalar(0));

		unsigned int val;
		Vec3b color;
		uchar blue, green, red;
		std::map<unsigned int, Vec3b> label_colors;
		unsigned int *pL = labels->data();
		for (int c = 0; c < width; c++) {
			for (int r = 0; r < height; r++) {		
				val = *pL++;
				if (label_colors.find(val) == label_colors.end()) {
					blue = rand() % 255;
					green = rand() % 255;
					red = rand() % 255;
					color = Vec3b(blue, green, red);
					label_colors[val] = color;
				}
			}
		}
		
		pL = labels->data();
		for (int c = 0; c < width; c++) {
			for (int r = 0; r < height; r++) {
				val = *pL++;
				img.at<Vec3b>(r,c) = label_colors[val];
			}
		}

		if (winname.compare("") == 0)
			display_mat(&img, "Segmentation Labels", orientation);
		else {
			display_mat_existingwindow(&img, winname, orientation);
			waitKey(1);
		}
	}

	// total number of coefficients in disparity_label_image must equal height*width
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static void DisplayDisparityLabelImage(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *disparity_label_image, int height, int width, float disp_step, float min_disp, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL, std::string winname = "") {
		assert(disparity_label_image->rows()*disparity_label_image->cols() == height*width);
		Eigen::Matrix<float, Dynamic, Dynamic> disparity_image(disparity_label_image->rows(), disparity_label_image->cols());
		disparity_image = disparity_label_image->cast<float>();
		disparity_image = disparity_image * disp_step;
		disparity_image = disparity_image.array() + min_disp;
		DisplayDisparityImage(&disparity_image, height, width, orientation, winname);
	}

	// total number of coefficients in img must equal height*width
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void DisplayDisparityImage(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *img, int height, int width, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL, std::string winname = "") {
		assert(img->rows()*img->cols() == height*width);

		Eigen::Matrix<float, Dynamic, Dynamic> depth_image = img->cast<float>();
		depth_image.resize(height, width);
		depth_image = depth_image.array().inverse();
		
		// ensure no div by 0 issues
		for (int c = 0; c < depth_image.cols(); c++) {
			for (int r = 0; r < depth_image.rows(); r++) {
				if (isinf(depth_image(r, c)))
					depth_image(r, c) = 0;
			}
		}
		
		DepthMap::DisplayDepthImage(&depth_image, orientation, winname);
	}

	// calls DisplayGrayscaleImage(), but first takes imgTrunc, which includes pixel information for output image pixels given by mask (either in compact form or in full form but with no valid data in the non-used positions), and creates a full output image with black pixels for masked-out locations
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2>
	static inline void DisplayGrayscaleImageTruncated(const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *imgTrunc, Matrix<bool, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *mask, int height, int width, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL, std::string winname = "") {
		Matrix<_Tp, _rows2, _cols2, _options2, _maxRows2, _maxCols2> img(mask->rows(), mask->cols());
		img.setZero();
		if ((img.rows() == imgTrunc->rows()) &&
			(img.cols() == imgTrunc->cols()))
			EigenMatlab::AssignByBooleans(&img, mask, imgTrunc);
		else
			EigenMatlab::AssignByTruncatedBooleans(&img, mask, imgTrunc);
		DisplayGrayscaleImage(&img, height, width, orientation, winname);
	}

	// displays a single-channel image in grayscale, interpolating [0,255] uchar grayscale value between [min_val, max_val] of image; outputs image of size height x width
	// assumes img is column-major while outputs a row-major cv::Mat of type CU_8UC1
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void DisplayGrayscaleImage(const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *img, int height, int width, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL, std::string winname = "") {
		assert(img->rows()*img->cols() == height*width);

		// find max and min values in image
		_Tp max_val = (*img)(0, 0);
		_Tp min_val = (*img)(0, 0);
		_Tp val;
		for (int r = 0; r < img->rows(); r++) {
			for (int c = 0; c < img->cols(); c++) {
				val = (*img)(r, c);
				if (val > max_val)
					max_val = val;
				if (val < min_val)
					min_val = val;
			}
		}

		//cout << "min_val " << min_val << ", max_val " << max_val << endl;

		Mat imgGray = Mat(height, width, CV_8UC1, Scalar(0));

		if (max_val != min_val) {
			_Tp rangeVals = max_val - min_val;

			float scaledVal, grayVal;
			float full_dval = 255.;
			int grayValRounded;
			int idx_img;
			for (int r = 0; r < height; r++) {
				for (int c = 0; c < width; c++) {
					idx_img = c * height + r;
					val = (*img)(idx_img);
					if (val > 0) {
						scaledVal = (static_cast<float>(val)-static_cast<float>(min_val)) / static_cast<float>(rangeVals);
						grayVal = scaledVal * full_dval;
						grayValRounded = round(grayVal);
						if (grayValRounded < 0) grayValRounded = 0.;
						if (grayValRounded >(int)full_dval) grayValRounded = (int)full_dval;
						imgGray.at<uchar>(r, c) = (uchar)grayValRounded;
					}
				}
			}
		}

		if (winname.compare("") == 0)
			display_mat(&imgGray, "Grayscale Float Image", orientation);
		else {
			display_mat_existingwindow(&imgGray, winname, orientation);
			waitKey(1);
		}
	}


	// saves a single-channel image in grayscale, interpolating [0,255] uchar grayscale value between [min_val, max_val] of image; outputs image of size height x width
	// assumes img is column-major while outputs a row-major cv::Mat of type CU_8UC1
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void SaveGrayscaleImage(const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *img, int height, int width, std::string savename) {
		assert(img->rows()*img->cols() == height*width);

		// find max and min values in image
		_Tp max_val = (*img)(0, 0);
		_Tp min_val = (*img)(0, 0);
		_Tp val;
		for (int r = 0; r < img->rows(); r++) {
			for (int c = 0; c < img->cols(); c++) {
				val = (*img)(r, c);
				if (val > max_val)
					max_val = val;
				if (val < min_val)
					min_val = val;
			}
		}

		Mat imgGray = Mat(height, width, CV_8UC1, Scalar(0));

		if (max_val != min_val) {
			_Tp rangeVals = max_val - min_val;

			float scaledVal, grayVal;
			float full_dval = 255.;
			int grayValRounded;
			int idx_img;
			for (int r = 0; r < height; r++) {
				for (int c = 0; c < width; c++) {
					idx_img = c * height + r;
					val = (*img)(idx_img);
					if (val > 0) {
						scaledVal = (val - min_val) / rangeVals;
						grayVal = scaledVal * full_dval;
						grayValRounded = round(grayVal);
						if (grayValRounded < 0) grayValRounded = 0.;
						if (grayValRounded >(int)full_dval) grayValRounded = (int)full_dval;
						imgGray.at<uchar>(r, c) = (uchar)grayValRounded;
					}
				}
			}
		}

		save_image_mat(&imgGray, savename.c_str(), "jpg");
	}

	// displays a 3-channel image in BGR, interpolating [0,255] uchar values between [min_val, max_val] of image in each channel; outputs image of size height x width
	// assumes img is column-major while outputs a row-major cv::Mat of type CU_8UC1
	template<typename _Tp>
	static inline void DisplayBGRImageScaled(const Matrix<_Tp, Dynamic, 3> *img, int height, int width, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL, std::string winname = "") {
		assert(img->rows() == height*width);

		// find max and min values in image
		_Tp max_val_b = (*img)(0, 0);
		_Tp min_val_b = (*img)(0, 0);
		_Tp max_val_g = (*img)(0, 1);
		_Tp min_val_g = (*img)(0, 1);
		_Tp max_val_r = (*img)(0, 2);
		_Tp min_val_r = (*img)(0, 2);
		_Tp val_b, val_g, val_r;
		for (int r = 0; r < img->rows(); r++) {
			val_b = (*img)(r, 0);
			if (val_b > max_val_b)
				max_val_b = val_b;
			if (val_b < min_val_b)
				min_val_b = val_b;

			val_g = (*img)(r, 1);
			if (val_g > max_val_g)
				max_val_g = val_g;
			if (val_g < min_val_g)
				min_val_g = val_g;

			val_r = (*img)(r, 2);
			if (val_r > max_val_r)
				max_val_r = val_r;
			if (val_r < min_val_r)
				min_val_r = val_r;
		}

		Mat imgcv = Mat(height, width, CV_8UC3, Scalar(0));

		if ((max_val_b != min_val_b) &&
			(max_val_g != min_val_g) && 
			(max_val_r != min_val_r)) {
			_Tp rangeVals_b = max_val_b - min_val_b;
			_Tp rangeVals_g = max_val_g - min_val_g;
			_Tp rangeVals_r = max_val_r - min_val_r;

			float scaledVal_b, scaledVal_g, scaledVal_r;
			float floatVal_b, floatVal_g, floatVal_r;
			float full_dval = 255.;
			int roundVal_b, roundVal_g, roundVal_r;
			int idx_img;
			for (int r = 0; r < img->rows(); r++) {
				val_b = (*img)(r, 0);
				val_g = (*img)(r, 1);
				val_r = (*img)(r, 2);
				if ((val_b > 0) &&
					(val_g > 0) &&
					(val_r > 0)) {
					scaledVal_b = (val_b - min_val_b) / rangeVals_b;
					floatVal_b = scaledVal_b * full_dval;
					roundVal_b = round(floatVal_b);
					if (roundVal_b < 0) roundVal_b = 0.;
					if (roundVal_b >(int)full_dval) roundVal_b = (int)full_dval;

					scaledVal_g = (val_g - min_val_g) / rangeVals_g;
					floatVal_g = scaledVal_g * full_dval;
					roundVal_g = round(floatVal_g);
					if (roundVal_g < 0) roundVal_g = 0.;
					if (roundVal_g >(int)full_dval) roundVal_g = (int)full_dval;

					scaledVal_r = (val_r - min_val_r) / rangeVals_r;
					floatVal_r = scaledVal_r * full_dval;
					roundVal_r = round(floatVal_r);
					if (roundVal_r < 0) roundVal_r = 0.;
					if (roundVal_r >(int)full_dval) roundVal_r = (int)full_dval;

					Point pt = PixIndexBwdCM(r, height);
					imgcv.at<Vec3b>(pt.y, pt.x) = Vec3b(roundVal_b, roundVal_g, roundVal_r);
				}
			}
		}

		if (winname.compare("") == 0)
			display_mat(&imgcv, "Float Image", orientation);
		else {
			display_mat_existingwindow(&imgcv, winname, orientation);
			waitKey(1);
		}
	}

	// displays a 3-channel image in BGR, interpolating [0,255] uchar values between [min_val, max_val] of image in each channel; outputs image of size height x width
	// assumes img is column-major while outputs a row-major cv::Mat of type CU_8UC1
	template<typename _Tp>
	static inline void DisplayBGRImage(const Matrix<_Tp, Dynamic, 3> *img, int height, int width, GLOBAL_AGI_CAMERA_ORIENTATION orientation = AGO_ORIGINAL, std::string winname = "") {
		assert(img->rows() == height*width);

		Mat imgcv = Mat(height, width, CV_8UC3, Scalar(0));

		float val_b, val_g, val_r;
		float full_dval = 255.;
		int roundVal_b, roundVal_g, roundVal_r;
		for (int r = 0; r < img->rows(); r++) {
			val_b = (*img)(r, 0);
			val_g = (*img)(r, 1);
			val_r = (*img)(r, 2);

			roundVal_b = round(val_b);
			if (roundVal_b < 0) roundVal_b = 0.;
			if (roundVal_b >(int)full_dval) roundVal_b = (int)full_dval;

			roundVal_g = round(val_g);
			if (roundVal_g < 0) roundVal_g = 0.;
			if (roundVal_g >(int)full_dval) roundVal_g = (int)full_dval;

			roundVal_r = round(val_r);
			if (roundVal_r < 0) roundVal_r = 0.;
			if (roundVal_r >(int)full_dval) roundVal_r = (int)full_dval;

			Point pt = PixIndexBwdCM(r, height);
			imgcv.at<Vec3b>(pt.y, pt.x) = Vec3b(roundVal_b, roundVal_g, roundVal_r);
		}

		if (winname.compare("") == 0)
			display_mat(&imgcv, "Float Image", orientation);
		else {
			display_mat_existingwindow(&imgcv, winname, orientation);
			waitKey(1);
		}
	}

};

#endif
