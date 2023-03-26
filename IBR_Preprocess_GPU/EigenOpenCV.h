#ifndef EigenOpenCV_H
#define EigenOpenCV_H

// OpenCV
#include "cv.h"
#include "highgui.h"
//#include "opencv2/core/eigen.hpp" // reinstate to use Eigen, but need to get past error at compilation when do so

// cvblob
#include <cvblob.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace cvb;

// functions to help convert between OpenCV and Eigen
class EigenOpenCV {

private:

public:
	
	static void ConvertOpenCVBGRMatToEigenMatrices(Mat *cv_m, Matrix<uchar, Dynamic, Dynamic> &eig_m_b, Matrix<uchar, Dynamic, Dynamic> &eig_m_g, Matrix<uchar, Dynamic, Dynamic> &eig_m_r); // updates Eigen matrices to be of size given by cv_m and updates their values to match BGR channels of cv_m


	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void eigen2cv(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst) {
		if (!(src.Flags & Eigen::RowMajorBit))
		{
			Mat _src(src.cols(), src.rows(), DataType<_Tp>::type,
				(void*)src.data(), src.stride()*sizeof(_Tp));
			transpose(_src, dst);
		}
		else
		{
			Mat _src(src.rows(), src.cols(), DataType<_Tp>::type,
				(void*)src.data(), src.stride()*sizeof(_Tp));
			_src.copyTo(dst);
		}
	}
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void cv2eigen(const Mat& src, Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst) {
		//CV_DbgAssert(src.rows == _rows && src.cols == _cols);
		if (!(dst.Flags & Eigen::RowMajorBit))
		{
			Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
				dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
			if (src.type() == _dst.type())
				transpose(src, _dst);
			else if (src.cols == src.rows)
			{
				src.convertTo(_dst, _dst.type());
				transpose(_dst, _dst);
			}
			else
				Mat(src.t()).convertTo(_dst, _dst.type());
			CV_DbgAssert(_dst.data == (uchar*)dst.data());
		}
		else
		{
			Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
				dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
			src.convertTo(_dst, _dst.type());
			CV_DbgAssert(_dst.data == (uchar*)dst.data());
		}
	}

	template<typename _Tp>
	static inline void cv2eigen(const Mat& src, Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst) {
		dst.resize(src.rows, src.cols);
		if (!(dst.Flags & Eigen::RowMajorBit))
		{
			Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
				dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
			if (src.type() == _dst.type())
				transpose(src, _dst);
			else if (src.cols == src.rows)
			{
				src.convertTo(_dst, _dst.type());
				transpose(_dst, _dst);
			}
			else
				Mat(src.t()).convertTo(_dst, _dst.type());
			CV_DbgAssert(_dst.data == (uchar*)dst.data());
		}
		else
		{
			Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
				dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
			src.convertTo(_dst, _dst.type());
			CV_DbgAssert(_dst.data == (uchar*)dst.data());
		}
	}


	template<typename _Tp>
	static inline void cv2eigen(const Mat& src, Eigen::Matrix<_Tp, Eigen::Dynamic, 1>& dst) {
		CV_Assert(src.cols == 1);
		dst.resize(src.rows);

		if (!(dst.Flags & Eigen::RowMajorBit))
		{
			Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
				dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
			if (src.type() == _dst.type())
				transpose(src, _dst);
			else
				Mat(src.t()).convertTo(_dst, _dst.type());
			CV_DbgAssert(_dst.data == (uchar*)dst.data());
		}
		else
		{
			Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
				dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
			src.convertTo(_dst, _dst.type());
			CV_DbgAssert(_dst.data == (uchar*)dst.data());
		}
	}


	template<typename _Tp>
	static inline void cv2eigen(const Mat& src, Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst) {
		CV_Assert(src.rows == 1);
		dst.resize(src.cols);
		if (!(dst.Flags & Eigen::RowMajorBit))
		{
			Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
				dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
			if (src.type() == _dst.type())
				transpose(src, _dst);
			else
				Mat(src.t()).convertTo(_dst, _dst.type());
			CV_DbgAssert(_dst.data == (uchar*)dst.data());
		}
		else
		{
			Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
				dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
			src.convertTo(_dst, _dst.type());
			CV_DbgAssert(_dst.data == (uchar*)dst.data());
		}
	}

	template<class T>
	static inline void cv2eigenImage(Mat *img, Eigen::Matrix<T, Dynamic, 3> *emat) { // converts CV_8UC3 OpenCV image of size hxw to an Eigen matrix of size h*w x 3 and type T
		assert(img->type() == CV_8UC3);
		int h = img->rows;
		int w = img->cols;
		(*emat) = Matrix<T, Dynamic, 3>(h*w, 3);
		uchar* p;
		int idx;
		for (int r = 0; r < h; r++) {
			p = img->ptr<uchar>(r);
			for (int c = 0; c < w; c++) {
				idx = PixIndexFwdCM(Point(c, r), h);
				(*emat)(idx, 0) = static_cast<T>(p[3 * c + 0]);
				(*emat)(idx, 1) = static_cast<T>(p[3 * c + 1]);
				(*emat)(idx, 2) = static_cast<T>(p[3 * c + 2]);
			}
		}
	};
	
	// imgThresh is a thresholded grayscale image
	static inline Matrix<unsigned int, Dynamic, Dynamic> SegmentUsingBlobs(Mat imgThresh, map<unsigned int, int> &label_counts) {
		bool debug = false;

		label_counts.erase(label_counts.begin(), label_counts.end());

		cvb::CvBlobs blobs; // typedef std::map<CvLabel,CvBlob *> CvBlobs;
		IplImage threshImg = IplImage(imgThresh); // converts header info without copying underlying data
		//IplImage *threshImg = cvCreateImage(cvSize(img->cols, img->rows), 8, 1);

		IplImage* labelImg = cvCreateImage(cvSize(imgThresh.cols, imgThresh.rows), IPL_DEPTH_LABEL, 1);//Image Variable for blobs; IPL_DEPTH_LABEL is 32S (unsigned integer)

		//Finding the blobs
		unsigned int result = cvLabel(&threshImg, labelImg, blobs);

		if (debug) {
			IplImage *frame = cvCreateImage(cvSize(imgThresh.cols, imgThresh.rows), 8, 3);
			//Rendering the blobs
			cvRenderBlobs(labelImg, blobs, frame, frame);
			//Showing the images
			cvNamedWindow("blobs", 1);
			cvShowImage("blobs", frame);
			cvWaitKey(0);
			cvDestroyWindow("blobs");
			cvReleaseImage(&frame);
		}

		// copy labelImage data to an eigen matrix; note: cv2eigen() crashes on use with CV_32S to unsigned int type matrices - reason unknown
		Matrix<unsigned int, Dynamic, Dynamic> seg(imgThresh.rows, imgThresh.cols);
		seg.setZero();
		Mat li = cvarrToMat(labelImg).clone();
		unsigned int *pI;
		unsigned int val;
		vector<unsigned int>::iterator it;
		for (int r = 0; r < li.rows; r++) {
			pI = li.ptr<unsigned int>(r);
			for (int c = 0; c < li.cols; c++) {
				val = pI[c];
				if (val > 100000) val = 0; // weird large numbers showingup
				seg(r, c) = val;

				// add to count for this label
				if (label_counts.find(val) == label_counts.end())
					label_counts[val] = 1;
				else label_counts[val] = label_counts[val] + 1;
			}
		}

		cvReleaseImage(&labelImg);
		blobs.erase(blobs.begin(), blobs.end());

		return seg;
	}
};

#endif