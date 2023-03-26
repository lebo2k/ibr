#include "EigenOpenCV.h"

// updates Eigen matrices to be of size given by cv_m and updates their values to match BGR channels of cv_m
static void ConvertOpenCVBGRMatToEigenMatrices(cv::Mat *cv_m, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_b, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_g, Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> &eig_m_r) {
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