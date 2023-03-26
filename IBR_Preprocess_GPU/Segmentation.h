#ifndef Segmentation_H
#define Segmentation_H

#include "Globals.h"
#include "Interpolation.h"
#include "EigenMatlab.h"
#include "DisplayImages.h"
#include "StereoData.h"

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// 3rd party libraries
#include "seg_ms/msImageProcessor.h"
#include "seg_gb/image.h"
#include "seg_gb/misc.h"
#include "seg_gb/segment-image.h"

using namespace std;
using namespace Eigen;


// image segmentation
class Segmentation {

private:
	
	Matrix<bool, Dynamic, Dynamic> unknown_mask_out_; // used depth mask for reference camera; true where pixel for which depth must be computed, false otherwise
	Matrix<bool, Dynamic, Dynamic> unknown_mask_out_adj_; // used depth mask for reference camera that's been adjusted to set pixels on image borders for which an averaging filter cannot be computed because they are within GLOBAL_PLNSEG_WINDOW pixels of the edge of the image on any side to false even if they were tru in unknown_mask_out_; true where pixel for which depth must be computed as long as within averaging filter bounds, false otherwise
	int num_pixels_out_; // total number of pixels in reference image, regardless of whether disparities are known or pixel is masked
	int num_pixels_out_adj_; // num_pixels_out_ minus any true pixels falling within GLOBAL_PLNSEG_WINDOW pixels of the edge of the image on any side in used_depth_map_out_
	int num_unknown_pixels_out_; // number of pixels in the reference image for which depth is unknown (excludes both masked-out pixels and pixels for which depth is known to a high degree of confidence)
	int num_unknown_pixels_out_adj_; // num_unknown_pixels_out_ minus any true pixels falling within GLOBAL_PLNSEG_WINDOW pixels of the edge of the image on any side in used_depth_map_out_
	Matrix<unsigned int, Dynamic, Dynamic> seg_labels_; // segmentation labels for imgT_out_; size num_unknown_pixels_out_ x b
	std::map<int, std::vector<unsigned int>> unknown_seg_labels_; // map of map size => vector of segmentation labels from seg_labels_ with column==map size and which apply to at least one unknown pixel; labels are sorted and unique

	void InitToCidOut(int cid_out); // called by SegmentPlanar()

	void SegmentPlanar_ePhoto(const float f, Eigen::Matrix<float, Dynamic, 3> *scratch, Eigen::Matrix<float, Dynamic, 1> *X); // compute matrix ePhoto for planar segmentation
	void SegmentPlanar_ePhoto(const Eigen::Matrix<float, Dynamic, 3> *F, Eigen::Matrix<float, Dynamic, 3> *scratch, Eigen::Matrix<float, Dynamic, 1> *X); // compute matrix ePhoto for planar segmentation
	void SegmentPlanar_AvgFilter(Eigen::Matrix<float, Dynamic, Dynamic> *Y, Eigen::Matrix<float, Dynamic, Dynamic> *scratch, Eigen::Matrix<float, Dynamic, 1> *Yavg, int height, int width, int window); // applies averging filter of size 1x(1+2*window) to Y, then another of size (1+2*window)x1; edges cases without enough values are ignored, so return matrix size is (height-2*window)x(width-2*window)
	void SegmentPlanar_AvgFilter_Unknowns(Eigen::Matrix<float, Dynamic, 1> *Y, Eigen::Matrix<float, Dynamic, Dynamic> *scratch1, Eigen::Matrix<float, Dynamic, Dynamic> *scratch2, Eigen::Matrix<float, Dynamic, Dynamic> *scratch3, Eigen::Matrix<float, Dynamic, 1> *Yavg, int window); // like SegmentPlanar_AvgFilter but operates on unknown pixels (and their neighbors) only; assumes any non-unknown pixel neighbors' values are constant zero because they are perfectly known; since are unknowns from reference image, can make other assumptions about image, such as dimensions
	void SegmentPlanar_CorrDisparities(Matrix<float, Dynamic, 1> *corr);
	void SegmentPlanar_GenSegs(); // generates image segmentations and updates seg_labels_
	void SegmentPlanar_WCPlaneFitting(Matrix<float, Dynamic, 1> *corr_disparities, Matrix<float, Dynamic, 3> *WC); // generates world coordinates for plane fitting; updates WC with world coordinates for plane fitting
	void SegmentPlanar_GenMaps(Eigen::Matrix<float, Dynamic, 3> *WC); // generates piece-wise planar disparity maps and updates sd_->D_segpln_ with them
	void SegmentPlanar_CwiseMin(const Eigen::Matrix<float, Dynamic, 1> *X, int index_X, Eigen::Matrix<float, Dynamic, 1> *min, Eigen::Matrix<int, Dynamic, 1> *min_indices); // computes coefficient-wise minimum between matrices X and min and updates min with the result.  If a coefficient in min was updated by a value from X, min_indices is also updated to have the value index_X in the assocated element position
	
	void DetermineUniqueUnknownSegLabels(); // updates unknown_seg_labels_, which is a map of map size => vector of segmentation labels from seg_labels_ with column==map size and which apply to at least one unknown pixel; labels are sorted and unique; requires that seg_labels_ has already been initialized

	// LO-RANSAC
	double nsamples(int ni, int ptNum, int pf, float conf);
	void RPlane(const Eigen::Matrix<float, Dynamic, 3> *pts, Eigen::Matrix<float, Dynamic, 1> *scratch_dist, Eigen::Matrix<bool, Dynamic, 1> *scratch_v, Eigen::Matrix<float, Dynamic, 1> *scratch_rm, Eigen::Matrix<float, Dynamic, 3> *scratch_pv, float th, int points_passed, int points_used, Eigen::Matrix<bool, Dynamic, 1> *inls); // updates inls with the result - coefficients of returned matrix are boolean values denoted whether a given index into arg pts denotes an inlier; enables you to weed out outliers from pts before computing best-fit plane from them

	// Utilities
	void MeshGrid(int height, int width, Eigen::Matrix<float, Dynamic, Dynamic> &, Eigen::Matrix<float, Dynamic, Dynamic> &Y); // mimics [X Y] = meshgrid(width, height)

	// expands matrix from compact used pixel size adjusted for averaging filter by removing GLOBAL_PLNSEG_WINDOW border rows and columns on every side, to full image size adjusted for averaging filter by setting any coefficients within GLOBAL_PLNSEG_WINDOW border rows and columns on every side to 0
	// returns full size with only adj coefficients filled in with values from the argument
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, Dynamic, _cols, _options, _maxRows, _maxCols> ExpandUnknownAdjToFullAdjSize(const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A) {
		Matrix<bool, Dynamic, Dynamic> mask = unknown_mask_out_adj_;
		mask.resize(mask.rows()*mask.cols(), 1);
		Matrix<bool, Dynamic, _cols> mask_rep = mask.replicate(1, A->cols());
		mask.resize(0, 0);
		Matrix<_Tp, Dynamic, _cols> B(mask_rep.rows(), mask_rep.cols());
		B.setZero();
		EigenMatlab::AssignByTruncatedBooleans(&B, &mask_rep, A);

		return B;
	}

public:

	StereoData *sd_;

	int cid_out_; // ID of reference (output) camera

	// Constructors and destructor
	Segmentation();
	~Segmentation();

	void Init(StereoData *sd);

	static void ComputeMeanShiftSegmentation(Matrix<unsigned int, Dynamic, 1> *seg_labels, Mat *img, int sigmaS, float sigmaR, int minRegion, int height, int width); // segments output texture image imgT_out_ using mean-shift; updates seg_labels with the result
	static void ComputeMeanShiftSegmentation_Grayscale(Matrix<unsigned int, Dynamic, 1> *seg_labels, Mat *img, int sigmaS, float sigmaR, int minRegion, int height, int width); // segments output texture image imgT_out_ using mean-shift; updates seg_labels with the result
	static void ComputeGBSegmentation(Matrix<unsigned int, Dynamic, 1> *seg_labels, Mat *img, int sigma, int k, int min_size, int height, int width, bool compress = false); // segments output texture image imgT_out_ using Felzenszwalb's method; updates seg_labels with the result
	void SegmentPlanar(int cid_out); // generate piecewise-planar disparity proposals for stereo depth reconstruction; updates matrix D_segpln_ and sets it to size height_*width_ x nMaps where nMaps == 14 hardcoded

};

#endif