#include "Segmentation.h"
#include "opencv2/gpu/gpu.hpp"

Segmentation::Segmentation() {
}

Segmentation::~Segmentation() {
}

void Segmentation::Init(StereoData *sd) {
	sd_ = sd; // don't delete it later because will be deleted by StereoReconstruction instance that calls this

	num_pixels_out_ = sd_->height_ * sd_->width_;
	num_unknown_pixels_out_ = sd_->num_unknown_pixels_[sd_->cid_out_];

	int h_adj = sd_->height_ - 2 * GLOBAL_PLNSEG_WINDOW;
	int w_adj = sd_->width_ - 2 * GLOBAL_PLNSEG_WINDOW;
	num_pixels_out_adj_ = h_adj * w_adj;

	unknown_mask_out_ = sd_->masks_unknowns_[sd_->cid_out_];
	unknown_mask_out_.resize(sd_->height_, sd_->width_);
	Matrix<bool, Dynamic, Dynamic> mask_unknown_tmp = unknown_mask_out_;
	
	bool *pM = mask_unknown_tmp.data();
	num_unknown_pixels_out_adj_ = num_unknown_pixels_out_;
	for (int c = 0; c < mask_unknown_tmp.cols(); c++) {
		for (int r = 0; r < mask_unknown_tmp.rows(); r++) {
			if ((c < GLOBAL_PLNSEG_WINDOW) ||
				(c >= sd_->width_ - GLOBAL_PLNSEG_WINDOW) ||
				(r < GLOBAL_PLNSEG_WINDOW) ||
				(r >= sd_->height_ - GLOBAL_PLNSEG_WINDOW)) {
				if (*pM) {
					*pM = false;
					num_unknown_pixels_out_adj_--;
				}
			}
			pM++;
		}
	}
	
	mask_unknown_tmp.resize(mask_unknown_tmp.rows()*mask_unknown_tmp.cols(), 1);
	unknown_mask_out_adj_ = mask_unknown_tmp;
}

// segments output texture image imgT_out_ using Felzenszwalb's method
// updates seg_labels with the result
// seg_labels must have width*height rows
// sigma - scalar parameter on smoothing kernel to use prior to segmentation.
// k - scalar parameter on prefered segment size.
// min_size - scalar indicating the minimum number of pixels per segment.
// compress - scalar boolean indicating whether the user wants the segment indices compressed to the range[1 num_segments]. Default: false
void Segmentation::ComputeGBSegmentation(Matrix<unsigned int, Dynamic, 1> *seg_labels, Mat *img, int sigma, int k, int min_size, int height, int width, bool compress) {
	bool debug = false;

	int num_pixels = height * width;
	assert(seg_labels->rows() == num_pixels, "Segmentation::ComputeGBSegmentation() seg_labels must have width*height rows");
	assert(img->rows == height && img->cols == width, "Segmentation::ComputeGBSegmentation() img must have height rows and width columns");

	// Read in the input image
	image<rgb> *input = new image<rgb>(width, height);
	uint8_t *B = (uint8_t *)imPtr(input, 0, 0);
	uchar* pT;
	for (int r = 0; r < height; r++) {
		pT = img->ptr<uchar>(r);
		for (int c = 0; c < width; c++) {
			// switch from BGR to RGB for segmentation to match ojw
			*B++ = pT[3 * c + 2];
			*B++ = pT[3 * c + 1];
			*B++ = pT[3 * c];
		}
	}

	// create the output image
	uint32_t *C = new uint32_t[num_pixels];

	// Segment the image
	int num_sets;
	segment_image(input, sigma, k, min_size, &num_sets, C);
	delete input;

	if (compress) {
		// compress the labelling: changes label values from current values to the same number of values, but contiguous and beginning with 1
		num_sets++; // not sure why it's being incremented, but I don't trust this number because even after being incremented, it's still 1 short; there is a 0 label in C - not sure if that has anything to do with it

		std::map<int, unsigned int> label_map; // map of compressed index => original index
		int num_labels = 0;
		int idx, label;
		bool found;
		for (int c = 0; c < width; c++) {
			for (int r = 0; r < height; r++) {
				idx = PixIndexFwdCM(Point(c, r), height);
				found = false;
				label = 0;
				while ((label < num_labels) &&
					(!found)) { // go through all existing labels
					if (label_map[label] == C[idx]) { // if the label for the current pixel already exists in the list of labels
						C[idx] = (uint32_t)(label + 1); // assign the pixel the current label index, which is incremented to be 1-indexed rather than the zero-indexed map array
						found = true;
					}
					label++;
				}

				if (!found) {
					// create a new label, updating both the value of the curent label index and the number of existing labels
					label = num_labels;
					num_labels++;
					//if (num_labels > num_sets) // if we've exceeded the number of label sets from the segment_image(), then we've assigned too many labels and there was an error
					//	cerr << "Segmentation::ComputeGBSegmentation() there are only " << num_sets << " sets, but we've already exceeded that number with " << num_labels << " labels in the compressed labeling" << endl;

					label_map[label] = C[idx]; // add the label to the compressed label map
					C[idx] = (uint32_t)(label + 1); // assign the pixel the current label index, which is incremented to be 1-indexed rather than the zero-indexed map array
				}
			}
		}
		label_map.erase(label_map.begin(), label_map.end());
	}

	// copy over the data
	for (int i = 0; i < num_pixels; i++) {
		(*seg_labels)(i, 0) = C[i];
	}

	if (debug) DisplayImages::DisplaySegmentedImage(seg_labels, height, width);
}


// segments output texture image imgT_out_ using mean-shift and places results in seg_labels_
// updates seg_labels with the result
// seg_labels must have width*height rows
void Segmentation::ComputeMeanShiftSegmentation(Matrix<unsigned int, Dynamic, 1> *seg_labels, Mat *img, int sigmaS, float sigmaR, int minRegion, int height, int width) {
	bool debug = false;

	assert(seg_labels->rows() == width*height, "Segmentation::ComputeMeanShiftSegmentation() seg_labels must have width*height rows");

	msImageProcessor im_proc;

	// Read in the input image
	uint8_t *temp_im = new uint8_t[height * width * 3];
	uint8_t *B = temp_im;

	uchar* pT;
	for (int r = 0; r < height; r++) {
		pT = img->ptr<uchar>(r);
		for (int c = 0; c < width; c++) {
			//temp_im[r*width + c] = pT[3 * c];
			//temp_im[r*width + c + height*width] = pT[3 * c + 1];
			//temp_im[r*width + c + height*width * 2] = pT[3 * c + 2];

			// switch from BGR to RGB for segmentation to match ojw
			*B++ = (uint8_t)pT[3 * c + 2];
			*B++ = (uint8_t)pT[3 * c + 1];
			*B++ = (uint8_t)pT[3 * c];
		}
	}

	im_proc.DefineImage(temp_im, COLOR, height, width);
	delete[] temp_im;
	if (im_proc.ErrorStatus == EL_ERROR)
		cerr << im_proc.ErrorMessage << endl;

	// Segment the image
	im_proc.Segment(sigmaS, sigmaR, minRegion, HIGH_SPEEDUP);
	if (im_proc.ErrorStatus == EL_ERROR)
		cerr << im_proc.ErrorMessage << endl;

	// Get regions
	int *labels = im_proc.GetLabels();

	// Create the output image
	seg_labels->setZero();
	int idx;
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) { // labels are row-major and seg_labels are column-major, so inner-loop on labels is across while index for seg_labels is computed separately
			idx = PixIndexFwdCM(Point(c, r), height);
			(*seg_labels)(idx, 0) = (unsigned int)(*labels++) + 1; // add 1 to convert from 0-indexed to 1-indexed
		}
	}

	if (debug) DisplayImages::DisplaySegmentedImage(seg_labels, height, width);
}

// compute matrix ePhoto for planar segmentation
// scratch_space is used to increase speed when this function is called in a loop so that it does not need to be reallocated for each call.  Coefficient values for this args is ignored at input and should be ignored at output; it must have height*width rows
// updates X with result and resizes it to height x width from args, with same number of elements as there are columns in arg F
// ephoto = @(F) log(2) - log(exp(sum((F-reshape(double(R), [], sz(3))) .^ 2, 2)*(-1/(options.col_thresh*sz(3))))+1);
void Segmentation::SegmentPlanar_ePhoto(const float f, Eigen::Matrix<float, Dynamic, 3> *scratch, Eigen::Matrix<float, Dynamic, 1> *X) {
	assert(X->rows() == num_unknown_pixels_out_, "Segmentation::SegmentPlanar_ePhoto() arg X must have number of rows equal to args num_unknown_pixels_out_ to match sd_->Aunknowns_[sd_->cid_out_]");
	
	Eigen::Matrix<float, Dynamic, 3> F(X->rows(), 3);
	F.setConstant(f);
	SegmentPlanar_ePhoto(&F, scratch, X);
}

// compute matrix ePhoto for planar segmentation
// F must have height*width rows
// scratch_space is used to increase speed when this function is called in a loop so that it does not need to be reallocated for each call.  Coefficient values for this args is ignored at input and should be ignored at output; it must have height*width rows
// updates X with result and resizes it to height x width from args, with same number of elements as there are columns in arg F
// ephoto = @(F) log(2) - log(exp(sum((F-reshape(double(R), [], sz(3))) .^ 2, 2)*(-1/(options.col_thresh*sz(3))))+1);
void Segmentation::SegmentPlanar_ePhoto(const Eigen::Matrix<float, Dynamic, 3> *F, Eigen::Matrix<float, Dynamic, 3> *scratch, Eigen::Matrix<float, Dynamic, 1> *X) {
	bool timing = false; double t;
	if (timing) t = (double)getTickCount();

	assert(F->rows() == X->rows() && F->rows() == num_unknown_pixels_out_, "Segmentation::SegmentPlanar_ePhoto() args F and X must have number of rows equal to args num_unknown_pixels_out_ to match sd_->Aunknowns_[sd_->cid_out_]");
	
	double col_thresh = GLOBAL_LABELING_ENERGY_COL_THRESHOLD;
	double factor = -1. / (col_thresh*3.);
	double log2 = log(2);

	(*scratch) = (*F) - sd_->Aunknowns_[sd_->cid_out_];
	(*scratch) = scratch->cwiseProduct((*scratch));
	
	(*X) = (*scratch) * VectorXf::Ones(scratch->cols()); // faster method for row-wise summation (rowwise().sum())
	
	(*X) *= factor;
	(*X) = X->array().exp();
	(*X) = X->array() + 1;
	(*X) = X->array().log();
	(*X) = -1 * (*X).array() + log(2);

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Segmentation::SegmentPlanar_ePhoto() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// applies averging filter of size 1x(1+2*window) to Y, then another of size (1+2*window)x1
// edges cases without enough values are ignored, so return matrix size is (height-2*window)*(width-2*window) x 1
void Segmentation::SegmentPlanar_AvgFilter(Eigen::Matrix<float, Dynamic, Dynamic> *Y, Eigen::Matrix<float, Dynamic, Dynamic> *scratch, Eigen::Matrix<float, Dynamic, 1> *Yavg, int height, int width, int window) {
	bool debug = false;

	bool timing = false; double t;
	if (timing) t = (double)getTickCount();

	int hadj = height - 2 * window;
	int wadj = width - 2 * window;
	assert(Y->rows() == height && Y->cols() == width, "Segmentation::SegmentPlanar_AvgFilter() arg Y must have size of height x width");
	assert(scratch->rows() == height && scratch->cols() == width, "Segmentation::SegmentPlanar_AvgFilter() arg scratch must have size of height x width");
	assert(Yavg->rows() == num_pixels_out_adj_, "Segmentation::SegmentPlanar_AvgFilter() arg Yavg must have number of rows equal to num_pixels_out_adj_");

	int idx;
	int wsize = 2 * window + 1;
	float mult = 1. / (float)wsize;

	(*scratch) = (*Y);

	// apply along rows
	Matrix<float, Dynamic, Dynamic> Y_mult = (*Y) * mult;
	float *pS = scratch->data();
	pS += window * height;
	for (int c = 0; c < wadj; c++) { // leave enough room on end for full filter_row; must adjust by +window when assigning it to a pixel in scratch
		for (int r = 0; r < height; r++) {
			*pS = Y_mult.block(r, c, 1, wsize).sum(); // so if window==2, Yavg at (0,0) should correspond to applying a row averaging filter to coefficients of Y centered at (2,2), which includes coefficients from (2,0) through (2,5), which are all in the same row (2) of Y
			pS++;
		}
	}
	Y_mult.resize(0, 0);

	// apply along columns to result from applying along rows
	Matrix<float, Dynamic, Dynamic> scratch_mult = (*scratch) * mult;
	float *pYavg = Yavg->data();
	for (int c = 0; c < wadj; c++) {
		for (int r = 0; r < hadj; r++) {
			*pYavg = scratch_mult.block(r, c + window, wsize, 1).sum(); // so if window==2, Yavg at (0,0) should correspond to applying a column averaging filter to coefficients of Y centered at (2,2), which includes coefficients from (0,2) through (5,2), which are all in the same column (2) of Y
			pYavg++;
		}
	}
	scratch_mult.resize(0, 0);

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Segmentation::SegmentPlanar_AvgFilter() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// like SegmentPlanar_AvgFilter but operates on unknown pixels (and their neighbors) only
// assumes any non-unknown pixel neighbors' values are constant zero because they are perfectly known
// since are unknowns from reference image, can make other assumptions about image, such as dimensions
// applies averging filter of size 1x(1+2*window) to Y, then another of size (1+2*window)x1
// edges cases without enough values are ignored, so return matrix size is (height-2*window)*(width-2*window) x 1
void Segmentation::SegmentPlanar_AvgFilter_Unknowns(Eigen::Matrix<float, Dynamic, 1> *Y, Eigen::Matrix<float, Dynamic, Dynamic> *scratch1, Eigen::Matrix<float, Dynamic, Dynamic> *scratch2, Eigen::Matrix<float, Dynamic, Dynamic> *scratch3, Eigen::Matrix<float, Dynamic, 1> *Yavg, int window) {
	bool debug = false;

	bool timing = false; double t;
	if (timing) t = (double)getTickCount();

	int height = sd_->height_;
	int width = sd_->width_;
	int hadj = sd_->height_ - 2 * window;
	int wadj = sd_->width_ - 2 * window;
	assert(Y->rows() == num_unknown_pixels_out_, "Segmentation::SegmentPlanar_AvgFilter() arg Y must have num_unknown_pixels_out_ rows");
	assert(scratch1->rows() == height && scratch1->cols() == width, "Segmentation::SegmentPlanar_AvgFilter() arg scratch1 must have size of width x height");
	assert(scratch2->rows() == hadj && scratch2->cols() == wadj, "Segmentation::SegmentPlanar_AvgFilter() arg scratch2 must have size of height x width");
	assert(scratch3->rows() == height && scratch3->cols() == width, "Segmentation::SegmentPlanar_AvgFilter() arg scratch2 must have size of height x width");
	assert(Yavg->rows() == num_unknown_pixels_out_adj_, "Segmentation::SegmentPlanar_AvgFilter() arg Yavg must have number of rows equal to num_unknown_pixels_out_adj_");

	int idx;
	int wsize = 2 * window + 1;
	float mult = 1. / (float)wsize;
	Mat kernel_row = Mat::ones(wsize, 1, CV_32F) / (float)(mult);
	Mat kernel_col = Mat::ones(1, wsize, CV_32F) / (float)(mult);

	scratch1->setZero();
	EigenMatlab::AssignByTruncatedBooleans(scratch1, &unknown_mask_out_, Y);

	Mat Ycv = cv::Mat(height, width, CV_32F);
	EigenOpenCV::eigen2cv(*scratch1, Ycv);
	//gpu::GpuMat Ycv_gpu;
	//Ycv_gpu.upload(Ycv);

	/// Apply filter
	//gpu::filter2D(Ycv_gpu, Ycv_gpu, -1, kernel_row, Point(-1, -1));
	//gpu::filter2D(Ycv_gpu, Ycv_gpu, -1, kernel_col, Point(-1, -1));

	filter2D(Ycv, Ycv, -1, kernel_row, Point(-1, -1), 0);
	filter2D(Ycv, Ycv, -1, kernel_col, Point(-1, -1), 0);
	
	//Ycv_gpu.download(Ycv);
	EigenOpenCV::cv2eigen(Ycv, *scratch1);
	Yavg->setZero();
	EigenMatlab::AssignByBooleansOfVals(Yavg, &unknown_mask_out_adj_, scratch1);


	/*
	// update scratch1 and scratch2 to be full images of 0s except in unknown pixels, which receive photoconsistency energy values from Y
	scratch2->setZero();
	EigenMatlab::AssignByTruncatedBooleans(scratch2, &unknown_mask_out_, Y);
	(*scratch3) = (*scratch2);
	(*scratch1) = scratch2->transpose();

	// apply along rows
	(*scratch1) *= mult;
	float *pS2 = scratch2->data();
	pS2 += window * height;
	unsigned int *pI = sd_->Iunknowns_out_.data(); // for pixels with unknown depths, column-major indices into full image for compact unknown pixel representation (compact number of them of full-image pixel indices)
	unsigned int idx_full;
	Point pt;
	for (int i = 0; i < num_unknown_pixels_out_; i++) {
		idx_full = *pI++;
		pt = PixIndexBwdCM(idx_full, sd_->height_);
		if ((pt.y < window) ||
			(pt.y >= width - window))
			continue; // outside possible averaging bounds
		(*scratch2)(pt.y, pt.x) = scratch1->block(pt.y - window, pt.x, wsize, 1).sum(); // so if window==2, Yavg at (0,0) should correspond to applying a row averaging filter to coefficients of Y centered at (2,2), which includes coefficients from (2,0) through (2,5), which are all in the same row (2) of Y; summing down columns for speed by transposing scratch1 first
	}

	// apply along columns to result from applying along rows
	(*scratch2) *= mult;
	pI = sd_->Iunknowns_out_.data();
	for (int i = 0; i < num_unknown_pixels_out_; i++) {
		idx_full = *pI++;
		pt = PixIndexBwdCM(idx_full, sd_->height_);
		if ((pt.y < window) ||
			(pt.y >= height - window) ||
			(pt.x < window) ||
			(pt.x >= width - window)) // no need to compute final answers for x out of averaging bounds (y's were needed in last loop so that they could be used in this computation)
			continue; // outside possible averaging bounds
		(*scratch3)(pt.y, pt.x) = scratch2->block(pt.y-window, pt.x, wsize, 1).sum(); // so if window==2, Yavg at (0,0) should correspond to applying a column averaging filter to coefficients of Y centered at (2,2), which includes coefficients from (0,2) through (5,2), which are all in the same column (2) of Y
	}

	// update Yavg with the results
	EigenMatlab::AssignByBooleansOfVals(Yavg, &unknown_mask_out_adj_, scratch3);
	*/

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Segmentation::SegmentPlanar_AvgFilter() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// computes coefficient-wise minimum between matrices X and min and updates min with the result.  If a coefficient in min was updated by a value from X, min_indices is also updated to have the value index_X in the assocated element position
void Segmentation::SegmentPlanar_CwiseMin(const Eigen::Matrix<float, Dynamic, 1> *X, int index_X, Eigen::Matrix<float, Dynamic, 1> *min, Eigen::Matrix<int, Dynamic, 1> *min_indices) {
	assert(X->rows() == min->rows() && X->rows() == min_indices->rows(), "Segmentation::SegmentPlanar_CwiseMin() arg matrices must have same number of rows");
	
	/*
	Matrix<bool, Dynamic, 1> update = X->array() < min->array();
	Matrix<bool, Dynamic, 1> same = update.array() == false;
	(*min_indices) = min_indices->cwiseProduct(same.cast<int>()) + index_X*update.cast<int>();
	(*min) = min->cwiseMin(*X);
	*/
	
	float xval;
	int curr_idx;
	const float *pX = X->data(); 
	float *pMin = min->data();
	int *pI = min_indices->data();
	for (int r = 0; r < min->rows(); r++) {
		xval = *pX;
		curr_idx = *pI;
		if ((curr_idx < 0) || // disparity indices are >=0, so a negative number denotes the minimum has not yet been set
			(xval < *pMin)) {
			*pMin = xval;
			*pI = index_X;
		}
		pX++; pMin++; pI++;
	}
	
}

// vast majority of time is spent in nested loops
void Segmentation::SegmentPlanar_CorrDisparities(Matrix<float, Dynamic, 1> *corr) {
	assert(corr->rows() == num_unknown_pixels_out_, "Segmentation::SegmentPlanar_Corr() corr must have num_unknown_pixels_out_ rows");
	
	bool debug = true;

	bool timing = true; double t;
	bool timing_loop_disp = true; double t_loop_disp;
	int timing_set_disp_loops = 100;

	if (timing) t = (double)getTickCount();
	
	int h_adj = sd_->height_ - 2 * GLOBAL_PLNSEG_WINDOW;
	int w_adj = sd_->width_ - 2 * GLOBAL_PLNSEG_WINDOW;
	
	Matrix<float, Dynamic, 3> WC(num_unknown_pixels_out_, 3); // data structure containing homogeneous pixel positions across columns (u,v,1)
	WC.col(0) = sd_->Xunknowns_out_.cast<float>();
	WC.col(1) = sd_->Yunknowns_out_.cast<float>();
	WC.col(2).setOnes();

	// track minimum corr values because maximization of normalized values is the same as minimization of non-normalized values
	Eigen::Matrix<float, Dynamic, 1> corr_min(num_unknown_pixels_out_adj_, 1); // minimum corr value for each pixel across all possible disparity values
	corr_min.setZero();
	Eigen::Matrix<int, Dynamic, 1> corr_disp_indices(num_unknown_pixels_out_adj_, 1); // disparity index of value used in corr_min for the associated pixel
	corr_disp_indices.setConstant(-1); // disparity indices are >=0, so set a negative number to denote the minimum has not yet been set
	
	// allocate matrix memory blocks for computation during nested loops
	Matrix<float, 3, 1> d;
	Matrix<float, Dynamic, 1> Z(num_unknown_pixels_out_, 1);
	Matrix<float, Dynamic, 1> oX(num_unknown_pixels_out_, 1);
	Matrix<float, Dynamic, 1> oY(num_unknown_pixels_out_, 1);
	Matrix<float, Dynamic, 3> Ainterp(num_unknown_pixels_out_, 3);
	Matrix<float, Dynamic, 3> scratch_SegmentPlanar_ePhoto(num_unknown_pixels_out_, 3); // scratch space arg for SegmentPlanar_ePhoto()
	Matrix<float, Dynamic, 1> Y(num_unknown_pixels_out_, 1);
	Matrix<float, Dynamic, Dynamic> scratch1_SegmentPlanar_AvgFilter(sd_->height_, sd_->width_);
	Matrix<float, Dynamic, Dynamic> scratch2_SegmentPlanar_AvgFilter(sd_->height_ - 2 * GLOBAL_PLNSEG_WINDOW, sd_->width_ - 2 * GLOBAL_PLNSEG_WINDOW);
	Matrix<float, Dynamic, Dynamic> scratch3_SegmentPlanar_AvgFilter(sd_->height_, sd_->width_);
	Matrix<float, Dynamic, 1> Yavg(num_unknown_pixels_out_adj_, 1);
	Matrix<float, Dynamic, 1> Ysum(num_unknown_pixels_out_adj_, 1);
	
	// set up camera-specific matrices for use in the following loop
	std::map<int, Eigen::Matrix<float, Dynamic, 3>> Xs; // map of camera ID => matrix X of size (num_pixels, 3); homogeneous screen space coordinates and projected, sort of
	std::map<int, Eigen::Matrix<float, 3, 1>> P4s; // map of camera ID => matrix P4 (last column of P for the camera, saved separately
	for (std::map<int, Mat>::iterator it = sd_->imgsT_.begin(); it != sd_->imgsT_.end(); ++it) {
		int cid = (*it).first;
		if (std::find(sd_->use_cids_.begin(), sd_->use_cids_.end(), cid) == sd_->use_cids_.end()) continue;
		//if (cid == sd_->cid_out_) continue; // notes in ojw_segpln.m say the images list used here excludes the reference image, but his code doesn't actually exclude it

		// project the points
		Eigen::Matrix<float, Dynamic, 3> X(num_unknown_pixels_out_, 3);
		X = WC * sd_->Ps_[cid].block<3, 3>(0, 0).transpose(); //  take homogeneous screen space coordinates and project them, sort of; num_pixelsx3 result
		Xs[cid] = X;
		Eigen::Matrix<float, 3, 1> P4;
		P4 = sd_->Ps_[cid].block<3, 1>(0, 3); // save last column of P separately
		P4s[cid] = P4;
	}
	
	if (timing_loop_disp) t_loop_disp = (double)getTickCount();
	for (int b = 0; b < sd_->disps_.size(); b++) {
		Ysum.setZero();

		for (std::map<int, Mat>::iterator it = sd_->imgsT_.begin(); it != sd_->imgsT_.end(); ++it) {

			int cid = (*it).first;
			if (std::find(sd_->use_cids_.begin(), sd_->use_cids_.end(), cid) == sd_->use_cids_.end()) continue;
			//if (cid == sd_->cid_out_) continue; // notes in ojw_segpln.m say the images list used here excludes the reference image, but his code doesn't actually exclude it

			// Vary image coordinates according to disparity
			d = sd_->disps_(b) * P4s[cid];

			Z = Xs[cid].block(0, 2, Xs[cid].rows(), 1).array() + d(2);

			Z = Z.array().inverse().matrix(); // array class inverse function is a fractional inverse, while matrix class version is a linear algebra inverse; can convert back and forth between Array and Matrix classes with .array() and .matrix(), and can also assign an array to a matrix with '=' and it will automatically convert

			oX = Xs[cid].block(0, 0, Xs[cid].rows(), 1).array() + d(0); // horizontal offsets
			oX = oX.cwiseProduct(Z);

			oY = Xs[cid].block(0, 1, Xs[cid].rows(), 1).array() + d(1); // vertical offsets
			oY = oY.cwiseProduct(Z);

			Interpolation::Interpolate<float>(sd_->width_, sd_->height_, &sd_->As_[cid], &oX, &oY, &sd_->masks_[cid], -1000, &Ainterp); // interpolate reprojected floating point image coordinates in input image cid (reprojected from reference image) to compute interpolate cid input image colors of surrounding pixels

			//Y = squeeze(Rinterp);
			
			SegmentPlanar_ePhoto(&Ainterp, &scratch_SegmentPlanar_ePhoto, &Y); // compute photoconsistency energies of reprojected, interpolated pixel colors based on how they differ from the actual pixel colors of the reference image; the pixels colors in Ainterp are ordered as per in the reference image, but carry the colors from input image cid, so a good reprojection should result in a matching or close to matching pixel color when compared with the reference pixel in the same position in the matrix
			
			SegmentPlanar_AvgFilter_Unknowns(&Y, &scratch1_SegmentPlanar_AvgFilter, &scratch2_SegmentPlanar_AvgFilter, &scratch3_SegmentPlanar_AvgFilter, &Yavg, GLOBAL_PLNSEG_WINDOW); // apply an averaging filter to the photoconsistency energies

			//if (debug) DebugPrintMatrix(&Yavg_unk, "Yavg_unk");

			Ysum += Yavg; // sum across cameras
		}

		SegmentPlanar_CwiseMin(&Ysum, b, &corr_min, &corr_disp_indices); // take the minimum across disparities - retain both the minimum and the index of the disparity to which it belongs

		if ((timing_loop_disp) &&
			(b%timing_set_disp_loops == 0) &&
			(b != 0)) {
			t_loop_disp = (double)getTickCount() - t_loop_disp;
			cout << "Segmentation::SegmentPlanar_Corr() loop time for set of " << timing_set_disp_loops << " loops ending at disparity " << b << " = " << t_loop_disp*1000. / getTickFrequency() << " ms (of " << sd_->disps_.size() << " disparities)" << endl;
			t_loop_disp = (double)getTickCount(); // reset timer for next batch
		}
	}
	
	// Normalize and extract highest scoring matches(winner takes all)
	SegmentPlanar_ePhoto(-1000., &scratch_SegmentPlanar_ePhoto, &Y);
	int num_input_images = sd_->use_cids_.size();
	Y *= (float)num_input_images;
	float x = Y(0, 0);
	Eigen::Matrix<float, Dynamic, 1> corr_max(num_unknown_pixels_out_adj_, 1); // minimums become maximums through normalization
	corr_max = corr_min;
	corr_max *= -1.;
	corr_max = corr_max.array() + x;
	corr_max *= (1 / x);
	
	if (debug) DebugPrintMatrix(&corr_max, "corr_max");

	// free memory
	corr_min.resize(0, 1);
	Z.resize(0, 1);
	oX.resize(0, 1);
	oY.resize(0, 1);
	//scratch_SegmentPlanar_ePhoto.resize(0, 3);
	Yavg.resize(0, 1);
	Ysum.resize(0, 1);
	Ainterp.resize(0, 3);
	Y.resize(0, 1);
	scratch1_SegmentPlanar_AvgFilter.resize(0, 0);
	scratch2_SegmentPlanar_AvgFilter.resize(0, 0);
	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = Xs.begin(); it != Xs.end(); it++) {
		(*it).second.resize(0, 3);
	}
	Xs.erase(Xs.begin(), Xs.end());

	// create corr_adj using disparities and corr_max values
	Eigen::Matrix<float, Dynamic, Dynamic> corr_adj(num_unknown_pixels_out_adj_, 1);
	Eigen::Matrix<float, Dynamic, 1> zeros(num_unknown_pixels_out_adj_, 1);
	zeros.setZero(num_unknown_pixels_out_adj_, 1);
	int idx_disp;
	for (int r = 0; r < num_unknown_pixels_out_adj_; r++) {
		idx_disp = corr_disp_indices(r, 0);
		if (idx_disp < 0) idx_disp = 0; // check for unassigned corr values - should not be the case, actually
		corr_adj(r, 0) = sd_->disps_(idx_disp, 0);
	}
	corr_adj = (corr_max.array() < 0.07).select(zeros, corr_adj); // sets corr_adj coefficients to zero for positions where corr_max coefficients are < 0.07

	if (debug) DebugPrintMatrix(&corr_adj, "corr_adj");

	// Return to original size by 1. transforming adjusted compact pixel size to adjusted full image size, padding 2 elements left, right, top, and bottom symmetrically (mirroring neighboring coefficients), then returning to compact unknown pixel representation again
	Matrix<float, Dynamic, Dynamic> corr_full = ExpandUnknownAdjToFullAdjSize(&corr_adj); // ExpandUnknownAdjToFullAdjSize returns full size with only adj coefficients filled in with values from the argument
	corr_full.resize(sd_->height_, sd_->width_);
	corr_full.row(0) = corr_full.row(3);
	corr_full.row(1) = corr_full.row(2);
	corr_full.row(sd_->height_ - 1) = corr_full.row(sd_->height_ - 4);
	corr_full.row(sd_->height_ - 2) = corr_full.row(sd_->height_ - 3);
	corr_full.col(0) = corr_full.col(3);
	corr_full.col(1) = corr_full.col(2);
	corr_full.col(sd_->width_ - 1) = corr_full.col(sd_->width_ - 4);
	corr_full.col(sd_->width_ - 2) = corr_full.col(sd_->width_ - 3);
	corr_full.resize(num_pixels_out_, 1);

	if (debug) DebugPrintMatrix(&corr_full, "corr_full");

	(*corr) = sd_->ContractFulltoUsedSize(&corr_full);

	if (debug) DebugPrintMatrix(corr, "corr");
	
	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Segmentation::SegmentPlanar_Corr() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// generates image segmentations and updates seg_labels_
// note: if R (reference image) is grayscale, would need to replicate it across color channels here as in: R = repmat(R, [1 1 3]);
// timing: currently takes 166 seconds in debug mode
void Segmentation::SegmentPlanar_GenSegs() {
	bool debug = true;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	
	Eigen::Matrix<float, 4, 1> segment_params;
	segment_params << 1, 1.5, 10, 100;
	Eigen::Matrix<int, 14, 1> mults;
	mults << 1, 2, 3, 4, 5, 6, 7, 3, 5, 8, 12, 24, 50, 100;
	int nMaps = mults.size();
	Eigen::Matrix<float, 4, 1> sp;
	seg_labels_.resize(num_unknown_pixels_out_, nMaps);
	Eigen::Matrix<unsigned int, Dynamic, 1> sl_tmp(num_pixels_out_, 1);
	Eigen::Matrix<unsigned int, Dynamic, 1> sl_tmp2(num_unknown_pixels_out_, 1);
	Mat imgT_out_masked = cv::Mat::zeros(sd_->imgsT_[sd_->cid_out_].rows, sd_->imgsT_[sd_->cid_out_].cols, CV_8UC3);
	Matrix<bool, Dynamic, Dynamic> mask = sd_->masks_[sd_->cid_out_];
	mask.resize(sd_->imgsT_[sd_->cid_out_].rows, sd_->imgsT_[sd_->cid_out_].cols);
	StereoData::MaskImg(&sd_->imgsT_[sd_->cid_out_], &mask, &imgT_out_masked);
	mask.resize(0, 0);
	for (int b = 0; b < nMaps; b++) {
		sp = segment_params * (float)mults(b, 0);
		if (b < 8) {// segment the image using mean shift
			if (debug) cout << "Segmentation::SegmentPlanar_GenSegs() computing MS segmentation for b = " << b << endl;
			ComputeMeanShiftSegmentation(&sl_tmp, &imgT_out_masked, sp(0, 0), sp(1, 0), sp(2, 0), sd_->height_, sd_->width_);
		}
		else { // segment the image using Felzenszwalb's method
			if (debug) cout << "Segmentation::SegmentPlanar_GenSegs() computing GB segmentation for b = " << b << endl;
			ComputeGBSegmentation(&sl_tmp, &imgT_out_masked, 0, sp(3, 0), sp(2, 0), sd_->height_, sd_->width_, 1);
		}
		sl_tmp2 = sd_->ContractFulltoUnknownSize(&sl_tmp);
		seg_labels_.block(0, b, seg_labels_.rows(), 1) = sl_tmp2;
	}
	sl_tmp.resize(0, 1);
	sl_tmp2.resize(0, 1);

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Segmentation::SegmentPlanar_GenSegs() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// generates world coordinates for plane fitting
// updates WC with world coordinates for plane fitting; WC is to be a data structure containing homogeneous pixel positions across columns (u,v,1)
// corr must first be set by SegmentPlanar_Corr()
// runs in debug mode in 745 milliseconds for an image of size 936x1404
void Segmentation::SegmentPlanar_WCPlaneFitting(Matrix<float, Dynamic, 1> *corr_disparities, Matrix<float, Dynamic, 3> *WC) {
	bool debug = false;
	
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	assert(WC->rows() == num_unknown_pixels_out_, "Segmentation::SegmentPlanar_WCPlaneFitting() WC must have num_unknown_pixels_out_ rows");
	assert(corr_disparities->rows() == num_unknown_pixels_out_ && corr_disparities->cols() == 1, "Segmentation::SegmentPlanar_Corr() corr must be of size num_unknown_pixels_out_ x 1");

	// calc depths from disparities and address infinite depth cases (set to 0 depth because those will be weeded out later using WC_nonzero_homogeneous boolean map of qualifying points)
	Eigen::Matrix<float, Dynamic, 1> depths(num_unknown_pixels_out_, 1);
	depths = corr_disparities->array().inverse();
	for (int r = 0; r < depths.rows(); r++) {
		if (isinf(depths(r, 0)))
			depths(r, 0) = 0;
	}
	
	// update WC
	WC->setZero();
	WC->block(0, 2, num_unknown_pixels_out_, 1) = depths;
	WC->block(0, 1, num_unknown_pixels_out_, 1) = WC->block(0, 2, num_unknown_pixels_out_, 1).cwiseProduct(sd_->Yunknowns_out_.cast<float>());
	WC->block(0, 0, num_unknown_pixels_out_, 1) = WC->block(0, 2, num_unknown_pixels_out_, 1).cwiseProduct(sd_->Xunknowns_out_.cast<float>());
	
	if (debug) {
		DebugPrintMatrix(WC, "WC");
		Matrix<bool, Dynamic, 1> WCz = WC->block(0, 2, num_unknown_pixels_out_, 1).cwiseAbs().array() < GLOBAL_FLOAT_ERROR;
		cout << "Number of WC points that have homogeneous coefficients equal to 0: " << WCz.count() << endl;
		cin.ignore();
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Segmentation::SegmentPlanar_WCPlaneFitting() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// generates piece-wise planar disparity maps and updates D with them
// WC and WC_nonzero_thirdval must first by updated by SegmentPlanar_WCPlaneFitting()
void Segmentation::SegmentPlanar_GenMaps(Eigen::Matrix<float, Dynamic, 3> *WC) {
	bool debug = true;
	
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	bool timing_mapsize_loop = true; double t_mapsize_loop;
	bool timing_label_loop = true; double t_label_loop;
	int timing_set_label_loops = 100;

	assert(WC->rows() == num_unknown_pixels_out_ && WC->cols() == 3, "Segmentation::SegmentPlanar_GenMaps() WC must have num_unknown_pixels_out_ rows and 3 columns");

	sd_->D_segpln_.resize(0, 0);
	sd_->D_segpln_ = Matrix<float, Dynamic, Dynamic>(sd_->width_*sd_->height_, seg_labels_.cols());
	sd_->D_segpln_.setZero();
	Matrix<float, Dynamic, 1> D_segpln_unk(num_unknown_pixels_out_, 1);
	float rt = 0.1; // 2 * min(abs(diff(Z:))))

	Eigen::Matrix<bool, Dynamic, 1> WC_nonzero_homogeneous = WC->block(0,2,WC->rows(),1).cwiseAbs().array() > GLOBAL_FLOAT_ERROR;

	// set up variables for nested loops
	Eigen::Matrix<float, Dynamic, 3> N(num_unknown_pixels_out_, 3);
	Eigen::Matrix<bool, Dynamic, 1> M(num_unknown_pixels_out_, 1);
	Eigen::Matrix<bool, Dynamic, 1> M2(num_unknown_pixels_out_, 1);
	Eigen::Matrix<float, Dynamic, 1> scratch_dist_RPlane(num_unknown_pixels_out_, 1);
	Eigen::Matrix<bool, Dynamic, 1> scratch_v_RPlane(num_unknown_pixels_out_, 1);
	Eigen::Matrix<float, Dynamic, 1> scratch_rm_RPlane(num_unknown_pixels_out_, 1);
	Eigen::Matrix<float, Dynamic, 3> scratch_pv_RPlane(num_unknown_pixels_out_, 3);
	Eigen::Matrix<float, Dynamic, Dynamic> rm(num_unknown_pixels_out_, 1);
	Eigen::Matrix<float, Dynamic, Dynamic> Y(num_unknown_pixels_out_, 1);
	Eigen::Matrix<float, Dynamic, Dynamic> X(num_unknown_pixels_out_, 1);
	Eigen::Matrix<float, Dynamic, Dynamic> tmp(num_unknown_pixels_out_, 1);
	Eigen::Matrix<float, 3, 1> N2;
	int points_used, points_inliers; // counts within the range [0, num_pixels] denoting number of points being used at various places in the nested loops ; note this "used" here implies something different from the concept of used pixels being masked-in pixels

	for (int b = 0; b < seg_labels_.cols(); b++) { // go through each map size
		if (timing_mapsize_loop) t_mapsize_loop = (double)getTickCount();
		int max_coeff = seg_labels_.block(0, b, num_unknown_pixels_out_, 1).maxCoeff();

		if (timing_label_loop) t_label_loop = (double)getTickCount();
		for (int a = 1; a <= max_coeff; a++) { // go through each label in seg_labels_ at the current map size b

			// debug testing
			if (debug) {
				int num = 0;
				for (int r = 0; r < seg_labels_.rows(); r++) {
					if (seg_labels_(r, b) == a) num++;
				}
				cout << "found num = " << num << endl;
			}

			// choose a segment
			// M = info.segments(:,:,b) == a;
			M = seg_labels_.block(0, b, num_unknown_pixels_out_, 1).array() == a; // M is matrix of booleans denoting whether map b in that pixel position has segment label a
			
			points_used = M.cast<int>().sum();
			if (debug) cout << "M count before excluding homogeneous 0 values " << points_used << endl;

			if (debug) { // display the points with the current label
				int val;
				for (int r = 0; r < M.rows(); r++) {
					val = M(r, 0);
					if (val>0) cout << "M(" << r << ", " << 0 << "): " << val << endl;
				}
				cin.ignore();
			}
			

			// N becomes set of world coordinates for points with the current segment label, excluding any points that have a homogeneous value of 0
			// N = WC(M,:);
			// N = N(N(:,3)~=0,:);
			M = M.cwiseProduct(WC_nonzero_homogeneous);
			points_used = M.count();
			if (debug) cout << "M count after excluding homogeneous 0 values " << points_used << endl;
			int i = 0;
			for (int r = 0; r < num_unknown_pixels_out_; r++) {
				if (M(r, 0)) {
					N.row(i) = WC->row(r);
					i++;
				}
			}

			if (debug) { // display the points with the current label
				int val;
				for (int r = 0; r < points_used; r++) {
					cout << "N.row(" << r << "): " << N.row(r) << endl;
				}
				cin.ignore();
			}

			points_inliers = points_used; // initialize before possible whittling down using RANSAC
			if (points_used > 3) { // if more than 3 points are in the list, weed out outliers using RANSAC before determining best-fit plane from remaining inliers
				// M_ = rplane(N, rt);
				RPlane(&N, &scratch_dist_RPlane, &scratch_v_RPlane, &scratch_rm_RPlane, &scratch_pv_RPlane, rt, num_unknown_pixels_out_, points_used, &M2); // ransac to weed out outliers; returns inliers - coefficients of returned matrix are integer boolean invalues denoted whether a given index into arg pts denotes an inlier

				if (debug) { // display inlier/outlier booleans for points used
					int val;
					for (int r = 0; r < points_used; r++) {
						val = M2(r, 0);
						cout << "M2(" << r << ", " << 0 << "): " << val << endl;
					}
					cin.ignore();
				}

				// N = N(M_,:);
				points_inliers = M2.block(0, 0, points_used, 1).count(); // reduce points of interest in N to include only points for which the corresponding coefficient of M_ is true
				int i = 0;
				for (int r = 0; r < num_unknown_pixels_out_; r++) {
					if (M2(r, 0)) {
						N.row(i) = N.row(r); // should work because r>=i and both start at 0 and are increasing with the loop, though potentially at different rates
						i++;
					}
				}
			}
			else if (points_used == 0) { // to prevent errors, though shouldn't occur; actually, will occur when using masking
				//cerr << "Segmentation::SegmentPlanar_GenMaps() points_used == 0 for map size " << b << " and label " << a << ".  No work done on proposal this iteration." << endl;
				continue;
			}

			// find least squares plane from inliers
			rm.setConstant(-1.);
			N2 = N.block(0, 0, points_inliers, 3).colPivHouseholderQr().solve(rm.block(0, 0, points_inliers, 1)); // Matlab: N2 = pv \ rm; If A is an m-by-n matrix with m ~= n and B is a column vector with m components, or a matrix with several such columns, then X = A\B is the solution in the least squares sense to the under- or overdetermined system of equations AX = B. The effective rank, k, of A, is determined from the QR decomposition with pivoting (see "Algorithm" for details). A solution X is computed which has at most k nonzero components per column. If k < n, this is usually not the same solution as pinv(A)*B, which is the least squares solution with the smallest norm, ||X||.
			if (debug) {
				cout << endl << "N2 least squares plane: " << N2 << endl << endl;
				cin.ignore();
			}

			// calculate first points_used coefficients for tmp; tmp holds points_used values to be copied into D_segpln_unk; X and Y values for tmp calculations are based on indices into full image for pixels for which M is true, where M only has coefficients representing used pixels
			int j = 0;
			double *pX = sd_->Xunknowns_out_.data();
			double *pY = sd_->Yunknowns_out_.data();
			for (int r = 0; r < num_unknown_pixels_out_; r++) {
				if (M(r, 0)) {
					X(j, 0) = (float)*pX;
					Y(j, 0) = (float)*pY;
					j++;
				}
				pX++;
				pY++;
			}
			X.block(0, 0, points_used, 1) = X.block(0, 0, points_used, 1) * N2(0, 0);
			Y.block(0, 0, points_used, 1) = Y.block(0, 0, points_used, 1) * N2(1, 0);
			tmp.block(0, 0, points_used, 1) = X.block(0, 0, points_used, 1) + Y.block(0, 0, points_used, 1);
			tmp.block(0, 0, points_used, 1) = tmp.block(0, 0, points_used, 1).array() + N2(2, 0);
			tmp.block(0, 0, points_used, 1) = -1 * tmp.block(0, 0, points_used, 1);

			if (debug) {
				cout << endl << "depths in matrix tmp.block(0, 0, points_used, 1) to be applied to current list of segment labels (identified by M) at depth b: " << tmp.block(0, 0, points_used, 1) << endl << endl;
				cin.ignore();
			}

			D_segpln_unk.setZero();
			j = 0;
			for (int r = 0; r < num_unknown_pixels_out_; r++) {
				if (M(r, 0)) {
					D_segpln_unk(r, 0) = tmp(j, 0);
					j++;
				}
			}
			sd_->D_segpln_.block(0, b, sd_->D_segpln_.rows(), 1) += sd_->ExpandUnknownToFullSize(&D_segpln_unk);
			
			if ((timing_label_loop) &&
				(a%timing_set_label_loops == 0)) {
				t_label_loop = (double)getTickCount() - t_label_loop;
				cout << "Segmentation::SegmentPlanar_GenMaps() label loop running time for mapsize index " << b << " for set of " << timing_set_label_loops << " loops ending at label index " << a << " = " << t_label_loop*1000. / getTickFrequency() << " ms (of " << max_coeff << " labels)" << endl;
				t_label_loop = (double)getTickCount(); // reset timer for next batch
			}
		}

		if (debug) {
			Eigen::Matrix<float, Dynamic, 1> dispm = sd_->D_segpln_.block(0, b, sd_->D_segpln_.rows(), 1);
			DisplayImages::DisplayGrayscaleImage(&dispm, sd_->height_, sd_->width_); // can have values outside of range of expected disparities (including negative values), so don't display as a disparity image?
		}

		if (timing_mapsize_loop) {
			t_mapsize_loop = (double)getTickCount() - t_mapsize_loop;
			cout << "Segmentation::SegmentPlanar_GenMaps() mapsize loop running time for mapsize index " << b << " = " << t_mapsize_loop*1000. / getTickFrequency() << " ms (of " << seg_labels_.cols() << "map sizes)" << endl;
		}
	}	

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Segmentation::SegmentPlanar_GenMaps() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// generate piecewise-planar disparity proposals for stereo depth reconstruction
// updates matrix sd_->D_segpln_ and sets it to size sd_->height_*sd_->width_ x nMaps where nMaps == 14 hardcoded
void Segmentation::SegmentPlanar() {
	bool debug = true;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	Eigen::Matrix<float, Dynamic, 1> corr_disparities(num_unknown_pixels_out_, 1);
	SegmentPlanar_CorrDisparities(&corr_disparities);

	SegmentPlanar_GenSegs(); // generates image segmentations and updates seg_labels_

	Eigen::Matrix<float, Dynamic, 3> WC(num_unknown_pixels_out_, 3); // data structure containing homogeneous pixel positions across columns (u,v,1)
	SegmentPlanar_WCPlaneFitting(&corr_disparities, &WC); // generate world coordinates for plane fitting

	SegmentPlanar_GenMaps(&WC); // generate piece-wise planar disparity maps

	// clean up results - shouldn't need these because I weed out earlier
	for (int r = 0; r < sd_->D_segpln_.rows(); r++) {
		for (int c = 0; c < sd_->D_segpln_.cols(); c++) {
			if (isnan(sd_->D_segpln_(r, c))) {
				cerr << "sd_->D_segpln_ nan for (r, c) (" << r << ", " << c << ")" << endl;
				sd_->D_segpln_(r, c) = 1e-100;
			}
		}
	}
	/*
	for (int r = 0; r < corr_disparities.rows(); r++) {
		if (isinf(corr_disparities(r, 0))) {
			cerr << "corr_disparities inf for (r, c) (" << r << ", " << 0 << ")" << endl;
			corr_disparities(r, 0) = 0;
		}
	}
	*/

	// note that ojw resize D to a 3D matrix of sd_->height_ x sd_->width_ x nMaps; skipping that here since Eigen can only represent 2D matrices
	// note that ojw saves corr4 to info.disp as an output to this function; skipping that here for now since not sure how/when info.disp is used by ojw later

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Segmentation::SegmentPlanar() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}

	if (debug) {
		for (int b = 0; b < sd_->D_segpln_.cols(); b++) {
			Eigen::Matrix<float, Dynamic, 1> dispm = sd_->D_segpln_.block(0, b, sd_->D_segpln_.rows(), 1);
			DisplayImages::DisplayGrayscaleImage(&dispm, sd_->height_, sd_->width_);// , sd_->disp_step_, sd_->min_disp_);
		}
	}

}

// updates inls with the result - coefficients of returned matrix are boolean values denoting whether a given index into arg pts denotes an inlier
// enables you to weed out outliers from pts before computing best-fit plane from them
// pts must include at least 3 rows, representing 3 points, each with values for 3 dimensions across columns, because need at least 3 points to define a plane
// scratch_* args are used to increase speed when this function is called in a loop so that they do not need to be reallocated for each call.  Coefficient values for these args are ignored at input and should be ignored at output; each should have the same number of rows as arg pts
// also to save time when calling inside a loop, sets sizes of matrix args to points_passed rows, but only the first points_used rows are used by this function
// note that when passing a pointer to an Eigen::Matrix as an arg, must use Dynamic for both axes or neither if want to use block member function - compiler won't let you otherwise
void Segmentation::RPlane(const Eigen::Matrix<float, Dynamic, 3> *pts, Eigen::Matrix<float, Dynamic, 1> *scratch_dist, Eigen::Matrix<bool, Dynamic, 1> *scratch_v, Eigen::Matrix<float, Dynamic, 1> *scratch_rm, Eigen::Matrix<float, Dynamic, 3> *scratch_pv, float th, int points_passed, int points_used, Eigen::Matrix<bool, Dynamic, 1> *inls) {
	assert(points_used >= 3, "Segmentation::RPlane() plane-fitting requires at least 3 points");
	assert(pts->rows() == points_passed, "Segmentation::RPlane() pts must have points_passed rows");
	assert(scratch_dist->rows() == points_passed, "Segmentation::RPlane() scratch_dist must have points_passed rows");
	assert(scratch_v->rows() == points_passed, "Segmentation::RPlane() scratch_v must have points_passed rows");
	assert(scratch_rm->rows() == points_passed, "Segmentation::RPlane() scratch_rm must have points_passed rows");
	assert(scratch_pv->rows() == points_passed, "Segmentation::RPlane() scratch_pv must have points_passed rows");

	bool debug = false;

	double max_sam = 500.;
	float conf = 0.95;
	int max_i = 3;
	int no_i;
	double no_sam = 0.;
	Eigen::Matrix<float, 3, 1> N;
	Eigen::Matrix<float, 3, 3> sam; // 3 sample points (across rows) with 3 dimensions each (across columns)
	Eigen::Matrix<float, 3, 1> div;
	div.setConstant(-1.);
	// reset inls for computation
	inls->setZero();
	int idx;

	while (no_sam < max_sam) {
		no_sam += 1.;

		if (debug) cout << "no_sam: " << endl << no_sam << endl << endl;

		// select 3 indices at random from the list pts; ensure the same point isn't picked twice
		int rand_pts[3];
		rand_pts[0] = rand() % points_used; // a random number in the range [0,points_used)
		rand_pts[1] = rand_pts[0];
		while (rand_pts[1] == rand_pts[0])
			rand_pts[1] = rand() % points_used; // a random number in the range [0,points_used)
		rand_pts[2] = rand_pts[1];
		while ((rand_pts[2] == rand_pts[1]) ||
			(rand_pts[2] == rand_pts[0]))
			rand_pts[2] = rand() % points_used; // a random number in the range [0,points_used)

		// assign the sample points given by the three random pts indices to sam
		for (int i = 0; i < 3; i++) {
			idx = rand_pts[i];
			sam.row(i) = pts->row(idx);
		}

		N = sam.colPivHouseholderQr().solve(div); // had it at "sam.inverse() * div;" but resulted in crashing when sam was too close to being a singular matrix (very small determinant, which Eigen rounded to 0)...maybe should result in #INF values for N that I'm glossing over here and should reinstate to be wiped in check at end of SegmentPlanar() ? // Matlab: N = sam \ div; If A is a square matrix, A\B is roughly the same as inv(A)*B, except it is computed in a different way. If A is an n-by-n matrix and B is a column vector with n components, or a matrix with several such columns, then X = A\B is the solution to the equation AX = B computed by Gaussian elimination (see "Algorithm" for details). A warning message prints if A is badly scaled or nearly singular.
		/*
		if (debug) {
		cout << "RPlane N: " << endl << N << endl << endl;
		cin.ignore();
		}
		*/
		// compute a distance of all points to a plane
		scratch_dist->block(0, 0, points_used, 1) = pts->block(0, 0, points_used, 3) * N;
		scratch_dist->block(0, 0, points_used, 1) = scratch_dist->block(0, 0, points_used, 1).array() + 1.;
		scratch_dist->block(0, 0, points_used, 1) = scratch_dist->block(0, 0, points_used, 1).cwiseAbs();
		(*scratch_v) = scratch_dist->array() < th;
		no_i = scratch_v->block(0, 0, points_used, 1).count();
		/*
		if (debug) {
		cout << "RPlane no_I: " << endl << no_i << endl << endl;
		cin.ignore();
		}
		*/
		if (max_i < no_i) { // re-estimate plane and inliers
			if (debug) cout << "Rplane re-estimating" << endl;

			scratch_rm->block(0, 0, no_i, 1).setConstant(-1.);

			// create matrix of points from pts where corresponding v value is 1 (not 0)
			int i = 0;
			for (int r = 0; r < points_used; r++) {
				if ((*scratch_v)(r, 0)) {
					scratch_pv->row(i) = pts->row(r);
					i++;
				}
			}
			assert(i == no_i, "Segmentation::RPlane() i should equal no_i here");

			N = scratch_pv->block(0, 0, no_i, 3).colPivHouseholderQr().solve(scratch_rm->block(0, 0, no_i, 1)); // only first no_i rows of pv and rm are being used this iteration; Matlab: N2 = pv \ rm; If A is an m-by-n matrix with m ~= n and B is a column vector with m components, or a matrix with several such columns, then X = A\B is the solution in the least squares sense to the under- or overdetermined system of equations AX = B. The effective rank, k, of A, is determined from the QR decomposition with pivoting (see "Algorithm" for details). A solution X is computed which has at most k nonzero components per column. If k < n, this is usually not the same solution as pinv(A)*B, which is the least squares solution with the smallest norm, ||X||.
			/*
			if (debug) {
			cout << "RPlane N re-estimate: " << endl << N << endl << endl;
			cin.ignore();
			}
			*/
			scratch_dist->block(0, 0, points_used, 1) = pts->block(0, 0, points_used, 3) * N;
			scratch_dist->block(0, 0, points_used, 1) = scratch_dist->block(0, 0, points_used, 1).array() + 1.;
			scratch_dist->block(0, 0, points_used, 1) = scratch_dist->block(0, 0, points_used, 1).cwiseAbs();
			(*scratch_v) = scratch_dist->array() < th;
			if (scratch_v->block(0, 0, points_used, 1).count() > inls->block(0, 0, points_used, 1).count()) {
				(*inls) = (*scratch_v);
				max_i = no_i;
				max_sam = std::min(max_sam, nsamples(inls->block(0, 0, points_used, 1).count(), points_used, 3, conf));
			}
		}
	}
}

double Segmentation::nsamples(int ni, int ptNum, int pf, float conf) {
	double SampleCnt;

	Eigen::Array<double, Dynamic, 1> a(pf, 1);
	Eigen::Array<double, Dynamic, 1> b(pf, 1);
	int j = 0;
	for (int i = ni - pf + 1; i <= ni; i++) {
		a(j, 0) = i;
		j++;
	}
	j = 0;
	for (int i = ptNum - pf + 1; i <= ptNum; i++) {
		b(j, 0) = i;
		j++;
	}
	Eigen::Array<double, Dynamic, 1> c = a * b.inverse();
	double q = c.prod();
	double eps = pow(2, -52); // eps in Matlab returns the distance from 1.0 to the next largest double-precision number, that is eps = 2^(-52)

	if ((1 - q) < eps)
		SampleCnt = 1;
	else
		SampleCnt = log(1 - conf) / log(1 - q);

	if (SampleCnt < 1) SampleCnt = 1;

	return SampleCnt;
}

void Segmentation::MeshGrid(int height, int width, Eigen::Matrix<float, Dynamic, Dynamic> &X, Eigen::Matrix<float, Dynamic, Dynamic> &Y) {
	assert(X.rows() == height && X.cols() == width, "Segmentation::MeshGrid() X must be of size width x height");
	assert(Y.rows() == height && Y.cols() == width, "Segmentation::MeshGrid() Y must be of size width x height");

	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			X(r, c) = (float)c;
			Y(r, c) = (float)r;
		}
	}

}