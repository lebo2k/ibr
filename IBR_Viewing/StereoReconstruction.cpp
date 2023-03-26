#include "StereoReconstruction.h"

StereoReconstruction::StereoReconstruction() {

	// default energy parameter values
	disp_thresh_ = GLOBAL_LABELING_ENERGY_DISPARITY_THRESHOLD;
	col_thresh_ = GLOBAL_LABELING_ENERGY_COL_THRESHOLD;
	occl_const_ = GLOBAL_LABELING_ENERGY_OCCLUSION_CONSTANT;
	occl_val_ = occl_const_ + log(2);
	lambda_l_ = GLOBAL_LABELING_ENERGY_LAMBDA_L;
	lambda_h_ = GLOBAL_LABELING_ENERGY_LAMBDA_H;

	// parameters for the mean-shift over-segmentation of the reference image
	ms_seg_sigmaS_ = GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAS;
	ms_seg_sigmaR_ = GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAR;
	ms_seg_minRegion_ = GLOBAL_MEAN_SHIFT_SEGMENTATION_MINREGION;

	// default optimization settings
	visibility_ = true;
	compress_graph_ = false;
	max_iters_ = 3000;
	converge_ = 0.01;
	average_over_ = 20;
	independent_ = false;
	window_ = 2.;

	// settings calculations
	if (visibility_) visibility_val_ = 10000.;
	else visibility_val_ = 0;
	max_iters_ = max_iters_ + average_over_ + 1;

}

StereoReconstruction::~StereoReconstruction() {
}

// initialize disp_labels_ data structure and min_disp_, max_disp_, and disp_step_ using args min_disp, max_disp and disp_step
void StereoReconstruction::InitDisps() {
	bool debug = false;

	// extend range by GLOBAL_EXTEND_DEPTH_RANGE front and back
	double min_depth_ext = min_depth_ * (1 - GLOBAL_EXTEND_DEPTH_RANGE);
	double max_depth_ext = max_depth_ * (1 + GLOBAL_EXTEND_DEPTH_RANGE);

	Matrix<double, 4, 2> depth_extremes_WS; // 2 world space points (0,0,min_depth,1) and (0,0,max_depth,1)
	depth_extremes_WS.setZero();
	depth_extremes_WS(2, 0) = min_depth_ext;
	depth_extremes_WS(2, 1) = max_depth_ext;
	depth_extremes_WS(3, 0) = 1.;
	depth_extremes_WS(3, 1) = 1.;

	double disp_vals = 0; // to hold the number of disparity levels; to be set by projecting points (0,0,min_depth,1) and (0,0,max_depth,1) into each reference camera, finding the pixel distance between projected near and far points, and ensuring the minimum spacing between disparity samples is 0.5 pixels in screen space

	for (map<int, Matrix<double, 3, 4>>::iterator it = Ps_.begin(); it != Ps_.end(); ++it) {
		if ((*it).first == cid_out_) continue;

		Matrix<double, 3, 2> depth_extremes_SS = (*it).second * depth_extremes_WS; // project world space points (0,0,min_depth,1) and (0,0,max_depth,1) into screen space of this reference camera
		// divide the screen space points through by the homogeneous coordinates to normalize them
		double h0 = depth_extremes_SS(2, 0);
		depth_extremes_SS.col(0) = depth_extremes_SS.col(0) / h0;
		double h1 = depth_extremes_SS(2, 1);
		depth_extremes_SS.col(1) = depth_extremes_SS.col(1) / h1;
		// find the maximum pixel distance among x and y dimensions for the two projected pixels
		double xdist = abs(depth_extremes_SS(0, 0) - depth_extremes_SS(0, 1));
		double ydist = abs(depth_extremes_SS(1, 0) - depth_extremes_SS(1, 1));
		disp_vals = max(disp_vals, max(xdist, ydist));
	}
	num_disps_ = ceil(disp_vals * 2.); // ensure minimum spacing is 0.5 pixels, so multiply minimum 1 pixel spacing by 2 to get minimum 0.5 pixel spacing and round up to ensure integer number of disparity levels is over the threshold

	// Calculate disparities - disps ends up as range from 1/min_depth to 1/max_depth with disp_vals number of steps, evenly spaced
	disps_ = ArrayXd(num_disps_); // set up disparities from 0 through (num_disps - 1)
	for (int i = 0; i < num_disps_; i++) {
		disps_(i) = (double)i;
	}
	// taking disp_vals to be 10, min depth to be 2m and max to be 8m, take it through the calcs...
	disps_ *= (1. - (min_depth_ext / max_depth_ext)) / (double)(num_disps_ - 1); // disps: 0, 3/36, 6/36, 9/36, ... , 27/36
	disps_ = (1. - disps_) / min_depth_ext; // disps: 1/2, 5/12, 3/8, ... , 1/8
	
	// order disparities from foreground to background => descending
	std::sort(disps_.data(), disps_.data() + disps_.size(), std::greater<double>()); // sorts values in descending order, but are already in that order at this point...uncomment if want to make doubly-sure
	//igl::sort(X, dim, mode, Y, IX); // igl method for sorting values in descending order
	
	max_disp_ = disps_(0);
	min_disp_ = disps_(num_disps_ - 1);
	disp_step_ = disps_(1) - disps_(0);

	if (debug) {
		cout << endl << endl << "StereoReconstruction::InitDisps()" << endl;
		cout << "Depth extended min and max: " << min_depth_ext << ", " << max_depth_ext << endl;
		cout << "Max disparity (closest): " << max_disp_ << endl;
		cout << "Min disparity (farthest): " << min_disp_ << endl;
		cout << "Number of disparities: " << num_disps_ << endl;
		cout << "Disparity step: " << disp_step_ << endl;
	}
}

// returns closest label to given disparity value
// labels are truncated to between 1 and the maximum label number
int StereoReconstruction::DispValueToLabel(double disp_val) {
	int label = round((disp_val - min_disp_) / disp_step_, 0);
	if (label < 1) label = 1;
	else if (label > disps_.size()) label = disps_.size();
	return label;
}

// transforms the reference frame and sets the extrinsics matrices and output extrinsics matrix such that the output camera extrinsics matrix is the identity
// RTins is a map of camera ID => pointer to camera extrinsics matrix
void StereoReconstruction::TransformReferenceFrame() {
	bool debug = false;
	
	Matrix<double, 3, 4> K_ext;
	K_ext.block(0,0,3,3) << K_;
	K_ext.col(3) << 0., 0., 0.;

	Matrix<double, 1, 4> ext;
	ext << 0., 0., 0., 1.;

	Matrix4d RTout_ext;
	RTout_ext << RTs_[cid_out_], ext;

	Matrix<double, 3, 4> Pout;
	Pout = K_ext * RTout_ext;
	Matrix4d Pout_ext;
	Pout_ext << Pout, ext;

	Matrix4d Pout_ext_inv = Pout_ext.inverse();

	for (std::map<int, Matrix<double, 3, 4>>::iterator it = RTs_.begin(); it != RTs_.end(); ++it) {
		int cid = (*it).first;
		Matrix<double, 4, 4> RT_ext;
		RT_ext << (*it).second, ext;
		Ps_[cid] = K_ext * RT_ext * Pout_ext_inv;
	}

	if (debug) {
		cout << endl << endl << "StereoReconstruction::TransformReferenceFrame()" << endl;
		cout << "K_ext" << endl << K_ext << endl;
		cout << "RTout_ext" << endl << RTout_ext << endl;
		cout << "Pout" << endl << Pout << endl;
		cout << "Pout_ext" << endl << Pout_ext << endl;
		cout << "Pout_ext_inv" << endl << Pout_ext_inv << endl;
		cout << "3x4 matrix should be the 3x3 identity matrix with a column of zeros tacked on the right" << endl << Pout * Pout_ext_inv << endl;
		cout << "Ps_:" << endl;
		for (std::map<int, Matrix<double, 3, 4>>::iterator it = Ps_.begin(); it != Ps_.end(); ++it) {
			cout << "Camera " << (*it).first << ": " << (*it).second << endl;
		}
		cin.ignore();
	}
}

// segments output texture image imgT_out_ using mean-shift and places results in seg_labels_
void StereoReconstruction::ComputeMeanShiftSegmentation() {
	bool debug = false;

	msImageProcessor im_proc;

	// Read in the input image
	uint8_t *temp_im = new uint8_t[height_ * width_ * 3];
	uint8_t *B = temp_im;
	uchar* pT;
	for (int r = 0; r < height_; r++) {
		pT = imgsT_[cid_out_].ptr<uchar>(r);
		for (int c = 0; c < width_; c++) {
			temp_im[r*width_ + c] = pT[3 * c];
			temp_im[r*width_ + c + height_*width_] = pT[3 * c + 1];
			temp_im[r*width_ + c + height_*width_ * 2] = pT[3 * c + 2];
		}
	}
	im_proc.DefineImage(temp_im, COLOR, height_, width_);
	delete temp_im;
	if (im_proc.ErrorStatus == EL_ERROR)
		cerr << im_proc.ErrorMessage << endl;

	// Segment the image
	im_proc.Segment(ms_seg_sigmaS_, ms_seg_sigmaR_, ms_seg_minRegion_, HIGH_SPEEDUP);
	if (im_proc.ErrorStatus == EL_ERROR)
		cerr << im_proc.ErrorMessage << endl;

	// Get regions
	int *labels = im_proc.GetLabels();

	// Create the output image
	seg_labels_ = Matrix<unsigned int, Dynamic, Dynamic>(height_, width_);
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++) {
			seg_labels_(r, c) = (uint32_t)(*labels++) + 1;
		}
	}

	if (debug) cout << "Mean-shift segmentation labels by pixel" << endl << seg_labels_ << endl;
}

// initializes data structure EW, which identifies smoothness edges that don't cross segmentation boundaries
void StereoReconstruction::InitEW() {
	bool debug = false;
	
	// create a matrix of integers (not unsigned) EW_step1 of size 3xn where n is the number of columns in SEI; values are found by using the corresponding SEI value as an index into seg_labels_ (where idx = col*rows + row) and retrieving the label value
	Eigen::Matrix<int, 3, Dynamic> EW_step1(3, SEI_.cols());
	for (int i = 0; i < EW_step1.rows(); i++) {
		for (int j = 0; j < EW_step1.cols(); j++) {
			int idx = SEI_(i, j);
			int c = std::floor(idx / imgsT_[cid_out_].rows);
			int r = idx - c * imgsT_[cid_out_].rows;
			EW_step1(i, j) = seg_labels_(r, c);
		}
	}

	if (debug) {
		cout << "EW_step1" << endl;
		for (int i = 0; i < 10; i++) {
			cout << "EW_step1(0, " << i << ") " << EW_step1(0, i) << endl;
			cout << "EW_step1(1, " << i << ") " << EW_step1(1, i) << endl;
			cout << "EW_step1(2, " << i << ") " << EW_step1(2, i) << endl;
		}
		cout << endl << " ... " << endl;
		for (int i = EW_step1.cols() - 10; i < EW_step1.cols(); i++) {
			cout << "EW_step1(0, " << i << ") " << EW_step1(0, i) << endl;
			cout << "EW_step1(1, " << i << ") " << EW_step1(1, i) << endl;
			cout << "EW_step1(2, " << i << ") " << EW_step1(2, i) << endl;
		}
		cin.ignore();
	}

	// create a matrix of integers EW_step2 of size 2xn where n is the number of columns in EW_step1; values for (r,c) are found by taking the difference between values of EW_step1 at the same position and values at the same column but one row higher
	Eigen::Matrix<int, 2, Dynamic> EW_step2(2, EW_step1.cols());
	for (int r = 0; r < EW_step2.rows(); r++) {
		for (int c = 0; c < EW_step2.cols(); c++) {
			EW_step2(r, c) = EW_step1(r + 1, c) - EW_step1(r, c);
		}
	}

	if (debug) {
		cout << "EW_step2" << endl;
		for (int i = 0; i < 10; i++) {
			cout << "EW_step2(0, " << i << ") " << EW_step2(0, i) << endl;
			cout << "EW_step2(1, " << i << ") " << EW_step2(1, i) << endl;
		}
		cout << endl << " ... " << endl;
		for (int i = EW_step2.cols() - 10; i < EW_step2.cols(); i++) {
			cout << "EW_step2(0, " << i << ") " << EW_step2(0, i) << endl;
			cout << "EW_step2(1, " << i << ") " << EW_step2(1, i) << endl;
		}
		cin.ignore();
	}

	// create a matrix of integers EW_step3 of size 1xn where n is the number of columns in EW_step2; values for (c) are found by checking each column c of EW_step2 and setting the EW_step3(0,c) to be 1 if all elements in the column of EW_step2 are 0, and setting it to 0 otherwise
	Eigen::Matrix<int, 1, Dynamic> EW_step3(1, EW_step2.cols());
	for (int c = 0; c < EW_step2.cols(); c++) {
		int a = EW_step2(0, c);
		int b = EW_step2(1, c);
		if ((a == 0) &&
			(b == 0))
			EW_step3(0, c) = 1;
		else EW_step3(0, c) = 0;
	}

	if (debug) {
		cout << "EW_step3" << endl;
		for (int i = 0; i < 10; i++) {
			cout << "EW_step3(0, " << i << ") " << EW_step3(0, i) << endl;
		}
		cout << endl << " ... " << endl;
		for (int i = EW_step3.cols() - 10; i < EW_step3.cols(); i++) {
			cout << "EW_step3(0, " << i << ") " << EW_step3(0, i) << endl;
		}
		cin.ignore();
	}
	
	Eigen::Matrix<int, 1, Dynamic> EW_step4(1, EW_step3.cols());
	//EW_step4 = EW_step3.cast<float>() * lambda_h_ + EW_step3.cwiseEqual(0).cast<float>() *lambda_l_;
	EW_step4 = EW_step3 * lambda_h_ + EW_step3.cwiseEqual(0).cast<int>() *lambda_l_;
	int num_in = imgsT_.size(); // number of images including reference (output) image
	EW_step4 = EW_step4 * num_in;

	if (debug) {
		cout << "EW_step4" << endl;
		for (int i = 0; i < 10; i++) {
			cout << "EW_step4(0, " << i << ") " << EW_step4(0, i) << endl;
		}
		cout << endl << " ... " << endl;
		for (int i = EW_step4.cols() / 2 - 10; i < EW_step4.cols() / 2; i++) {
			cout << "EW_step4(0, " << i << ") " << EW_step4(0, i) << endl;
		}
		cout << endl << " ... " << endl;
		for (int i = EW_step4.cols() - 10; i < EW_step4.cols(); i++) {
			cout << "EW_step4(0, " << i << ") " << EW_step4(0, i) << endl;
		}
		cin.ignore();
	}
	
	// update EW to be a matrix of size EW_step3*8 x 1 where each element is repeated 8 times
	EW_ = Matrix<int, Dynamic, 1>(8 * EW_step4.cols(), 1);
	for (int c = 0; c < EW_step4.cols(); c++) {
		for (int i = 0; i < 8; i++) {
			EW_(8 * c + i, 0) = EW_step4(0, c);
		}
	}

	if (debug) {
		for (int i = 0; i < 10; i++) {
			cout << "EW_(" << i << ", 0) " << EW_(i, 0) << endl;
		}
		cout << endl << " ... " << endl;
		for (int i = EW_.rows() / 2 - 10; i < EW_.rows() / 2; i++) {
			cout << "EW_(" << i << ", 0) " << EW_(i, 0) << endl;
		}
		cout << endl << " ... " << endl;
		for (int i = EW_.rows() - 10; i < EW_.rows(); i++) {
			cout << "EW_(" << i << ", 0) " << EW_(i, 0) << endl;
		}
		cin.ignore();

		cout << "EW_" << endl << EW_ << endl;
	}
}

void StereoReconstruction::Init(std::map<int, Mat*> imgsT, std::map<int, Mat*> imgsD, std::map<int, Matrix<double, 3, 4>> RTs, double min_depth, double max_depth) {
	bool debug = false;

	// copy args locally
	min_depth_ = min_depth;
	max_depth_ = max_depth;
	for (std::map<int, Mat*>::iterator it = imgsT.begin(); it != imgsT.end(); ++it) {
		int cid = (*it).first;
		assert(RTs.find(cid) != RTs.end(), "StereoReconstruction::Stereo() imgsT and RTs must contain the same camera IDs");
		Mat img = cv::Mat::zeros((*it).second->size(), (*it).second->type());
		(*it).second->copyTo(img);
		imgsT_[cid] = img;
	}

	for (std::map<int, Mat*>::iterator it = imgsD.begin(); it != imgsD.end(); ++it) {
		int cid = (*it).first;
		assert(RTs.find(cid) != RTs.end(), "StereoReconstruction::Stereo() imgsD and RTs must contain the same camera IDs");
		Mat img = cv::Mat::zeros((*it).second->size(), (*it).second->type());
		(*it).second->copyTo(img);
		imgsD_[cid] = img;
	}

	for (std::map<int, Matrix<double, 3, 4>>::iterator it = RTs.begin(); it != RTs.end(); ++it) {
		int cid = (*it).first;
		assert(imgsT.find(cid) != imgsT.end(), "StereoReconstruction::Stereo() imgsT and RTs must contain the same camera IDs");
		Matrix<double, 3, 4> RT = (*it).second;
		RTs_[cid] = RT;
	}
}

// updates disparity map of size sz_out from perspective of camera with calibration K and extrinsics RT
// images is map of camera ID => BGR image
// height and width are pixel sizes of output display
// RTins is map of camera ID => camera extrinsics matrix
Matrix<float, Dynamic, Dynamic> StereoReconstruction::Stereo(int cid_out, Matrix3d K) {
	bool debug = false;

	// copy args locally
	cid_out_ = cid_out;
	height_ = imgsT_[cid_out_].rows;
	width_ = imgsT_[cid_out_].cols;
	K_ = K;
	
	TransformReferenceFrame(); // transform world space so output camera is at the origin
	
	InitDisps();

	// set up R_: a 2-dimensional matrix where the second dimension (columns) has size equal to the number of color channels per pixel. The number of rows equals imgT_out's rows*cols*2, where all values are repeated vertically, as in Matlab's repmat[2 1]
	R_ = Matrix<float, Dynamic, 3>(2 * height_*width_, 3);
	uchar* pT;
	for (int r = 0; r < height_; r++) {
		pT = imgsT_[cid_out_].ptr<uchar>(r);
		for (int c = 0; c < width_; c++) {
			int idx = c*height_ + r;
			R_(idx, 0) = (float)pT[3 * c + 0];
			R_(idx, 1) = (float)pT[3 * c + 1];
			R_(idx, 2) = (float)pT[3 * c + 2];
			R_(2 * idx, 0) = (float)pT[3 * c + 0];
			R_(2 * idx, 1) = (float)pT[3 * c + 1];
			R_(2 * idx, 2) = (float)pT[3 * c + 2];
		}
	}
	if (debug) {
		for (int i = 0; i < R_.rows(); i++) {
			cout << "R_(" << i << ", 0) " << R_(i, 0) << endl;
			cout << "R_(" << i << ", 1) " << R_(i, 1) << endl;
			cout << "R_(" << i << ", 2) " << R_(i, 2) << endl;
			cin.ignore();
		}
	}

	// set up SEI: 3xi (i==((rows)*(cols-2)+(rows-2)*(cols)) of output image) matrix of pixel location indices displaying across its rows connectivity of the output image (first vertical connectivity, then horizonal connectivity)
	SEI_ = Matrix<unsigned int, 3, Dynamic>(3, (height_)*(width_ - 2) + (height_ - 2)*(width_));
	for (int c = 0; c < width_; c++) {
		for (int r = 0; r < height_ - 2; r++) {
			int idx = c * (height_ - 2) + r;
			SEI_(0, idx) = idx;
			SEI_(1, idx) = idx + 1;
			SEI_(2, idx) = idx + 2;
		}
	}
	int offset = width_ * (height_ - 2);
	for (int c = 0; c < width_ - 2; c++) {
		for (int r = 0; r < height_; r++) {
			int idx = c * height_ + r;
			SEI_(0, idx + offset) = idx;
			SEI_(1, idx + offset) = idx + height_;
			SEI_(2, idx + offset) = idx + height_ * 2;
		}
	}
	if (debug) {
		for (int i = 0; i < 10; i++) {
			cout << "SEI_(0, " << i << ") " << SEI_(0, i) << endl;
			cout << "SEI_(1, " << i << ") " << SEI_(1, i) << endl;
			cout << "SEI_(2, " << i << ") " << SEI_(2, i) << endl;
			cin.ignore();
		}
		cout << endl << " ... " << endl;
		for (int i = SEI_.cols() - 10; i < SEI_.cols(); i++) {
			cout << "SEI_(0, " << i << ") " << SEI_(0, i) << endl;
			cout << "SEI_(1, " << i << ") " << SEI_(1, i) << endl;
			cout << "SEI_(2, " << i << ") " << SEI_(2, i) << endl;
			cin.ignore();
		}
	}

	
	ComputeMeanShiftSegmentation(); // segment the output texture image using mean shift
	
	InitEW(); // Find smoothness edges that don't cross segmentation boundaries

	SegmentPlanar();

	// set up our robust kernels
	//vals.ephoto = @(F)log(2) - log(exp(sum(F . ^ 2, 2)*(-1 / (col_thresh_*3))) + 1); // function handle to data cost energy function
	//vals.esmooth = @(F)EW.*min(abs(F), disp_thresh_); // function handle to smoothness prior energy function

	// set up proposals inputs
	//Dproposals = { info.segpln_optim.D, info.sameuni_optim.D, 2, 2, 2, 2 };

	// 1. SameUni(random fronto - parallel)
	// 2. SegPln(prototypical segment - based stereo proposals)
	// 3. Smooth*
	/*
	case 1
		% SameUni(random fronto - parallel)
		[D info.sameuni_optim] = ojw_stereo_optim(vals, @(n)1, options);
		info.sameuni_optim.D = D;
		case 2
			% SegPln(prototypical segment - based stereo proposals)
			[Dproposals info.segpln_gen] = ojw_segpln(images, P, disps, R, options);
			clear R
				Dproposals = @(n)Dproposals(:, : , mod(n - 1, size(Dproposals, 3)) + 1);
			[D info.segpln_optim] = ojw_stereo_optim(vals, Dproposals, options);
			clear Dproposals
				info.segpln_optim.D = D;
			case 3
				% Smooth*
				Dproposals = { info.segpln_optim.D, info.sameuni_optim.D, 2, 2, 2, 2 };
			Dproposals = @(n)Dproposals{ mod(n - 1, 6) + 1 };
			[D info.smooth_optim] = ojw_stereo_optim(vals, Dproposals, options);
			clear Dproposals
				info.smooth_optim.D = D;
	*/
	//Stereo_optim();

	Matrix<float, Dynamic, Dynamic> retval;
	return retval;
}

// generate piecewise-planar disparity proposals for stereo depth reconstruction
void StereoReconstruction::SegmentPlanar() {
	Eigen::Matrix<uchar, Dynamic, 3> Ruchar = R_.cast<uchar>();

	Eigen::Matrix<double, Dynamic, 3> WC(height_*width_, 3); // data structure containing homogeneous pixel positions across columns (u,v,1)
	WC.setOnes();
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++) {
			int idx = c*height_ + r;
			WC(idx, 0) = c;
			WC(idx, 1) = r;
		}
	}

	int h_adj = height_ - 2 * window_;
	int w_adj = width_ - 2 * window_;
	Eigen::Matrix<double, Dynamic, Dynamic> corr(h_adj*w_adj, num_disps_); // note that we're combining width and height here to avoid a 3D matrix

	//ephoto = @(F) log(2) - log(exp(sum((F-reshape(double(R), [], sz(3))) .^ 2, 2)*(-1/(options.col_thresh*sz(3))))+1);
	//filt = fspecial('average', [1 1 + 2 * options.window]); % creates a pre - defined 2D averaging filter

	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;

		// project the points
		Eigen::Matrix<double, Dynamic, 3> X = WC * Ps_[cid].block<3, 3>(0, 0); //  take homogeneous screen space coordinates and project them, sort of; nx3 result
		Eigen::Matrix<double, 3, 1> P4 = Ps_[cid].block<3, 1>(0, 3); // savelast column of P separately

		for (int b = 0; b < disps_.size(); b++) {
			// Vary image coordinates according to disparity
			Eigen::Matrix<double, 3, 1> d = disps_(b) * P4;
			Eigen::Matrix<double, Dynamic, 1> Z = X.block(0,2,X.size(),1).array() + d(2);
			Z = Z.array().inverse(); // array class inverse function is a fractional inverse, while matrix class version is a linear algebra inverse; can convert back and forth between Array and Matrix classes with .array() and .matrix(), and can also assign an array to a matrix with '=' and it will automatically convert

			Eigen::Matrix<double, Dynamic, 1> oX = X.block(0, 0, X.size(), 1).array() + d(0); // horizontal offsets
			oX = oX.cwiseProduct(Z);

			Eigen::Matrix<double, Dynamic, 1> oY = X.block(0, 1, X.size(), 1).array() + d(1); // vertical offsets
			oY = oY.cwiseProduct(Z);

			Eigen::Matrix<int, Dynamic, 3> Rinterp = Interpolation::Interpolate(width_, height_, Ruchar, oX, oY, -1000);

			//Y = squeeze(Rinterp);

		}
	}
	
}

/*
Matrix<float, Dynamic, Dynamic> StereoReconstruction::Stereo_optim(GLOBAL_PROPOSAL_METHOD proposal_method) {
	bool timing = true; double t;

	// create initial arrays
	Matrix<unsigned short, Dynamic, Dynamic> map(height_, width_);
	map.setZero();
	Matrix<double, Dynamic, 1> energy(max_iters_, 1);
	energy.setZero();
	Matrix<unsigned int, Dynamic, 1> numbers(max_iters_, 1); // number[updated; unlabelled; independent regions]
	numbers.setZero();
	Matrix<double, Dynamic, 1> timings(max_iters_, 1); // cumulative timings of[proposal; data; smoothness; optimization; finish]
	timings.setZero();

	// initialize disparity map values; unavailable values initialized to a random value
	D_ = Matrix<float, Dynamic, Dynamic>(height_, width_);
	float* pD;
	for (int r = 0; r < height_; r++) {
		pD = imgD_out_.ptr<float>(r);
		for (int c = 0; c < width_; c++) {
			if (pD[c] != 0.)
				D_(r, c) = 1. / pD[c];
			else D_(r, c) = min_disp_ + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max_disp_ - min_disp_)));
		}
	}

	converge_ = converge_ * 0.01 * (float)average_over_;
	int iter = average_over_ + 1;
	int energy_last_col_idx = energy.cols() - 1;
	for (int r = 0; r < average_over_; r++) {
		energy(r, energy_last_col_idx) = DBL_MAX; // for float, would be FLT_MAX
	}
	energy(iter, energy_last_col_idx) = DBL_MAX / 1e20;

	while ((1 - (energy(iter, energy_last_col_idx) / energy(iter - average_over_, energy_last_col_idx)) > converge_) &&
		(iter < max_iters_)) {
		iter++;
		if (timing) t = (double)getTickCount();

		// set the new (proposal) depth map
		int Dnew;
		
		if (proposal_method == SAME_UNI) Dnew = 1;
		else if (proposal_method == SEG_PLN) Dnew = ??;
		else Dnew = DisparityProposalSelectionUpdate_SmoothStar(iter - (average_over_ + 1)); // smooth*
		
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "StereoReconstruction::Stereo_optim() proposal generation time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}

}
*/