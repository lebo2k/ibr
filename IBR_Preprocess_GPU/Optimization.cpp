#include "Optimization.h"

Optimization::Optimization() {
}

Optimization::~Optimization() {
}

// requires reference image chosen and data set for sd
void Optimization::Init(StereoData *sd, int cid_out) {
	sd_ = sd; // don't delete it later because will be deleted by StereoReconstruction instance that calls this
	cid_out_ = cid_out;

	occl_val_ = (float)GLOBAL_LABELING_ENERGY_OCCLUSION_CONSTANT + log(2);
	scale_factor_ = (float)1e5 / FuseProposals_ePhoto((float)1e6);
	occl_cost_ = round(scale_factor_ * occl_val_);
	Kinf_ = occl_cost_ + 1; // smaller value seems to avoid errors in QPBO
	energy_val = 0.;
	num_in_ = sd_->use_cids_[cid_out_].size() - 1; // number of used input images excluding the reference(output) image - see ojw_stereo.m for where vals.I is set to exclude the reference image
	num_used_pixels_out_ = sd_->num_used_pixels_[cid_out_];
	num_unknown_pixels_out_ = sd_->num_unknown_pixels_[cid_out_];
	num_pixels_out_ = sd_->num_pixels_[cid_out_];
	oobv_ = -1000; // used in an interpolation call below, so must be type int

	InitSEI();
	//InitEW(); // InitSEI() must be called before this
	InitEW_new(); // InitSEI() must be called before this
	InitPairwiseInputs();
}

// initializes data structure EW, which identifies smoothness edges that don't cross segmentation boundaries
// requires reference image chosen and data set for sd
// requires that SEI_ has been initialized first through a call to InitSEI()
void Optimization::InitEW() {
	bool debug = false;

	if (debug) cout << "Optimization::InitEW()" << endl;
	
	/*
	// build an image to segment using StereoData::unknown_disps_valid_ member data.  By pixel, set output to equal maximum valid disparity
	Mat imgToSegment = Mat::zeros(sd_->heights_[cid_out_], sd_->widths_[cid_out_], CV_8UC1);
	uchar *pI;
	int idx_full, idx_unk;
	int h = sd_->heights_[cid_out_];
	float max;
	float max_scaled;
	for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
		pI = imgToSegment.ptr<uchar>(r);
		for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
			idx_full = PixIndexFwdCM(Point(c, r), h);
			idx_unk = sd_->unknown_maps_fwd_[cid_out_](idx_full, 0);
			max = sd_->GetMaxValidDisparity(cid_out_, idx_unk);
			max_scaled = 255 * (max - sd_->min_disps_[cid_out_]) / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_]);
			pI[c] = max_scaled;
		}
	}
	*/

	/*
	Mat imgT_gray = cv::Mat(sd_->imgsT_[cid_out_].size(), CV_8UC1, Scalar(0));
	cvtColor(sd_->imgsT_[cid_out_], imgT_gray, CV_BGR2GRAY); //convert the color space
	Mat imgT_gray_ET = DetermineEigenTransform(&imgT_gray, &sd_->masks_[cid_out_]);
	*/
	Matrix<unsigned int, Dynamic, 1> seg(sd_->num_pixels_[cid_out_], 1);
	//Segmentation::ComputeMeanShiftSegmentation(&seg, &imgToSegment, GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAS, GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAR, GLOBAL_MEAN_SHIFT_SEGMENTATION_MINREGION, sd_->heights_[cid_out_], sd_->widths_[cid_out_]); // segment the output texture image using mean shift
	Segmentation::ComputeMeanShiftSegmentation(&seg, &sd_->imgsT_[cid_out_], GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAS, GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAR, GLOBAL_MEAN_SHIFT_SEGMENTATION_MINREGION, sd_->heights_[cid_out_], sd_->widths_[cid_out_]); // segment the output texture image using mean shift; original
	//Segmentation::ComputeMeanShiftSegmentation_Grayscale(&seg, &imgToSegment, GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAS, GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAR, GLOBAL_MEAN_SHIFT_SEGMENTATION_MINREGION, sd_->heights_[cid_out_], sd_->widths_[cid_out_]); // segment the output texture image using mean shift

	/*
	Eigen::Matrix<float, 4, 1> segment_params;
	segment_params << 1, 1.5, 10, 100;
	Eigen::Matrix<int, 14, 1> mults;
	mults << 8, 12, 24, 50, 100;
	int nMaps = mults.size();
	Eigen::Matrix<float, 4, 1> sp;
	display_mat(&imgToSegment, "imgToSegment", sd_->orientations_[cid_out_]);
	for (int b = 0; b < nMaps; b++) {
		sp = segment_params * (float)mults(b, 0);
		Segmentation::ComputeGBSegmentation(&seg, &imgToSegment, 0, sp(3, 0), sp(2, 0), sd_->heights_[cid_out_], sd_->widths_[cid_out_], 1);
		DisplayImages::DisplaySegmentedImage(&seg, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
	*/
	if (debug) {
		DebugPrintMatrix(&seg, "Mean-shift segmentation for EW_ computation");
		//display_mat(&imgToSegment, "imgToSegment", sd_->orientations_[cid_out_]);
		DisplayImages::DisplaySegmentedImage(&seg, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}

	// EW = reshape(~any(diff(int32(info.segment(SEI))), 1), 1, []);
	Matrix<unsigned int, 3, Dynamic> seg_of_SEI = EigenMatlab::AccessByIndices(&seg, &SEI_); // since we're indexing into a full image segmentation, use SEI_ here instead of SEI_used_ because indices in SEI_ are full-image indices that only include used pixels, whereas SEI_used_ indices are indices into the compact used representation
	Matrix<int, 3, Dynamic> seg_of_SEI_int = seg_of_SEI.cast<int>();
	seg_of_SEI.resize(3, 0);
	Matrix<int, 2, Dynamic> diff_seg_of_SEI_int = EigenMatlab::Diff(&seg_of_SEI_int);
	seg_of_SEI_int.resize(3, 0);
	Matrix<bool, Dynamic, Dynamic> not_any_diff_seg_of_SEI_int = diff_seg_of_SEI_int.colwise().any().array() == 0; // result has true where pixels of triple-clique are all in the same segment, false where in different segments
	diff_seg_of_SEI_int.resize(2, 0);
	not_any_diff_seg_of_SEI_int.resize(1, not_any_diff_seg_of_SEI_int.rows() * not_any_diff_seg_of_SEI_int.cols());

	// EW = EW * options.lambda_h + ~EW * options.lambda_l;
	Matrix<int, 1, Dynamic> EW1 = not_any_diff_seg_of_SEI_int.cast<int>() * GLOBAL_LABELING_ENERGY_LAMBDA_H + not_any_diff_seg_of_SEI_int.cwiseEqual(0).cast<int>() * GLOBAL_LABELING_ENERGY_LAMBDA_L; // result has lamda_h where pixels of triple-clique were all in the same segment, and lamda_l where they were split among different segments
	not_any_diff_seg_of_SEI_int.resize(0, 0);

	// EW = EW * (num_in / ((options.connect==8) + 1));
	EW1 = EW1 * sd_->use_cids_[cid_out_].size(); // multiply result by the number of cameras being used, including the reference camera
	
	// EW = reshape(repmat(EW, [4*(1+(options.planar~=0)) 1]), [], 1);
	Matrix<int, Dynamic, Dynamic> EW2 = EW1.replicate(8, 1);
	EW1.resize(1, 0);
	EW2.resize(EW2.rows()*EW2.cols(), 1);
	EW_ = EW2;
	EW2.resize(0, 0);

	if (debug) DebugPrintMatrix(&EW_, "EW_");
}

// initializes data structure EW_, which identifies smoothness edges that don't cross segmentation boundaries
// segmentation boundaries are derived from pixels with mask values of 128 instead of 0 or 255
// requires reference image chosen and data set for sd
// requires that SEI_ has been initialized first through a call to InitSEI()
void Optimization::InitEW_new() {
	bool debug = false;

	if (debug) cout << "Optimization::InitEW_new()" << endl;

	Matrix<bool, 1, Dynamic> seg(1, SEI_.cols()); // to have value of true where all three pixels in the corresponding SEI_ triple-clique column are in the same segment, false otherwise
	seg.setConstant(true);

	// define significant disparity difference
	float depth_dist_for_each_disp_label = (sd_->max_depths_[cid_out_] - sd_->min_depths_[cid_out_]) / sd_->nums_disps_[cid_out_];
	int max_num_disp_labels = ceil(static_cast<float>(GLOBAL_MESH_EDGE_DISTANCE_MAX) / depth_dist_for_each_disp_label); // maximum number of disparity label steps that correspond to GLOBAL_MESH_EDGE_DISTANCE_MAX

	// for debug visualization
	Matrix<bool, Dynamic, 1> vis(sd_->heights_[cid_out_] * sd_->widths_[cid_out_], 1);
	vis.setConstant(true);
	
	/*
	// add segmentation boundaries at pixels with mask values of 128 instead of 0 or 255
	bool *pSeg = seg.data();
	unsigned int *pSEI = SEI_.data();
	unsigned int idx_full1, idx_full2, idx_full3; // indices in SEI_ are full-image indices that only include used pixels, whereas SEI_used_ indices are indices into the compact used representation
	for (int c = 0; c < SEI_.cols(); c++) {
		idx_full1 = *pSEI++;
		idx_full2 = *pSEI++;
		idx_full3 = *pSEI++;

		if ((sd_->masks_int_[cid_out_](idx_full2, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
			(sd_->masks_int_[cid_out_](idx_full2, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL)) {
			*pSeg = false;
			if (debug) vis(idx_full2, 0) = false;
		}

		pSeg++;
	}
	*/

	// EW = EW * options.lambda_h + ~EW * options.lambda_l;
	Matrix<int, 1, Dynamic> EW1 = seg.cast<int>() * GLOBAL_LABELING_ENERGY_LAMBDA_H + seg.cwiseEqual(0).cast<int>() * GLOBAL_LABELING_ENERGY_LAMBDA_L; // result has lamda_h where pixels of triple-clique were all in the same segment, and lamda_l where they were split among different segments
	seg.resize(1, 0);

	// EW = EW * (num_in / ((options.connect==8) + 1));
	EW1 = EW1 * sd_->use_cids_[cid_out_].size(); // multiply result by the number of cameras being used, including the reference camera

	// EW = reshape(repmat(EW, [4*(1+(options.planar~=0)) 1]), [], 1);
	Matrix<int, Dynamic, Dynamic> EW2 = EW1.replicate(8, 1);
	EW1.resize(1, 0);
	EW2.resize(EW2.rows()*EW2.cols(), 1);
	EW_ = EW2;
	EW2.resize(0, 0);

	if (debug) {
		//DisplayImages::DisplayGrayscaleImage(&sd_->masks_int_[cid_out_], sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		DisplayImages::DisplayGrayscaleImage(&vis, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		DebugPrintMatrix(&EW_, "EW_");
	}
}

// initializes data structure SEI_
// set up SEI: 3xi (i==(3*(rows)*(cols-2)+3*(rows-2)*(cols)) of output image) matrix of pixel location indices (including separate color channels in RGB order) displaying across its rows connectivity of the output image (first vertical connectivity, then horizonal connectivity); colors are num_pixels apart
// requires reference image chosen and data set for sd
// note that triple-clique smoothness should include triple-cliques that include high-confidence pixels as long as one unknown pixel is present in the clique.  Masked-out pixels, however, should not appear in any triple cliques.
// triple-clique smoothness indices should also include any masked-in triple-cliques with mask values of 128 instead of 0 or 255 since these will comprise segmentation boundaries in EW for which we must have indices in SEI
void Optimization::InitSEI() {
	bool debug = false;

	if (debug) {
		cout << "Optimization::InitSEI()" << endl;
		cout << "height " << sd_->heights_[cid_out_] << ", width " << sd_->widths_[cid_out_] << ", num_pixels " << sd_->num_pixels_[cid_out_] << endl;
	}

	Matrix<unsigned int, 3, Dynamic> SEI(3, (sd_->heights_[cid_out_] - 2)*(sd_->widths_[cid_out_]) + (sd_->heights_[cid_out_])*(sd_->widths_[cid_out_] - 2));
	SEI.setZero();
	Matrix<unsigned int, Dynamic, Dynamic> T_sei(sd_->num_pixels_[cid_out_], 1);
	T_sei.col(0).setLinSpaced(sd_->num_pixels_[cid_out_], 0, sd_->num_pixels_[cid_out_] - 1);
	T_sei.resize(sd_->heights_[cid_out_], sd_->widths_[cid_out_]);
	Matrix<unsigned int, Dynamic, Dynamic> T_sei_1 = T_sei.block(0, 0, T_sei.rows() - 2, T_sei.cols());
	T_sei_1.resize(1, T_sei_1.rows()*T_sei_1.cols());
	Matrix<unsigned int, Dynamic, Dynamic> T_sei_2 = T_sei.block(1, 0, T_sei.rows() - 2, T_sei.cols());
	T_sei_2.resize(1, T_sei_2.rows()*T_sei_2.cols());
	Matrix<unsigned int, Dynamic, Dynamic> T_sei_3 = T_sei.block(2, 0, T_sei.rows() - 2, T_sei.cols());
	T_sei_3.resize(1, T_sei_3.rows()*T_sei_3.cols());
	Matrix<unsigned int, Dynamic, Dynamic> T_sei_4 = T_sei.block(0, 0, T_sei.rows(), T_sei.cols() - 2);
	T_sei_4.resize(1, T_sei_4.rows()*T_sei_4.cols());
	Matrix<unsigned int, Dynamic, Dynamic> T_sei_5 = T_sei.block(0, 1, T_sei.rows(), T_sei.cols() - 2);
	T_sei_5.resize(1, T_sei_5.rows()*T_sei_5.cols());
	Matrix<unsigned int, Dynamic, Dynamic> T_sei_6 = T_sei.block(0, 2, T_sei.rows(), T_sei.cols() - 2);
	T_sei_6.resize(1, T_sei_6.rows()*T_sei_6.cols());
	SEI.block(0, 0, 1, T_sei_1.cols()) = T_sei_1;
	SEI.block(1, 0, 1, T_sei_2.cols()) = T_sei_2;
	SEI.block(2, 0, 1, T_sei_3.cols()) = T_sei_3;
	SEI.block(0, T_sei_1.cols(), 1, T_sei_4.cols()) = T_sei_4;
	SEI.block(1, T_sei_2.cols(), 1, T_sei_5.cols()) = T_sei_5;
	SEI.block(2, T_sei_3.cols(), 1, T_sei_6.cols()) = T_sei_6;
	T_sei_1.resize(0, 0);
	T_sei_2.resize(0, 0);
	T_sei_3.resize(0, 0);
	T_sei_4.resize(0, 0);
	T_sei_5.resize(0, 0);
	T_sei_6.resize(0, 0);

	if (debug) {
		DebugPrintMatrix(&SEI, "SEI");
		DebugPrintMatrix(&sd_->masks_[cid_out_], "sd_->masks_[cid_out_]");
		DebugPrintMatrix(&sd_->masks_unknowns_[cid_out_], "sd_->masks_unknowns_[cid_out_]");
	}

	// eliminate columns that include masked pixels in the triple clique; leave in masked-in pixels regardless of the availability of depth information as long as there is at least one pixel in the clique for which no depth information is available
	// triple-clique smoothness indices should also exclude any masked-in triple-cliques that cross segmentation boundaries since these should be excluded from EW, as well as masked-out pixels
	Matrix<unsigned int, 3, Dynamic> tmpSEI(3, SEI.cols());
	unsigned int *pS = SEI.data();
	int trunc_cols = 0;
	int idx = 0;
	Point p;
	unsigned int label;
	bool has_unknown_depth, label_match, has_masked_out_pixel;
	for (int c = 0; c < SEI.cols(); c++) {
		has_unknown_depth = false;
		has_masked_out_pixel = false;
		label_match = true;
		for (int r = 0; r < 3; r++) {
			idx = *pS++;
			p = PixIndexBwdCM(idx, sd_->heights_[cid_out_]);
			if (!sd_->masks_[cid_out_](idx, 0)) has_masked_out_pixel = true;
			if (sd_->masks_unknowns_[cid_out_](idx, 0)) has_unknown_depth = true;
			if (r == 0) label = sd_->segs_[cid_out_](p.y, p.x);
			else if (label != sd_->segs_[cid_out_](p.y, p.x)) label_match = false;
		}
		if ((has_unknown_depth) &&
			(label_match) &&
			(!has_masked_out_pixel)) {
			tmpSEI.col(trunc_cols) = SEI.col(c);
			trunc_cols++;
		}
	}
	SEI_ = tmpSEI.block(0, 0, 3, trunc_cols);

	SEI_used_ = sd_->MapFulltoUsedIndices(cid_out_, &SEI_);

	if (debug) {
		DebugPrintMatrix(&SEI_, "SEI_");
		DebugPrintMatrix(&SEI_used_, "SEI_used_");

		Mat SEItest = cv::Mat::zeros(sd_->heights_[cid_out_], sd_->widths_[cid_out_], CV_8UC1);
		unsigned int *pI = SEI_.data();
		int idx;
		Point pt;
		for (int c = 0; c < SEI_.cols(); c++) {
			for (int r = 0; r < SEI_.rows(); r++) {
				idx = *pI++;
				pt = PixIndexBwdCM(idx, sd_->heights_[cid_out_]);
				SEItest.at<uchar>(pt.y, pt.x) = 255;
			}
		}
		display_mat(&SEItest, "SEItest", sd_->orientations_[cid_out_]);
	}
}

// pre-computes some data for WC, EI, and E data structures for use in Optimization::FuseProposals() for each image in turn assume it's the reference image
// only uses pixels that are not masked out and for which a disparity is not known with high confidence
// updates WCs_, EIs_, and Es_ in sd_
// WC indices are expected to be in column-major order
void Optimization::InitPairwiseInputs() {
	bool debug = false;
	
	// Calculate the homogenous coordinates of our two labellings
	WC_partial_ = Matrix<double, Dynamic, 4>(num_unknown_pixels_out_ * 2, 4); // data structure containing homogeneous pixel positions across columns with disparities (u,v,1,disp), with the two input disparity maps one after another vertically
	WC_partial_.block(0, 0, num_unknown_pixels_out_, 1) = sd_->Xunknowns_[cid_out_];
	WC_partial_.block(num_unknown_pixels_out_, 0, num_unknown_pixels_out_, 1) = sd_->Xunknowns_[cid_out_];
	WC_partial_.block(0, 1, num_unknown_pixels_out_, 1) = sd_->Yunknowns_[cid_out_];
	WC_partial_.block(num_unknown_pixels_out_, 1, num_unknown_pixels_out_, 1) = sd_->Yunknowns_[cid_out_];
	WC_partial_.block(0, 2, WC_partial_.rows(), 1).setOnes();

	if (debug) DebugPrintMatrix(&WC_partial_, "WC_partial_");

	// initialize arrays for the data terms EI and E ... assume visibility_==true, so data edges are needed (we don't provide for the alternative case here)
	// EI has 2 rows and # columns = 2*num_in*num_pixels (where num_in is the number of input images excluding the reference image); the first row repeats 0:(num_pixels-1) over and over again; the second row starts at num_pixels and increments each column until the last
	// EI is a 2xpairwise_energy_sets matrix where the first num_pixels correspond to the Dcurr proposal projected into the first input image, the next num_pixels correspond to the Dnew proposal projected into the first input image, the next num_pixels correspond to the Dcurr proposal projected into the second input image, the next num_pixels correspond to the Dnew proposal projected into the second input image, etc.  So it each input image gets num_pixels*2 columns, in image order, where the first half are for the Dcurr proposal and the second half for the Dnew proposal; or perhaps it's really the direction of the cost that's switching back and forth?
	// EI = reshape(repmat(uint32(tp+(0:num_in-1)*2*tp), [2*tp 1]), 1, []);
	// EI = [repmat(uint32(1:tp), [1 2 * num_in]); EI + repmat(uint32(1:2 * tp), [1 num_in])];
	int pairwise_energy_sets = 2 * num_in_ * num_unknown_pixels_out_;
	EI_partial_.resize(2, pairwise_energy_sets); // pairwise energy nodes; column is index into pairwise energies column in E; values are column-major indices into images - for each non-reference input image, there are num_pixels columns of pairs of pixels in that input image and the reference image; the first row is the index into the reference image (start node), the second the corresponding index into an input image (end node)
	Matrix<unsigned int, 1, Dynamic> Iunknowns_unk(1, num_unknown_pixels_out_);
	Iunknowns_unk.setLinSpaced(num_unknown_pixels_out_, 0, (num_unknown_pixels_out_ - 1));
	Matrix<unsigned int, 1, Dynamic> Iunknowns_full = sd_->MapUnknownToFullIndices(cid_out_, &Iunknowns_unk, 2);
	Iunknowns_unk.resize(1, 0);
	Matrix<unsigned int, 1, Dynamic> Iunknowns_used = sd_->MapFulltoUsedIndices(cid_out_, &Iunknowns_full, 2);
	Iunknowns_full.resize(1, 0);

	int i = 0;
	for (int c = 0; c < pairwise_energy_sets; c += num_unknown_pixels_out_) {
		EI_partial_.block(0, c, 1, num_unknown_pixels_out_) = Iunknowns_used;
		EI_partial_.block(1, c, 1, num_unknown_pixels_out_) = Iunknowns_used.array() + num_used_pixels_out_*(i + 1); // each set of input image indices should start after the previous max index, even though there are only num_used_pixels_out_ terms for each proposal (of 2) for each input image combo with the reference image
		i++;
	}

	Iunknowns_used.resize(1, 0);
	
	/*
	for (std::vector<int>::iterator it = sd_->use_cids_[cid_out_].begin(); it != sd_->use_cids_[cid_out_].end(); ++it) {
		int cid = (*it);
		if (cid == cid_out_) continue;
		for (int j = 0; j < 2; j++) { // once for each proposal of the two
			EI_partial_.block(0, i, 1, num_used_pixels_out_) = Iunknowns_.transpose();
			EI_partial_.block(1, i, 1, num_used_pixels_out_) = Iunknowns_.transpose().array() + num_used_pixels_out_ + i; // each set of input image indices should start after the previous max index
			i += sd_->num_used_pixels_[cid];
		}
	}
	*/
	/*
	for (int c = 0; c < pairwise_energy_sets; c += num_used_pixels_out_) {
		EI_partial_.block(0, c, 1, num_used_pixels_out_) = Iunknowns_.transpose();
		EI_partial_.block(1, c, 1, num_used_pixels_out_) = Iunknowns_.transpose().array() + num_used_pixels_out_ + c; // each set of input image indices should start after the previous max index
	}
	*/

	if (debug) DebugPrintMatrix(&EI_partial_, "EI_partial_");


	// E 1st num_pixels values (2nd dimension position 0) has value occl_cost when 3rd dimension equals 0 (every odd position in rows); and 3rd stretch of num_pixel values (2nd dimension position 2) has value occl_cost when 3rd dimension equals 1 (every even position in columns)
	// E is a 4xpairwise_energy_sets matrix where the first num_pixels correspond to the Dcurr proposal projected into the first input image, the next num_pixels correspond to the Dnew proposal projected into the first input image, the next num_pixels correspond to the Dcurr proposal projected into the second input image, the next num_pixels correspond to the Dnew proposal projected into the second input image, etc.  So it each input image gets num_pixels*2 columns, in image order, where the first half are for the Dcurr proposal and the second half for the Dnew proposal; or perhaps it's really the direction of the cost that's switching back and forth?
	E_partial_ = Matrix<int, 4, Dynamic>(4, pairwise_energy_sets); // pairwise energies - each column holds the energy values for a node (pixel) pair; columns are ordered same as in EI; energies are initialized to occl_cost (as if one node is occluding the other) in one direction for first input image, opposite for second, back to original for third, opposite for fourth, etc.
	E_partial_.setZero();
	Matrix<int, 1, Dynamic> tmp(1, num_unknown_pixels_out_);
	tmp.setConstant(occl_cost_);
	int r;
	bool E_first_row = true;
	for (int i = 0; i < E_partial_.cols(); i += num_unknown_pixels_out_) {
		if (E_first_row) r = 0;
		else r = 2;
		E_partial_.block(r, i, 1, num_unknown_pixels_out_) = tmp;
		E_first_row = !E_first_row;
	}

	if (debug) DebugPrintMatrix(&E_partial_, "E_partial_");
}

// compute matrix ePhoto for fuse depths
// vals.ephoto = @(F)log(2) - log(exp(sum(F . ^ 2, 2)*(-1 / (col_thresh*3))) + 1); // function handle to data cost energy function
void Optimization::FuseProposals_ePhoto(Eigen::Matrix<double, Dynamic, 3> *F, Eigen::Matrix<double, Dynamic, 1> *X, Eigen::Matrix<double, Dynamic, 3> *scratch) {
	assert(F->rows() == X->rows());
	//assert(F->rows() == scratch->rows(), "Optimization::FuseProposals_ePhoto() F and scratch must have the same number of rows");
	
	double col_thresh = GLOBAL_LABELING_ENERGY_COL_THRESHOLD;
	double factor = -1. / (col_thresh*3.);
	
	(*scratch) = F->cwiseProduct((*F));

	//(*X) = scratch.rowwise().sum(); // slower method for row-wise summation
	(*X) = (*scratch) * VectorXd::Ones(3); // faster method for row-wise summation

	(*X) = (*X) * factor;
	(*X) = X->array().exp();
	(*X) = X->array() + 1;
	(*X) = X->array().log();
	(*X) = -1 * (*X).array() + log(2);
}

// vals.esmooth = @(F) EW .* min(abs(F), disp_thresh); since options.smoothness_kernel == 1 for truncated linear kernel
// operates on F in-place
void Optimization::FuseProposals_eSmooth(Matrix<double, Dynamic, Dynamic> *F) {
	assert(F->cols() == EW_.cols() && F->rows() == EW_.rows());
	double disp_thresh = GLOBAL_LABELING_ENERGY_DISPARITY_THRESHOLD;
	(*F) = F->cwiseAbs();
	(*F) = F->cwiseMin(disp_thresh);
	(*F) = F->cwiseProduct(EW_.cast<double>());
}

// photoconsistency costs are based on the color error if the pixel is visible, and are occl_cost_ if the pixel is not visible; so update E to take into account label segmentation and mapping so that if a pixel projects to an unassigned label in the input image (and if another label is assigned for that input image as the mapping, meaning we have some info for it to compare against), then it's considered not visible and that's reflected in photoconsistency costs in E by setting it to occl_cost_
// the first 2 columns of WC hold all (x,y) unknown coordinates for both Dcurr and Dnew, while the first 2 columns of T hold the reprojection of those coordinates (x,y) into cid_in (the first sd_->num_unknown_pixels_[cid_out_] group of coordinates in WC and T corresponds to Dcurr and the second group in WC and T corresponds to Dnew)
// map<int, int> label_mappings is map of label_out for cid_out_ => label_in for cid_in
// img_num is index into used images that is used in FuseDepths()
void Optimization::UpdatePhotoconsistencyVisibility(int cid_in, int img_num, map<int, int> label_mappings, Matrix<double, Dynamic, 4> *WC, Matrix<double, Dynamic, 3> *T, Matrix<int, 4, Dynamic> *E) {
	assert(WC->rows() == T->rows());
	assert(WC->rows() == 2 * sd_->num_unknown_pixels_[cid_out_]);

	bool debug = false;

	if (debug) cout << "Optimization::UpdatePhotoconsistencyVisibility()" << endl;
	
	double *pXcurr = WC->col(0).data();
	double *pYcurr = WC->col(1).data() + sd_->num_unknown_pixels_[cid_out_];
	double *pXnew = WC->col(0).data() + sd_->num_unknown_pixels_[cid_out_];
	double *pYnew = WC->col(1).data() + sd_->num_unknown_pixels_[cid_out_];
	double *pXcurr_reproj = T->col(0).data();
	double *pYcurr_reproj = T->col(1).data() + sd_->num_unknown_pixels_[cid_out_];
	double *pXnew_reproj = T->col(0).data() + sd_->num_unknown_pixels_[cid_out_];
	double *pYnew_reproj = T->col(1).data() + sd_->num_unknown_pixels_[cid_out_];
	double Xcurrd, Ycurrd, Xnewd, Ynewd, Xcurrd_reproj, Ycurrd_reproj, Xnewd_reproj, Ynewd_reproj;
	int Xcurr, Ycurr, Xnew, Ynew, Xcurr_reproj, Ycurr_reproj, Xnew_reproj, Ynew_reproj;
	unsigned int label_out, label_in;
	bool visible_curr, visible_new;
	int idx_full_curr, idx_full_new, idx_used_curr, idx_used_new;
	for (int idx_unk = 0; idx_unk < sd_->num_unknown_pixels_[cid_out_]; idx_unk++) {
		if (debug) cout << "processing idx_unk " << idx_unk << endl;

		Xcurrd = *pXcurr++;
		Ycurrd = *pYcurr++;
		Xnewd = *pXnew++;
		Ynewd = *pYnew++;
		Xcurrd_reproj = *pXcurr_reproj++;
		Ycurrd_reproj = *pYcurr_reproj++;
		Xnewd_reproj = *pXnew_reproj++;
		Ynewd_reproj = *pYnew_reproj++;

		Xcurr = round(Xcurrd);
		Ycurr = round(Ycurrd);
		Xnew = round(Xnewd);
		Ynew = round(Ynewd);
		Xcurr_reproj = round(Xcurrd_reproj);
		Ycurr_reproj = round(Ycurrd_reproj);
		Xnew_reproj = round(Xnewd_reproj);
		Ynew_reproj = round(Ynewd_reproj);

		assert((Xcurr >= 0) && (Ycurr >= 0) &&
				(Xcurr < sd_->widths_[cid_out_]) && (Ycurr < sd_->heights_[cid_out_]) &&
				(Xnew >= 0) && (Ynew >= 0) &&
				(Xnew < sd_->widths_[cid_out_]) && (Ynew < sd_->heights_[cid_out_]));

		if ((Xcurr_reproj < 0) || (Ycurr_reproj < 0) ||
			(Xcurr_reproj >= sd_->widths_[cid_in]) || (Ycurr_reproj >= sd_->heights_[cid_in])) {
			visible_curr = false;
		}
		else {
			label_out = sd_->segs_[cid_out_](Ycurr, Xcurr);
			label_in = sd_->segs_[cid_in](Ycurr_reproj, Xcurr_reproj);
			if (label_mappings[label_out] == label_in)
				visible_curr = true;
			else visible_curr = false;
		}

		if ((Xnew_reproj < 0) || (Ynew_reproj < 0) ||
			(Xnew_reproj >= sd_->widths_[cid_in]) || (Ynew_reproj >= sd_->heights_[cid_in])) {
			visible_new = false;
		}
		else {
			label_out = sd_->segs_[cid_out_](Ynew, Xnew);
			label_in = sd_->segs_[cid_in](Ynew_reproj, Xnew_reproj);
			if (label_mappings[label_out] == label_in)
				visible_new = true;
			else visible_new = false;
		}

		idx_full_curr = PixIndexFwdCM(Point(Xcurr, Ycurr), sd_->heights_[cid_out_]);
		idx_full_new = PixIndexFwdCM(Point(Xnew, Ynew), sd_->heights_[cid_out_]);
		idx_used_curr = sd_->used_maps_fwd_[cid_out_](idx_full_curr, 0);
		idx_used_new = sd_->used_maps_fwd_[cid_out_](idx_full_new, 0);

		if (debug) {
			cout << "idx_used_curr " << idx_used_curr  << endl;
			cout << "idx_used_new " << idx_used_new << endl;
			cout << "visible_curr " << visible_curr << endl;
			cout << "visible_new " << visible_new << endl;
			cout << endl;
		}

		if (!visible_curr)
			(*E)(1, num_unknown_pixels_out_*(img_num * 2) + idx_unk) = occl_cost_;
		if (!visible_new)
			(*E)(3, num_unknown_pixels_out_*(img_num * 2 + 1) + idx_unk) = occl_cost_;
	}
}

// updates Dswap to booleans denoting whether the value in Dcurr should be replaced by the value in Dnew
// updates the value in energy in the row given by iter to the energy of the fused disparity map
// updates sd_->timings_xx with additional timing information for the iteration iter (iter indexes the column of timings) for the following types of timing: smoothness_term_eval, qpbo_fuse_time\
// updates sd_->count_xx
// Dcurr, Dnew, and Dswap are of size sd_->num_pixels_[cid_out_] x 1
// energy is of size max_iters x 1
// timings is of size 4 x max_iters where the 4 rows are for: data_term_eval, smoothness_term_eval, qpbo_fuse_time, finish_time
// if (!include_smoothness_terms), then smoothness triple-clique terms are not used in QPBO
// map<int, map<int, int>> label_mappings; // for cid_out_, map of cid_in => label_out => label_in
void Optimization::FuseProposals(const Eigen::Matrix<double, Dynamic, 1> *Dcurr, const Eigen::Matrix<double, Dynamic, 1> *Dnew, Eigen::Matrix<bool, Dynamic, 1> *Dswap, const int iter, Eigen::Matrix<double, Dynamic, 1> *energy, map<int, map<int, int>> label_mappings, bool include_smoothness_terms) {
	assert(Dcurr->rows() == num_pixels_out_ && Dnew->rows() == num_pixels_out_ && Dswap->rows() == num_pixels_out_);
	assert(iter < energy->rows() && iter >= 0);

	bool debug = true;

	//if (iter != 1) debug = false; // remove this line *****************************************************************************************

	cout << "Optimization::FuseProposals() iteration " << iter << endl;

	double t_start = (double)getTickCount();
	double t_last = t_start;
	double t;

	// Calculate the homogenous coordinates of our two labellings
	Matrix<double, Dynamic, 1> Dcurrtrunc_unk = sd_->ContractFullToUnknownSize(cid_out_, Dcurr);
	Matrix<double, Dynamic, 1> Dnewtrunc_unk = sd_->ContractFullToUnknownSize(cid_out_, Dnew);
	Matrix<double, Dynamic, 4> WC = WC_partial_; // data structure containing homogeneous pixel positions across columns with disparities (u,v,1,disp), with the two input disparity maps one after another vertically
	WC.block(0, 3, num_unknown_pixels_out_, 1) = Dcurrtrunc_unk;
	WC.block(num_unknown_pixels_out_, 3, num_unknown_pixels_out_, 1) = Dnewtrunc_unk;
	Dcurrtrunc_unk.resize(0, 1);
	Dnewtrunc_unk.resize(0, 1);

	// initialize arrays for the data terms EI and E ... assume visibility_==true, so data edges are needed (we don't provide for the alternative case here)
	// EI has 2 rows and # columns = 2*num_in*num_pixels (where num_in is the number of input images excluding the reference image); the first row repeats 0:(num_pixels-1) over and over again; the second row starts at num_pixels and increments each column until the last
	// EI is a 2xpairwise_energy_sets matrix where the first num_pixels correspond to the Dcurr proposal projected into the first input image, the next num_pixels correspond to the Dnew proposal projected into the first input image, the next num_pixels correspond to the Dcurr proposal projected into the second input image, the next num_pixels correspond to the Dnew proposal projected into the second input image, etc.  So it each input image gets num_pixels*2 columns, in image order, where the first half are for the Dcurr proposal and the second half for the Dnew proposal; or perhaps it's really the direction of the cost that's switching back and forth?
	// EI = reshape(repmat(uint32(tp+(0:num_in-1)*2*tp), [2*tp 1]), 1, []);
	// EI = [repmat(uint32(1:tp), [1 2 * num_in]); EI + repmat(uint32(1:2 * tp), [1 num_in])];
	Matrix<unsigned int, 2, Dynamic> EI = EI_partial_; // (2, pairwise_energy_sets); pairwise energy nodes; column is index into pairwise energies column in E; values are column-major indices into images - for each non-reference input image, there are num_pixels columns of pairs of pixels in that input image and the reference image; the first row is the index into the reference image, the second the corresponding index into an input image

	// E 1st num_pixels values (2nd dimension position 0) has value occl_cost when 3rd dimension equals 0 (every odd position in rows); and 3rd stretch of num_pixel values (2nd dimension position 2) has value occl_cost when 3rd dimension equals 1 (every even position in columns)
	// E is a 4xpairwise_energy_sets matrix where the first num_pixels correspond to the Dcurr proposal projected into the first input image, the next num_pixels correspond to the Dnew proposal projected into the first input image, the next num_pixels correspond to the Dcurr proposal projected into the second input image, the next num_pixels correspond to the Dnew proposal projected into the second input image, etc.  So it each input image gets num_pixels*2 columns, in image order, where the first half are for the Dcurr proposal and the second half for the Dnew proposal; or perhaps it's really the direction of the cost that's switching back and forth?
	Matrix<int, 4, Dynamic> E = E_partial_; // (4, pairwise_energy_sets); pairwise energies - each column holds the energy values for a node (pixel) pair; columns are ordered same as in EI; energies are initialized to occl_cost (as if one node is occluding the other) in one direction for first input image, opposite for second, back to original for third, opposite for fourth, etc.

	Matrix<unsigned int, 2, Dynamic> TEI(2, 0); // columns are concatenated on with each iteration; there are 2 sets of num_unknown_pixels_out_ columns for every input image (one for Dcurr and one for Dnew); local indices are always according to compact unknown representation for the associated image; indices should always increment from the max already in use, so the amount by which to increment is different for each image (reference and each input) since they have differing numbers of unknown pixels)
	Matrix<int, 4, Dynamic> TEinit(4, 0); // columns are concatenated on with each iteration (final TE is created later with a different size)

	Matrix<double, Dynamic, 3> T(num_unknown_pixels_out_ * 2, 3); // visibility edge costs
	Matrix<double, Dynamic, 1> N(num_unknown_pixels_out_ * 2, 1); // visibility edge indices
	Matrix<double, Dynamic, 3> Runk_stacked = sd_->Aunknowns_[cid_out_].replicate(2, 1).cast<double>(); // (num_unknown_pixels_out_ * 2, 3); // stacked texture image

	// for use below - matrices that are the same size for every loop
	Matrix<double, Dynamic, 3> A(num_unknown_pixels_out_, 3); // used to calculate photoconsistency
	Matrix<double, Dynamic, 3> M(num_unknown_pixels_out_ * 2, 3); // used to calculate photoconsistency
	Matrix<double, Dynamic, 1> X(num_unknown_pixels_out_ * 2, 1); // used to calculate photoconsistency
	Matrix<double, Dynamic, 1> Y(num_unknown_pixels_out_ * 2, 1); // used to calculate photoconsistency
	Matrix<int, Dynamic, 1> IA(num_unknown_pixels_out_ * 2, 1);
	Matrix<double, Dynamic, 1> WC3 = WC.block(0, 3, WC.rows(), 1).array().inverse(); // used to find interactions; size (num_pixels * 2, 1)
	Matrix<double, Dynamic, 3> Tunsorted(num_unknown_pixels_out_ * 2, 3); // used to find interactions
	VectorXi IM(num_unknown_pixels_out_ * 2); // used to find interactions; typedef Matrix< int , Dynamic , 1> VectorXi
	Eigen::Matrix<double, Dynamic, 3> ePhoto_scratch(num_unknown_pixels_out_ * 2, 3); // scratch matrix for FuseProposals_ePhoto() call in loop below so don't have to reallocate every loop
	Matrix<int, Dynamic, 2> M_compressgraph(num_used_pixels_out_, 2); // used to determine the photoconsistency nodes which have no interactions when are planning to compress the graph
	Matrix<int, Dynamic, 2> U_compressgraph(num_used_pixels_out_, 2); // used to initialize matrix U when planning to compress the graph
	Matrix<int, Dynamic, 2> IA_compressgraph_unk(num_unknown_pixels_out_, 2); // used to initialize matrix U when planning to compress the graph; based on unknown pixels like IA, but reconfigured from IA
	Matrix<int, Dynamic, 2> IA_compressgraph_used(num_used_pixels_out_, 2); // used to initialize matrix U when planning to compress the graph; based on used pixels

	if (debug) cout << "completed variable setup" << endl;

	int img_num = 0; // 0-indexed counter of which image we're on in the loop; used to index list of images, including 4th dimension of E
	for (vector<int>::iterator it = sd_->use_cids_[cid_out_].begin(); it != sd_->use_cids_[cid_out_].end(); ++it) { // add them in order of closest to camera angle
		int cid = (*it);
		if (cid == cid_out_) continue; // ojw_stereo_optim.m sets num_in to the number of images excluding the reference image (numel(vals.I) where vals.I=images(2:end) from ojw_stereo.m), then sets vals.P to P(2:end), but takes the transpose of each by permuting rows

		//if (debug) cout << "investigating cid " << cid << endl;

		// calculate the coordinates in the input image
		// T = WC * vals.P(:,:,a); // project points from output screen space into the input screen space for camera cid using each of the two disparity proposals (one stacked on top of the other in rows)
		T = WC * sd_->Pout2ins_[cid_out_][cid].transpose().cast<double>(); // vals.P is transpose of P; project WC into screen space of current input image; note that WC is in screen space of reference image and Ps have been altered to transform directly from reference image screen space to input image screen space
		// N = 1 ./ T(:,3);
		N = T.block(0, 2, T.rows(), 1).array().inverse(); // determine homogeneous coordinates to divide by
		// T(:,1) = T(:,1) .* N; // normalize screen space x coordinate
		T.block(0, 0, T.rows(), 1) = T.block(0, 0, T.rows(), 1).cwiseProduct(N); // divide by homogeneous coordinates
		// T(:,2) = T(:,2) .* N; // normalize screen space y coordinate
		T.block(0, 1, T.rows(), 1) = T.block(0, 1, T.rows(), 1).cwiseProduct(N); // divide by homogeneous coordinates
		N.resize(0, 1);

		//if (debug) DebugPrintMatrix(&T, "T");

		// Calculate photoconsistency
		// M = vgg_interp2(vals.I{a}, T(:,1), T(:,2), 'linear', oobv); // use interpolation to determine pixel color at the floating point x and y screen space pixel coordinates in input screen space after projecting points from output screen space using each of the two disparity proposals
		A = sd_->As_[cid].cast<double>();
		X = T.block(0, 0, T.rows(), 1);
		Y = T.block(0, 1, T.rows(), 1);
		Interpolation::Interpolate<double>(sd_->imgsT_[cid].cols, sd_->imgsT_[cid].rows, &A, &X, &Y, &sd_->masks_[cid], oobv_, &M);
		//if (debug) DebugPrintMatrix(&M, "M");
		// M = squeeze(M) - vals.R; // find difference in colors between a pixel color in the output image and its color in the input image when projected there with each of the disparity proposals.  Note that M's color values are colors from the input image, but the pixels are ordered as in the output image, so they correspond to those colors in vals.R
		//if (debug) DebugPrintMatrix(&M, "M");
		M -= Runk_stacked;
		//if (debug) DebugPrintMatrix(&M, "M");
		// IA = reshape(cast(scale_factor * vals.ephoto(M), class(Kinf)), tp, 2); // photoconsistency is found by the application of the ephoto function to the pixel color differences to determine energy values
		FuseProposals_ePhoto(&M, &X, &ePhoto_scratch);
		//if (debug) DebugPrintMatrix(&X, "X");
		X = X * scale_factor_;
		IA = EigenMatlab::cwiseRound(&X); //IA = X2.cast<int>(); // this truncates rather than rounds
		//if (debug) DebugPrintMatrix(&IA, "IA");

		// since visibility==true, set up photoconsistency edges; // E is a matrix of ints that represents a 4D matrix of sizes num_unknown_pixels_out_ x 4 x 2 x num_in; note that corresponding EI represents unknown pixels with used pixel indices to be consistent with other matrices represented here
		E.block(1, num_unknown_pixels_out_*(img_num * 2), 1, num_unknown_pixels_out_) = IA.block(0, 0, num_unknown_pixels_out_, 1).transpose(); // Dcurr photoconsistency costs in the second row of E
		E.block(3, num_unknown_pixels_out_*(img_num * 2 + 1), 1, num_unknown_pixels_out_) = IA.block(num_unknown_pixels_out_, 0, num_unknown_pixels_out_, 1).transpose(); // Dnew photoconsistency costs in the fourth row of E
		//if (debug) DebugPrintMatrix(&E, "E");

		// photoconsistency costs are based on the color error if the pixel is visible, and are occl_cost_ if the pixel is not visible; so update E to take into account label segmentation and mapping so that if a pixel projects to an unassigned label in the input image (and if another label is assigned for that input image as the mapping, meaning we have some info for it to compare against), then it's considered not visible and that's reflected in photoconsistency costs in E by setting it to occl_cost_
		if (label_mappings.size() > 0)
			UpdatePhotoconsistencyVisibility(cid, img_num, label_mappings[cid], &WC, &T, &E);
		//if (debug) DebugPrintMatrix(&E, "E");

		// Find interactions (i.e. occlusions between the two disparity proposals - determines whether there are occlusions and, if so, which proposal is closer for that pixel pair)
		// T(:,3) = T(:,3) ./ WC(:,4); // dividing by disparity proposals is equivalent to multiplying by depth - wind up with data structure that has columns: screen space x in input image, screen space y in input image, camera space depth in reference image
		T.block(0, 2, T.rows(), 1) = T.block(0, 2, T.rows(), 1).cwiseProduct(WC3);
		//if (debug) DebugPrintMatrix(&T, "T");
		// [T M] = sortrows(T);
		Tunsorted = T;
		igl::sortrows(Tunsorted, true, T, IM); // igl method for sorting rows - sorts rows by first column, then second, and so on (in this case, first column has X values, second Y, etc); returns values in IM that are all 0-indexed; T = Tunsorted(IM,:).  This allows us to use the faster FindInterations() algorithm we have implemented here.  Then will have to unsort to original order again afterward.
		//if (debug) DebugPrintMatrix(&Tunsorted, "Tunsorted");

		// N = find_interactions(T, 0.5);
		Eigen::Matrix<unsigned int, 2, Dynamic> Noccl; // different size every loop
		FindInteractions(&T, 0.5, &Noccl); // Noccl becomes a list of pairs of indices for occluding/occluded pixel pairs.  First row gets index of occluding pixel and second gets index of occluded pixel.  Decides by finding pixels that project to same point in input image and comparing their depths in the refernce image camera space.
		//if (debug) DebugPrintMatrix(&Noccl, "Noccl");

		// unsort interactions - were computed with sorted rows to enable using the fast FindInteractions algorithm, but now must be reordered back to original order; returned by FindInteractions() as compact unknown pixel indices once reordered (unsorted) properly here; doesn't really "unsort" the order - rather, changes the indices to reflect original ordering of unknown pixel indices
		// N = M(N);
		Eigen::Matrix<int, 2, Dynamic> Noccl_unsorted = EigenMatlab::AccessByIndices(&IM, &Noccl);
		Noccl.resize(2, 0);
		//if (debug) DebugPrintMatrix(&Noccl_unsorted, "Noccl_unsorted");

		// convert unknown pixel indices in Noccl_unsorted to used pixel indices
		Eigen::Matrix<int, 2, Dynamic> Noccl_unsorted_full = sd_->MapUnknownToFullIndices(cid_out_, &Noccl_unsorted, 2);
		Noccl_unsorted.resize(2, 0);
		//if (debug) DebugPrintMatrix(&Noccl_unsorted_full, "Noccl_unsorted_full");
		Eigen::Matrix<int, 2, Dynamic> Noccl_unsorted_used = sd_->MapFulltoUsedIndices(cid_out_, &Noccl_unsorted_full, 2);
		Noccl_unsorted_full.resize(2, 0);
		//if (debug) DebugPrintMatrix(&Noccl_unsorted_used, "Noccl_unsorted_used");

		// remove interactions between the same node - find columns with index pairs that are num_pixels apart and remove those columns from the list in Noccl_unsorted; since it's hard to simply resize Noccl_unsorted, must create a new matrix of the correct size and copy retained column data over; if the absolute difference in indices for an interaction is equal to the number of pixels in the screen, it's a pixel interacting with itself, so cull these from the list in Noccl_unsorted
		// N = uint32(N(:,abs(diff(N))~=tp));
		Eigen::Matrix<int, 1, Dynamic> Noccl_unsorted_diff(1, Noccl_unsorted_used.cols()); // different size every loop; make it an int so can have negatives of which we then take the abs
		Noccl_unsorted_diff = Noccl_unsorted_used.row(1) - Noccl_unsorted_used.row(0);
		Noccl_unsorted_diff = Noccl_unsorted_diff.cwiseAbs();
		//if (debug) DebugPrintMatrix(&Noccl_unsorted_diff, "Noccl_unsorted_diff");
		Eigen::Matrix<bool, 1, Dynamic> Noccl_keep = Noccl_unsorted_diff.array() != num_used_pixels_out_; // different size every loop; we're working in compact used pixel indices at this point
		Noccl_unsorted_diff.resize(1, 0);
		//if (debug) DebugPrintMatrix(&Noccl_keep, "Noccl_keep");
		Eigen::Matrix<unsigned int, 2, Dynamic> Noccl_cleaned = EigenMatlab::TruncateByBooleansColumns(&Noccl_unsorted_used, &Noccl_keep).cast<unsigned int>();
		Noccl_keep.resize(1, 0);
		Noccl_unsorted_used.resize(2, 0);
		//if (debug) DebugPrintMatrix(&Noccl_cleaned, "Noccl_cleaned");

		// add the pixel interactions to the graph; these interactions are between a pixel p in the reference image and another pixel r in the reference image where either of the two proposal depths for p occludes r at one of the two proposal depths.  So the index for p is its position in the reference image (compact used pixel index) and for r depends on whether its at the depth for Dcurr or Dnew.  The edge goes from p to Vi(r,d0) or Vi(r,d1) where V is the code for a visibility node, i is the input image, d0 is the depth from Dcurr and d1 is the depth from Dnew.  Energies in TE will be set so that if the Dcurr depth for p generated the occlusion, the second row gets Kinf_, and if the Dnew depth for p generated the occlusion, the fourth row gets Kinf_.
		// M = N(1,:) >= tp; // edit from > tp to >= tp since we're using 0-indexed indices rather than 1-indexed
		Eigen::Matrix<bool, 1, Dynamic> Moccl_gt = Noccl_cleaned.row(0).array() >= num_used_pixels_out_; // different size every loop; need to identify index numbers from second proposal Dnew so that can subtract num_pixels_out_ from their indices to return them to the correct index range for the image; we're working in compact used pixel indices at this point
		// TEI = [TEI [N(1,:)-uint32(tp*M); (tp*2*a-tp)+N(2,:)]];
		//if (debug) DebugPrintMatrix(&Moccl_gt, "Moccl_gt");
		Eigen::Matrix<unsigned int, 2, Dynamic> TEI_ext(2, Noccl_cleaned.cols()); // different size every loop
		TEI_ext.row(0) = Noccl_cleaned.row(0) - (unsigned int)num_used_pixels_out_ * Moccl_gt.cast<unsigned int>(); // for those values that are >= num_pixels in size, subtract num_pixels (these are the second proposal's index numbers that need to be corrected since are currently pushed out by num_pixels)
		TEI_ext.row(1) = Noccl_cleaned.row(1).array() + (unsigned int)((num_used_pixels_out_ * 2 * (img_num + 1)) - num_used_pixels_out_); // img_num is 0-indexed, but must use a 1-indexed version here; must transform index values into range corresponding to appropriate input image, where there are num_used_pixels_out_ indices for each reference-input image pair, and the second row must start higher than the first row's indices, which are in the range [0, num_pixels_out_)
		//if (debug) DebugPrintMatrix(&TEI_ext, "TEI_ext");
		EigenMatlab::ConcatHorizontally(&TEI, &TEI_ext);
		TEI_ext.resize(2, 0);
		//if (debug) DebugPrintMatrix(&TEI, "TEI");
		
		// pair-wise visibility energy terms carry energy values of Kinf for an occluded pixel; Kinf is 1 greater than occl_cost_, so is largest possible cost
		// T = zeros(4, numel(M));
		Eigen::Matrix<int, 4, Dynamic> TE_ext(4, Moccl_gt.cols()); // different size every loop
		TE_ext.setZero();
		// T(2,~M) = Kinf; // Kinf for Dcurr proposal visibility energy if the Dnew index appeared first in the pair from FindInteractions(), and 0 for the Dnew proposal visibility energy (Dnew occludes Dcurr)
		TE_ext.row(1) = (Moccl_gt.array() == false).select(Eigen::Matrix<int, 1, Dynamic>::Constant(1, TE_ext.cols(), Kinf_), TE_ext.row(1)); // sets TE.row(1) coefficients to Kinf for positions where Moccl_gt coefficients == false
		// T(4,M) = Kinf; // Kinf for Dnew proposal visibility energy if the Dcurr index appeared first in the pair from FindInteractions(), and 0 for the Dcurr proposal visibility energy (Dcurr occludes Dnew)
		TE_ext.row(3) = (Moccl_gt.array()).select(Eigen::Matrix<int, 1, Dynamic>::Constant(1, TE_ext.cols(), Kinf_), TE_ext.row(3)); // sets TE.row(3) coefficients to Kinf for positions where Moccl_gt coefficients == true
		//if (debug) DebugPrintMatrix(&TE_ext, "TE_ext");
		// TE = [TE T];
		EigenMatlab::ConcatHorizontally(&TEinit, &TE_ext);
		Moccl_gt.resize(1, 0);
		TE_ext.resize(4, 0);
		//if (debug) DebugPrintMatrix(&TEinit, "TEinit");

		// ?? add pair-wise smoothness terms of Kinf_ where the depth distance between two adjacent pixels of the same segment (up/down, left/right, or diagonal) is greater than or equal to GLOBAL_MESH_EDGE_DISTANCE_MAX, and 0 everywhere else; indices in TEI_smooth are used pixel indices where the first n=num_used_pixels_[cid_out_] are for Dcurr and the second n are for Dnew
		

		if (GLOBAL_OPTIMIZATION_COMPRESS_GRAPH) {
			// determine the photoconsistency nodes which have no interactions
			// M = ones(tp, 2, class(Kinf));
			M_compressgraph.setOnes();
			// M(N(2, :)) = 0;
			Matrix<unsigned int, 1, Dynamic> Noccl_cleaned_row1 = Noccl_cleaned.row(1);
			EigenMatlab::AssignByIndices(&M_compressgraph, &Noccl_cleaned_row1, 0);
			//if (debug) DebugPrintMatrix(&M_compressgraph, "M_compressgraph");
			// add those photoconsistency terms to the unaries
			// U = U + M.*IA; // note that .* is a component-wise multiplication
			// IA is currently of size num_unknown_pixels_out_*2 x 1; need it to be of size num_used_pixels_out_ x 2 (IA_compressgraph), which is tricky
			IA_compressgraph_unk.block(0, 0, num_unknown_pixels_out_, 1) = IA.block(0, 0, num_unknown_pixels_out_, 1);
			IA_compressgraph_unk.block(0, 1, num_unknown_pixels_out_, 1) = IA.block(num_unknown_pixels_out_, 0, num_unknown_pixels_out_, 1);
			//if (debug) DebugPrintMatrix(&IA_compressgraph_unk, "IA_compressgraph_unk");
			IA_compressgraph_used = sd_->ExpandUnknownToUsedSize(cid_out_, &IA_compressgraph_unk, 1);
			//if (debug) DebugPrintMatrix(&IA_compressgraph_used, "IA_compressgraph_used");
			U_compressgraph = U_compressgraph + M_compressgraph.cwiseProduct(IA_compressgraph_used);
			//if (debug) DebugPrintMatrix(&U_compressgraph, "U_compressgraph");
		}
		Noccl_cleaned.resize(2, 0);
		
		img_num++;
	}
	// clear temporary matrices from memory
	WC.resize(0, 4);
	WC3.resize(0, 1);
	A.resize(0, 3);
	M.resize(0, 3);
	X.resize(0, 1);
	Y.resize(0, 1);
	IA.resize(0, 1);
	T.resize(0, 3);
	Tunsorted.resize(0, 3);
	IM.resize(0, 1); // typedef Matrix< int , Dynamic , 1> VectorXi
	Runk_stacked.resize(0, 3);
	IA_compressgraph_unk.resize(0, 2);
	IA_compressgraph_used.resize(0, 2);

	if (debug) cout << "completed main variable assignment loop" << endl;

	Matrix<unsigned int, 2, Dynamic> EI_ = EI;

	Eigen::Matrix<int, 2, Dynamic> U;
	Matrix<unsigned int, 2, Dynamic> TEI_N = TEI; // used to potentially modify through QPBO_CompressGraph, then definitely extend E; don't do with TEI because need to use it later unmodified by QPBO_CompressGraph
	Matrix<int, 4, Dynamic> TEinit_T = TEinit; // used to potentially modify through QPBO_CompressGraph, then definitely extend EI_; don't do with TEinit because need to use it later unmodified by QPBO_CompressGraph
	if (GLOBAL_OPTIMIZATION_COMPRESS_GRAPH) {
		U = U_compressgraph.transpose();
		QPBO_CompressGraph(&U, &E, &EI, &TEinit_T, &TEI_N, num_in_, &EI_); // the unary and pairwise energies as they stand are entirely correct, i.e. will give the correct labelling. However, it can be compressed into a smaller but equivalent graph, which will be faster to solve, by removing superfluous nodes and edges.
	}
	else {
		U.resize(2, num_used_pixels_out_ + num_used_pixels_out_ * 2 * num_in_); // if try to use num_used_pixels_out_ instead, even for just one of the two places here, crashes on exceeding index bounds while adding pairwise terms
		U.setZero();
	}
	M_compressgraph.resize(0, 2);
	U_compressgraph.resize(0, 2);

	// concatenate data and visibility edges
	EigenMatlab::ConcatHorizontally(&E, &TEinit_T);
	EigenMatlab::ConcatHorizontally(&EI_, &TEI_N);
	TEinit_T.resize(4, 0);
	TEI_N.resize(2, 0);

	// changes TE to a single-row matrix of TE-pre.cols() width that has integer boolean values of 1 where the 2nd row of TE-pre does not equal 0, and 0 everywhere else
	Matrix<int, 1, Dynamic> TE(1, TEinit.cols()); // TEinit's columns are added in the previous loop, so don't know a priori what number will exist
	TE.setZero();
	TE = (TEinit.row(1).array() != 0).select(Eigen::Matrix<int, 1, Dynamic>::Ones(1, TE.cols()), TE); // sets TE coefficients to 1 for positions where TEinit.row(1) coefficients != 0

	// time data term evaluation
	t = (double)getTickCount() - t_last;
	t_last = (double)getTickCount();
	sd_->timings_data_term_eval_(0, iter) = t*1000. / getTickFrequency(); // set to ms
	cout << "Optimization::FuseProposals() data running time = " << t*1000. / getTickFrequency() << " ms" << endl;

	// compute SE, an 8xR triple clique energy table, each column containing the energies[E000 E001 E010 E011 E100 E101 E110 E111] for a given triple clique.
	Matrix<double, Dynamic, 1> Dcurrtrunc_used = sd_->ContractFullToUsedSize(cid_out_, Dcurr);
	Matrix<double, Dynamic, 1> Dnewtrunc_used = sd_->ContractFullToUsedSize(cid_out_, Dnew);
	Matrix<int, 8, Dynamic> SE;
	ComputeSE(&Dcurrtrunc_used, &Dnewtrunc_used, scale_factor_, &SE);
	Dcurrtrunc_used.resize(0, 1);
	Dnewtrunc_used.resize(0, 1);
	
	// time smoothness term evaluation
	t = (double)getTickCount() - t_last;
	t_last = (double)getTickCount();
	sd_->timings_smoothness_term_eval_(0, iter) = t*1000. / getTickFrequency(); // set to ms
	cout << "Optimization::FuseProposals() smoothness running time = " << t*1000. / getTickFrequency() << " ms" << endl;

	
	if (debug) {
		//cout << "num_pixels_out_ " << num_pixels_out_  << endl;
		//cout << "num_used_pixels_out_ " << num_used_pixels_out_ << endl;
		//cout << "num_unknown_pixels_out_ " << num_unknown_pixels_out_ << endl;
		//DebugPrintMatrix(&EI_, "EI_");
		//DebugPrintMatrix(&E, "E");
		//DebugPrintMatrix(&SEI_used_, "SEI_used_");
		//DebugPrintMatrix(&SE, "SE");
	}
	

	// fuse the two labellings, using contract and/or improve if desired
	Matrix<int, Dynamic, 1> Mqpbo_used(num_used_pixels_out_, 1); // M
	try {
		QPBO_eval(&U, &EI_, &E, &SEI_used_, &SE, &Mqpbo_used, iter, include_smoothness_terms);
	}
	catch (int e) { // error probably due to probe failure
		sd_->count_unlabelled_vals_(0, iter) = 0;
		sd_->count_unlabelled_regions_(0, iter) = 0;
		sd_->count_unlabelled_after_QPBOP_(0, iter) = 0;
		Mqpbo_used = Matrix<int, Dynamic, 1>::Zero(num_used_pixels_out_, 1);
	}

	//if (debug) DebugPrintMatrix(&Mqpbo_used, "Mqpbo_used");
	
	// calculate energies
	Dswap->setConstant(false);
	Matrix<int, 4, Dynamic> Etrunc(4, EI.cols());
	Etrunc = E.block(0, 0, 4, EI.cols());
	Matrix<bool, Dynamic, Dynamic> Vout_used;
	Matrix<int, Dynamic, 1> Uout_used(num_used_pixels_out_, 1);
	Matrix<int, Dynamic, 1> Eout_used;
	Matrix<int, Dynamic, 1> SEout_used;
	Matrix<bool, Dynamic, 1> Dswap_used;
	
	if (sd_->count_unlabelled_vals_(0, iter) > 0) { // assume QBPO-R, so vals.improve==2
		int num_regions = sd_->count_unlabelled_regions_(0, iter);
		Matrix<int, Dynamic, 1> Mqpbo_used_cl_out(num_used_pixels_out_, 1);
		QPBO_ChooseLabels(&Mqpbo_used, &U, &Etrunc, &EI, &SE, &SEI_used_, &TE, &TEI, num_in_, &Mqpbo_used_cl_out, num_regions, &Uout_used, &Eout_used, &SEout_used, &Vout_used);
		sd_->count_unlabelled_regions_(0, iter) = num_regions;
		energy_val = Uout_used.sum() + Eout_used.sum() + SEout_used.sum();
		Dswap_used = Mqpbo_used_cl_out.array() > 0;
		Mqpbo_used = Mqpbo_used_cl_out;
	}
	else {
		Dswap_used = Mqpbo_used.array() > 0;
		QPBO_CalcVisEnergy(&Dswap_used, &U, &Etrunc, &EI, &SE, &SEI_used_, &TE, &TEI, num_in_, &Uout_used, &Eout_used, &SEout_used, &Vout_used);
		energy_val = Uout_used.sum() + Eout_used.sum() + SEout_used.sum();
	}

	if (debug) {
		//DebugPrintMatrix(&Uout_used, "Uout_used");
		//DebugPrintMatrix(&Eout_used, "Eout_used");
		//DebugPrintMatrix(&SEout_used, "SEout_used");
		//DebugPrintMatrix(&Vout_used, "Vout_used");
	}

	(*energy)(iter, 0) = energy_val;
	sd_->count_updated_vals_(0, iter) = Dswap_used.sum();

	// copy unknown pixel results from Dswap_used into Dswap
	//(*Dswap) = sd_->ExpandUsedToFullSize(cid_out_, &Dswap_used); // don't do it this way because only want to copy over unknown Dswap_used values (not all Dswap_used values since are for all used pixels) and set rest to false
	Dswap->setConstant(false);
	bool *pM = sd_->masks_[cid_out_].data();
	bool *pMunk = sd_->masks_unknowns_[cid_out_].data();
	bool *pDS = Dswap->data();
	bool *pDSused = Dswap_used.data();
	for (int i = 0; i < num_pixels_out_; i++) {
		if (*pMunk++) *pDS = *pDSused; // copy over unknown pixel fuse results
		if (*pM++) pDSused++; // advance through used pixel fuse results
		pDS++; // advance through all pixel fuse results
	}
	
	Matrix<bool, Dynamic, Dynamic> Vout = sd_->ExpandUsedToFullSize(cid_out_, &Vout_used, 2);
	Matrix<int, Dynamic, 1> Uout = sd_->ExpandUsedToFullSize(cid_out_, &Uout_used);

	//if (debug) DebugPrintMatrix(Dswap, "Dswap");

	// time optimization term evaluation
	t = (double)getTickCount() - t_last;
	t_last = (double)getTickCount();
	sd_->timings_qpbo_fuse_time_(0, iter) = t*1000. / getTickFrequency(); // set to ms
	cout << "Optimization::FuseProposals() optimization running time = " << t*1000. / getTickFrequency() << " ms" << endl;

	// total time for iteration
	t = (double)getTickCount() - t_start; // time from start of optimization
	sd_->timings_iteration_time_(0, iter) = t*1000. / getTickFrequency(); // set to ms
	cout << "Optimization::FuseProposals() iteration running time = " << t*1000. / getTickFrequency() << " ms" << endl;


	if (GLOBAL_OPTIMIZATION_DEBUG_VIEW_PLOTS) {
		Eigen::Matrix<bool, Dynamic, Dynamic> Visibilities(num_pixels_out_, sd_->nums_disps_[cid_out_]); // matrix of output pixel visibilities by disparity

		// generate output visibilities
		// T = (tp * N) + (1:tp)';
		// for b = 1:num_in
		//		V(1:tp, b) = V(T);
		//		T = T + 2 * tp;
		// end
		Matrix<int, Dynamic, 1> Tvis(num_pixels_out_, 1);
		Tvis = num_pixels_out_ * Dswap->cast<int>();
		Matrix<int, Dynamic, 1> pix_range(num_pixels_out_, 1);
		pix_range.setLinSpaced(num_pixels_out_, 0, num_pixels_out_ - 1);
		Tvis = Tvis + pix_range;
		pix_range.resize(0, 1);
		//Tvis = Tvis + sd_->Iunknowns_[cid_out_].cast<int>(); // Iunknowns_[cid_out_] is of type unsigned int
		for (int b = 0; b < num_in_; b++) {
			Visibilities.block(0, b, num_pixels_out_, 1) = EigenMatlab::AccessByIndices(&Vout, &Tvis);
			Tvis = Tvis.array() + 2 * num_pixels_out_;
		}

		// V(tp+1:end,:) = [];
		Matrix<bool, Dynamic, Dynamic> tmpVout = Vout;
		Vout.resize(num_pixels_out_, tmpVout.cols());
		Vout = tmpVout.block(0, 0, num_pixels_out_, tmpVout.cols());
		tmpVout.resize(0, 0);

		// display output figures
		Matrix<unsigned int, 2, Dynamic> EI_full = MapEIUsedToFull(&EI);
		UpdateOutputFigures(Dcurr, Dnew, Dswap, &Visibilities, &EI_full, &Uout, &Eout_used, &SEout_used, &Vout); // use EI with used indices to match Eout_used
	}
}

void Optimization::QPBO_CompressGraph(Matrix<int, 2, Dynamic> *U, Matrix<int, 4, Dynamic> *E, Matrix<unsigned int, 2, Dynamic> *EI, Matrix<int, 4, Dynamic> *TE, Matrix<unsigned int, 2, Dynamic> *TEI, int num_in, Matrix<unsigned int, 2, Dynamic> *EI_) {

	// count the number of interactions per input sample
	// SE = accum(TEI(2, :)', (1:size(TEI, 2))', [tp + tp * 2 * num_in 1], @num_first); // if a pixel in an input image appears more than once, use the negative of the number of times it appears; otherwise use the position pixel index
	Matrix<unsigned int, Dynamic, 1> TEI_row1_trans = TEI->row(1).transpose();
	Matrix<int, Dynamic, 1> vals(TEI->cols(), 1);
	vals.setLinSpaced(vals.rows(), 0, vals.rows() - 1);
	Matrix<int, Dynamic, 1> SE_used = EigenMatlab::Accumarray_NumFirst(&TEI_row1_trans, &vals, num_used_pixels_out_ + num_used_pixels_out_ * 2 * num_in, 1); // TEI's indices are in used pixel space even though the number of them present is a factor of the total number of unknown pixels
	TEI_row1_trans.resize(0, 1);
	vals.resize(0, 1);
	// SE = SE(tp + 1:end) // truncate to interactions after the first input image
	Matrix<int, Dynamic, 1> SE_trunc_used = SE_used.block(num_used_pixels_out_, 0, SE_used.rows() - num_used_pixels_out_, 1);
	SE_used.resize(0, 1);

	// switch to unknown numbers of coefficients
	Matrix<int, Dynamic, 1> SE_trunc_unk = sd_->ContractUsedToUnknownSize(cid_out_, &SE_trunc_used, 2 * num_in);
	
	// remove single interactions, attaching the photoconsitency edge directly to the interacting pixel
	// M = find(SE > 0); // find the returns the indices of non-zero values; finding the indices of pixels with single interactions
	Matrix<unsigned int, Dynamic, 1> M_tmp(SE_trunc_unk.rows(), 1);
	unsigned int *pM_tmp = M_tmp.data();
	int *pSE_trunc_unk = SE_trunc_unk.data();
	unsigned int i = 0;
	int j = 0;
	for (int r = 0; r < SE_trunc_unk.rows(); r++) {
		if (*pSE_trunc_unk > 0) {
			*pM_tmp = i;
			j++;
			pM_tmp++;
		}
		pSE_trunc_unk++;
		i++;
	}
	Matrix<unsigned int, Dynamic, 1> M(j, 1);
	M = M_tmp.block(0, 0, j, 1);
	M_tmp.resize(0, 1);
	// L = SE(M); // L becomes the pixel index of the pixels with single interations
	Matrix<int, 1, Dynamic> L = EigenMatlab::AccessByIndices(&SE_trunc_unk, &M); // note that L should be a column vector but is a row vector here
	L = L.array() - 1; // to adjust back to 0-indexing since EigenMatlab::Accumarray_NumFirst() incremented single indices by 1
	// EI(2, M) = TEI(1, L); // L is indices from SE that are greater than 0, which serve as indices into TEI columns
	Matrix<unsigned int, 1, Dynamic> TEI_row0 = TEI->row(0);
	Matrix<unsigned int, 1, Dynamic> TEI_row0_ofL = EigenMatlab::AccessByIndicesColumns(&TEI_row0, &L);
	TEI_row0.resize(1, 0);
	Matrix<unsigned int, 1, Dynamic> EI_row1 = EI->row(1);
	EigenMatlab::AssignByIndices(&EI_row1, &M, &TEI_row0_ofL);
	TEI_row0_ofL.resize(1, 0);
	EI->row(1) = EI_row1;
	EI_row1.resize(1, 0);
	// M = M(TE(4, L)~= 0);
	Matrix<int, 1, Dynamic> TE_row3 = TE->row(3);
	Matrix<int, 1, Dynamic> TE_row3_ofL = EigenMatlab::AccessByIndicesColumns(&TE_row3, &L);
	TE_row3.resize(1, 0);
	Matrix<bool, 1, Dynamic> TE_row3_ofL_notzero = TE_row3_ofL.array() != 0;
	TE_row3_ofL.resize(1, 0);
	Matrix<unsigned int, Dynamic, 1> M2 = EigenMatlab::TruncateByBooleans(&M, &TE_row3_ofL_notzero);
	TE_row3_ofL_notzero.resize(1, 0);
	M.resize(0, 1);
	// E(:, M) = E([2 1 4 3], M); // permutes rows of E for those columns indexed by M
	Matrix<int, 4, Dynamic> E_perm(4, E->cols());
	E_perm.row(0) = E->row(1);
	E_perm.row(1) = E->row(0);
	E_perm.row(2) = E->row(3);
	E_perm.row(3) = E->row(2);
	Matrix<int, 4, Dynamic> E_perm_ofM = EigenMatlab::AccessByIndicesColumns(&E_perm, &M2);
	E_perm.resize(4, 0);
	EigenMatlab::AssignByIndicesCols(E, &M2, &E_perm_ofM);
	M2.resize(0, 1);
	E_perm_ofM.resize(4, 0);
	// TEI(:, L) = [];
	Matrix<unsigned int, 2, Dynamic> TEI_new = EigenMatlab::TruncateByIndicesColumns(TEI, &L);
	TEI->resize(2, 0);
	(*TEI) = TEI_new;
	TEI_new.resize(2, 0);
	// TE(:, L) = [];
	Matrix<int, 4, Dynamic> TE_new = EigenMatlab::TruncateByIndicesColumns(TE, &L);
	TE->resize(4, 0);
	(*TE) = TE_new;
	TE_new.resize(4, 0);
	L.resize(1, 0);
	
	// remove the superfluous edges - photoconsistency edges with no interactions, that have already been incorporated into the unary term
	// M = SE ~= 0;
	Matrix<bool, Dynamic, 1> M3 = SE_trunc_unk.array() != 0;
	Matrix<bool, 1, Dynamic> M4 = M3.transpose();
	M3.resize(0, 1);
	// E = E(:, M);
	Matrix<int, 4, Dynamic> E_tmp = EigenMatlab::TruncateByBooleansColumns(E, &M4);
	(*E) = E_tmp;
	E_tmp.resize(4, 0);
	// EI = EI(:, M);
	Matrix<unsigned int, 2, Dynamic> EI_tmp = EigenMatlab::TruncateByBooleansColumns(EI, &M4);
	(*EI) = EI_tmp;
	EI_tmp.resize(2, 0);
	M4.resize(1, 0);

	// compress the node indices
	// M = zeros(tp + 2 * tp*num_in, 1, 'uint32');
	Matrix<unsigned int, Dynamic, 1> M5(num_used_pixels_out_ + 2 * num_used_pixels_out_ * num_in, 1);
	M5.setZero();
	// SE = SE < 0;
	Matrix<bool, Dynamic, 1> SE2 = SE_trunc_used.array() < 0;
	// L = sum(SE);
	int L_count_used = SE2.count();

	// M([true(tp, 1); SE]) = uint32(1) :uint32(L + tp);
	Matrix<bool, Dynamic, 1> M5_bools(num_used_pixels_out_, 1);
	M5_bools.setConstant(true);
	EigenMatlab::ConcatVertically(&M5_bools, &SE2);
	Matrix<unsigned int, Dynamic, 1> M5_vals(L_count_used + num_used_pixels_out_, 1);
	M5_vals.setLinSpaced(L_count_used + num_used_pixels_out_, 0, L_count_used + num_used_pixels_out_ - 1);
	EigenMatlab::AssignByTruncatedBooleans(&M5, &M5_bools, &M5_vals);
	SE2.resize(0, 1);
	M5_bools.resize(0, 1);
	M5_vals.resize(0, 1);
	SE_trunc_unk.resize(0, 1);
	SE_trunc_used.resize(0, 1);

	// EI_ = EI;
	(*EI_) = (*EI);
	// EI_(2, :) = M(EI(2, :));
	EI_row1 = EI->row(1); // EI_row1 was defined earlier, then erased
	EI_->row(1) = EigenMatlab::AccessByIndices(&M5, &EI_row1);
	// TEI(2, :) = M(TEI(2, :));
	Matrix<unsigned int, 1, Dynamic> TEI_row1 = TEI->row(1);
	TEI->row(1) = EigenMatlab::AccessByIndices(&M5, &TEI_row1);
	M5.resize(0, 1);
	// U = [U zeros(2, L, class(U))];
	Matrix<int, 2, Dynamic> U_ext(2, L_count_used);
	EigenMatlab::ConcatHorizontally(U, &U_ext);
}

// the indices in EI are used pixel indices, but for each input image, the second row's index (each image comprises 2*num_used_pixels_out_ columns) is shifted by (num_used_pixels_out_ + (num_used_pixels_out_ * 2 * img_num)) where img_num is the 0-indexed index of the image we're on in order of those already added to EI
// therefore, indices must first be transformed back to the range [0, num_used_pixels_out_), then mapped to full pixel indices, then transformed back so that each image in the second row (each image still comprises 2*num_used_pixels_out_ columns) is shifted by (num_pixels_out_ + (num_pixels_out_ * 2 * img_num))
// to make matters more complicated, each image's pixels indices are each in two parts: one set of num_used_pixels_out_ columns corresponding to Dcurr and another num_used_pixels_out_ columns corresponding to Dnew, and the Dnew columns are shifted by an additional num_used_pixels_out_ columns pixels
// so basically, the second row starts at num_used_pixels_out_, and after every num_used_pixels_out_ columns num_used_pixels_out_ is added to the indices in the next set
Matrix<unsigned int, 2, Dynamic> Optimization::MapEIUsedToFull(Matrix<unsigned int, 2, Dynamic> *EI) {
	Matrix<unsigned int, 2, Dynamic> EI_full(2, EI->cols());
	
	// map first row
	Matrix<unsigned int, 1, Dynamic> EI_row0(1, EI->cols());
	EI_row0 = EI->row(0);
	EI_full.row(0) = sd_->MapUsedToFullIndices(cid_out_, &EI_row0);
	EI_row0.resize(1, 0);
	
	// find input image adjustments to second row
	Matrix<unsigned int, 1, Dynamic> num_adj(1, EI->cols());
	num_adj = EI->row(1).array() / num_used_pixels_out_;
	Matrix<int, 1, Dynamic> num_adj_floor = EigenMatlab::cwiseFloor(&num_adj);
	Matrix<unsigned int, 1, Dynamic> num_adj_floor_used = num_adj_floor.cast<unsigned int>() * num_used_pixels_out_;
	
	// map second row; blocks of pixels by image are of size num_unknown_pixels_out_, but indices are in compact used pixel space, not compact unknown pixel space
	Matrix<unsigned int, 1, Dynamic> EI_row1_adj = EI->row(1) - num_adj_floor_used;
	Matrix<unsigned int, 1, Dynamic> EI_row1_adj_full = sd_->MapUsedToFullIndices(cid_out_, &EI_row1_adj);
	EI_full.row(1) = EI_row1_adj_full + (num_adj_floor.cast<unsigned int>() * num_pixels_out_);

	return EI_full;
}

// updates Mout, num_regions, Uout, Eout, SEout, Vout
// num_regions looks to be the number of unknown regions, labeled with negative numbers in M (as opposed to known 0 or 1 values)
void Optimization::QPBO_ChooseLabels(Matrix<int, Dynamic, 1> *M, Matrix<int, 2, Dynamic> *U, Matrix<int, 4, Dynamic> *E, Matrix<unsigned int, 2, Dynamic> *EI, Matrix<int, 8, Dynamic> *SE, Matrix<unsigned int, 3, Dynamic> *SEI, Matrix<int, 1, Dynamic> *TE, Matrix<unsigned int, 2, Dynamic> *TEI, int num_in, Matrix<int, Dynamic, 1> *Mout, int &num_regions, Matrix<int, Dynamic, 1> *Uout, Matrix<int, Dynamic, 1> *Eout, Matrix<int, Dynamic, 1> *SEout, Matrix<bool, Dynamic, Dynamic> *Vout) {
	assert(M->rows() == num_used_pixels_out_);
	assert(Mout->rows() == num_used_pixels_out_);
	//assert(U->cols() == (num_used_pixels_out_ + (num_used_pixels_out_ * 2 * num_in))); // not necessarily true when compress the graph
	//assert(EI->cols() == (num_unknown_pixels_out_ * 2 * num_in)); // not necessarily true when compress the graph
	assert(TE->cols() == TEI->cols());
	assert(Uout->rows() == num_used_pixels_out_);

	bool debug = false;
	bool timing = true;
	double t;
	if (timing) t = (double)getTickCount();

	// calculate visibilities and regions assuming unlabelled pixels are set to 0 then 1.
	Matrix<bool, Dynamic, 1> L_1(M->rows(), 1);
	L_1.setConstant(false);
	L_1 = (M->array() == 1).select(Matrix<int, Dynamic, 1>::Ones(M->rows(), 1), L_1).cast<bool>(); // sets L coefficients to 1 for positions where M coefficients == 1
	Matrix<bool, Dynamic, 1> L_2(M->rows(), 1);
	L_2.setConstant(false);
	L_2 = (M->array() != 0).select(Matrix<int, Dynamic, 1>::Ones(M->rows(), 1), L_2).cast<bool>(); // sets L coefficients to 1 for positions where M coefficients == 1
	Matrix<int, Dynamic, 1> U_1(num_used_pixels_out_, 1);
	Matrix<int, Dynamic, 1> E_1;
	Matrix<int, Dynamic, 1> SE_1;
	Matrix<bool, Dynamic, Dynamic> V_1(0, 0);
	Matrix<int, Dynamic, 1> U_2(num_used_pixels_out_, 1);
	Matrix<int, Dynamic, 1> E_2;
	Matrix<int, Dynamic, 1> SE_2;
	Matrix<bool, Dynamic, Dynamic> V_2(0, 0);
	QPBO_CalcVisEnergy(&L_1, U, E, EI, SE, SEI, TE, TEI, num_in, &U_1, &E_1, &SE_1, &V_1);
	QPBO_CalcVisEnergy(&L_2, U, E, EI, SE, SEI, TE, TEI, num_in, &U_2, &E_2, &SE_2, &V_2); // note that SE_2 winds up with some 0 coefficient values that didn't appear in the Matlab run - not sure whether that's an issue ... affects SE_2_2 below by giving it 0 coefficient values ****************************************
	num_regions = -1 * M->minCoeff();

	// adjust M by adding 1 to accommodate our 0-indexing instead of 1-indexing since otherwise the indices would be in the range [num_regions, 1] and we want [num_regions-1, 2] where 1 is Dcurr, 2 is Dnew, and <=0 is unlabelled so that [num_regions-1,0] corresponds to the negative of the region index
	Matrix<int, Dynamic, 1> Madj = M->array() + 1;

	// assume improve == 2 here (QPBO-R), so don't code the other branch
	// we want to do optimal splice
	double sz_1 = num_regions;
	double sz_2 = 1;

	// Merge strongly connected regions that are connected by visibility cliques

	//TEI2 = TEI(:, M(TEI(1, :))<=0); // making it <=0 instead of <0 to accommodate adjusted M
	Matrix<unsigned int, 1, Dynamic> TEI_row0 = TEI->block(0, 0, 1, TEI->cols());
	Matrix<int, 1, Dynamic> M_of_TEI_row0 = EigenMatlab::AccessByIndices(&Madj, &TEI_row0);
	TEI_row0.resize(1, 0);
	Matrix<bool, 1, Dynamic> M_of_TEI_row0_ltzero = M_of_TEI_row0.array() <= 0;
	Matrix<unsigned int, 2, Dynamic> TEI2 = EigenMatlab::TruncateByBooleansColumns(TEI, &M_of_TEI_row0_ltzero);
	M_of_TEI_row0_ltzero.resize(1, 0);

	// tp = numel(M);
	int tp = Madj.rows() * Madj.cols();

	// M = repmat(M, [1 + 2 * num_in 1]);
	Matrix<int, Dynamic, 1> Mrep = Madj.replicate((1 + 2 * num_in), 1);

	// M(TEI2(2, :)) = M(TEI2(1, :));
	Matrix<unsigned int, 1, Dynamic> TEI2_row0 = TEI2.block(0, 0, 1, TEI2.cols());
	Matrix<unsigned int, 1, Dynamic> TEI2_row1 = TEI2.block(1, 0, 1, TEI2.cols());
	Matrix<int, 1, Dynamic> M_of_TEI2_row0 = EigenMatlab::AccessByIndices(&Mrep, &TEI2_row0);
	EigenMatlab::AssignByIndices(&Mrep, &TEI2_row1, &M_of_TEI2_row0);
	TEI2_row0.resize(1, 0);
	TEI2_row1.resize(1, 0);
	M_of_TEI2_row0.resize(1, 0);

	// go through each region and determine whether a labelling of 1 or 0 gives a lower energy, starting with the visibility edges

	// EI2 = min(M(EI))';
	Matrix<int, 2, Dynamic> M_of_EI = EigenMatlab::AccessByIndices(&Mrep, EI);
	Matrix<int, Dynamic, 1> EI2 = M_of_EI.colwise().minCoeff().transpose();
	M_of_EI.resize(2, 0);

	// T = EI2 <= 0; // making it <=0 instead of <0 to accommodate adjusted M
	Matrix<bool, Dynamic, 1> T = EI2.array() <= 0;

	// E2 = E2(T) - E_(T);
	Matrix<int, Dynamic, 1> E_2_of_T = EigenMatlab::TruncateByBooleansRows(&E_2, &T);
	Matrix<int, Dynamic, 1> E_1_of_T = EigenMatlab::TruncateByBooleansRows(&E_1, &T);
	E_2 = E_2_of_T - E_1_of_T;
	E_2_of_T.resize(0, 1);
	E_1_of_T.resize(0, 1);

	// engy = accum(-EI2(T), E_2, sz);
	Matrix<int, Dynamic, 1> EI2_of_T = EigenMatlab::TruncateByBooleansRows(&EI2, &T);
	T.resize(0, 1);
	EI2.resize(0, 1);
	EI2_of_T = -1 * EI2_of_T;
	Matrix<int, Dynamic, Dynamic> engy = EigenMatlab::Accumarray(&EI2_of_T, &E_2, num_regions, 1);
	EI2_of_T.resize(0, 1);
	E_2.resize(0, 1);

	// M2 = Mrep(1:tp);
	Matrix<int, Dynamic, 1> M2 = Mrep.block(0, 0, num_used_pixels_out_, 1);
	Mrep.resize(0, 1);

	// go through each region and determine whether a labelling of 1 or 0 gives a lower energy

	Matrix<int, 3, Dynamic> SEI2 = EigenMatlab::AccessByIndices(&M2, SEI); // SEI2 = M2(SEI);

	// T2 = any(SEI2 <= 0);  // making it <=0 instead of <0 to accommodate adjusted M
	Matrix<bool, 3, Dynamic> SEI2_ltzero = SEI2.array() <= 0;
	Matrix<bool, 1, Dynamic> T2 = SEI2_ltzero.colwise().any();
	SEI2_ltzero.resize(3, 0);

	Matrix<int, 3, Dynamic> SEI2_of_T2 = EigenMatlab::TruncateByBooleansColumns(&SEI2, &T2); // SEI2 = SEI2(:, T2);

	// SE_2 = SE_2(T2) - SE_(T2);
	Matrix<bool, Dynamic, 1> T2_trans = T2.transpose();
	Matrix<int, Dynamic, 1> SE_1_of_T2 = EigenMatlab::TruncateByBooleansRows(&SE_1, &T2_trans);
	Matrix<int, Dynamic, 1> SE_2_of_T2 = EigenMatlab::TruncateByBooleansRows(&SE_2, &T2_trans);
	Matrix<int, Dynamic, 1> SE_2_2 = SE_2_of_T2 - SE_1_of_T2;
	SE_1_of_T2.resize(0, 1);
	SE_2_of_T2.resize(0, 1);
	T2.resize(1, 0);
	T2_trans.resize(0, 1);
	SE_2.resize(0, 1);

	// T2_2 = -min(SEI2_of_T2);
	Matrix<int, 1, Dynamic> T2_2 = -1 * SEI2_of_T2.colwise().minCoeff();
	SEI2_of_T2.resize(3, 0);

	// engy = engy + accum(T2_2(:), SE_2_2, sz);
	Matrix<int, Dynamic, 1> T2_2_colon = T2_2.transpose();
	T2_2.resize(1, 0);
	engy = engy + EigenMatlab::Accumarray(&T2_2_colon, &SE_2_2, num_regions, 1);
	T2_2_colon.resize(0, 1);
	SE_2_2.resize(0, 1);

	Matrix<bool, Dynamic, 1> T2_3 = M2.array() <= 0; // T2_3 = M2 <= 0;  // making it <=0 instead of <0 to accommodate adjusted M

	// U_2 = U_2(T2_3) - U_(T2_3);
	Matrix<int, Dynamic, 1> U_1_of_T = EigenMatlab::TruncateByBooleansRows(&U_1, &T2_3);
	Matrix<int, Dynamic, 1> U_2_of_T = EigenMatlab::TruncateByBooleansRows(&U_2, &T2_3);
	U_2.resize(0, 1);
	U_2 = U_2_of_T - U_1_of_T;
	U_1_of_T.resize(0, 1);
	U_2_of_T.resize(0, 1);

	// engy = engy + accum(-M2(T2_3), U_2, sz);
	Matrix<int, Dynamic, 1> M2_of_T = -1 * EigenMatlab::TruncateByBooleansRows(&M2, &T2_3);
	engy = engy + EigenMatlab::Accumarray(&M2_of_T, &U_2, num_regions, 1);
	M2_of_T.resize(0, 1);

	bool update = false;
	for (int b = 0; b < num_regions; b++) {
		if (engy(b) < 0) {
			// M2(M2 == -b) = b + 1;
			Matrix<bool, Dynamic, 1> M2_eq_negb = M2.array() == -1 * b; // (M2 == -b)
			EigenMatlab::AssignByBooleans(&M2, &M2_eq_negb, b + 1);
			M2_eq_negb.resize(0, 1);
			update = true;
		}
	}

	if (update) {
		if (debug) {
			cout << "update triggered" << endl;
			cin.ignore();
		}
		Matrix<bool, Dynamic, 1> M2_gtzero = M2.array() >= 0; // M2 > 0;  // making it <=0 instead of <0 to accommodate adjusted M
		QPBO_CalcVisEnergy(&M2_gtzero, U, E, EI, SE, SEI, TE, TEI, num_in, &U_1, &E_1, &SE_1, &V_1);
		M2_gtzero.resize(0, 1);
	}

	M2 = M2.array() - 1; // readjust M2 to counter original addition to M

	(*Mout) = M2;
	(*Uout) = U_1;
	(*Eout) = E_1;
	(*SEout) = SE_1;
	(*Vout) = V_1;

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Optimization::QPBO_ChooseLabels() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

void Optimization::InitOutputFigures() {
	namedWindow("Reference Image", WINDOW_AUTOSIZE);
	namedWindow("Current Best Disparity Map", WINDOW_AUTOSIZE);
	namedWindow("Energies", WINDOW_AUTOSIZE);
	//namedWindow("Current Proposal Labelling", WINDOW_AUTOSIZE);
	namedWindow("Visibilities", WINDOW_AUTOSIZE);
	namedWindow("Object Disparity Edges", WINDOW_AUTOSIZE);
	imshow("Reference Image", sd_->imgsT_[cid_out_]);
}

void Optimization::CloseOutputFigures() {
	destroyWindow("Reference Image");
	destroyWindow("Current Best Disparity Map");
	destroyWindow("Energies");
	//destroyWindow("Current Proposal Labelling");
	destroyWindow("Visibilities");
	destroyWindow("Object Disparity Edges");
}

void Optimization::UpdateOutputFigures(const Eigen::Matrix<double, Dynamic, 1> *Dcurr, const Eigen::Matrix<double, Dynamic, 1> *Dnew, Eigen::Matrix<bool, Dynamic, 1> *Dswap, Matrix<bool, Dynamic, Dynamic> *Visibilities, Matrix<unsigned int, 2, Dynamic> *EI, Matrix<int, Dynamic, 1> *Uout, Matrix<int, Dynamic, 1> *Eout, Matrix<int, Dynamic, 1> *SEout, Matrix<bool, Dynamic, Dynamic> *Vout) {
	bool debug = false;
	if (debug) cout << endl << endl << "Optimization::UpdateOutputFigures()" << endl;

	int num_in = sd_->use_cids_[cid_out_].size() - 1; // number of used input images excluding the reference(output) image - see ojw_stereo.m for where vals.I is set to exclude the reference image

	// display output figures
	// U_ = double(U_) + accum(EI(1, :)', E_, [tp 1]);
	// U_ = reshape(U_, sp(1), sp(2));
	Matrix<int, Dynamic, 1> EI_0_trans = EI->block(0, 0, 1, EI->cols()).transpose().cast<int>(); // take first row of EI (and transpose and cast it)
	Matrix<double, Dynamic, 1> Udisplay = Uout->cast<double>() + EigenMatlab::Accumarray(&EI_0_trans, Eout, num_pixels_out_, 1).cast<double>(); // result = Uout + accumulation of Eout according to result indices given by EI_0_trans (from first row of EI)
	EI_0_trans.resize(0, 1);
	if (debug) DebugPrintMatrix(&Udisplay, "Udisplay 1");

	// take off the occlusion costs and normalize
	// EI = reshape(sum(V(1:tp, : ), 2), sp(1), sp(2));
	//Matrix<double, Dynamic, 1> EIdisplay = Vout->block(0, 0, num_pixels_out_, Vout->cols()).cast<int>().rowwise().sum().cast<double>(); // result = sum across first num_pixels rows of Vout for all its columns; when this is 0 for a pixel, we get the associated pixel in the energy image reset to a low number (wiped out to black in the image)
	Matrix<double, Dynamic, 1> EIdisplay = Vout->block(0, 0, num_pixels_out_, Vout->cols()).cast<double>() * VectorXd::Ones(Vout->cols()); // faster method for row-wise summation (rowwise().sum())

	// U_ = U_ - (num_in - EI) * double(occl_cost); // subtract occlusion cost for every input point that is not occluding the current reference point?
	Udisplay = Udisplay.array() - (double)(num_in * occl_cost_);
	Udisplay = Udisplay + ((double)occl_cost_) * EIdisplay;
	if (debug) DebugPrintMatrix(&Udisplay, "Udisplay 2");
	// E_ = EI ~= 0;
	Matrix<bool, Dynamic, 1> Edisplay = EIdisplay.cwiseAbs().array() > GLOBAL_FLOAT_ERROR; // result is true if pixels are occluded, false otherwise?
	// U_(E_) = U_(E_) . / EI(E_); // divide the energies of occluded pixels by the number of input points that are occluding them?
	Matrix<double, Dynamic, 1> Udisplay_nvals = Udisplay;
	Udisplay_nvals = Udisplay_nvals.cwiseProduct(EIdisplay.array().inverse().matrix());
	EigenMatlab::AssignByBooleans(&Udisplay, &Edisplay, &Udisplay_nvals);
	Udisplay_nvals.resize(0, 1);
	//DisplayImages::DisplayGrayscaleImageTruncated(&Udisplay, &sd_->masks_[cid_out_], sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_], "Energies");
	DisplayImages::DisplayGrayscaleImage(&Udisplay, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_], "Energies");
	if (debug) DebugPrintMatrix(&Udisplay, "Udisplay");
	Udisplay.resize(0, 1);

	// new disparity map proposal
	Matrix<double, Dynamic, 1> Dnewproposal = (*Dcurr);
	EigenMatlab::AssignByBooleans(&Dnewproposal, Dswap, Dnew);
	Matrix<double, Dynamic, 1> Dfull(num_pixels_out_, 1);
	Dfull.setZero();
	EigenMatlab::AssignByBooleans(&Dfull, &sd_->masks_[cid_out_], &Dnewproposal);
	//DisplayImages::DisplayGrayscaleImageTruncated(&Dfull, &sd_->masks_[cid_out_], sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_], "Current Best Disparity Map");
	DisplayImages::DisplayGrayscaleImage(&Dfull, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_], "Current Best Disparity Map");
	if (debug) DebugPrintMatrix(&Dfull, "Dfull");
	Dnewproposal.resize(0, 1);
	Dfull.resize(0, 1);

	// sc(reshape(sum(V, 2), sp(1), sp(2)), [0 num_in], 'contrast');
	//Matrix<int, Dynamic, 1> Vdisplay = Visibilities->cast<int>().rowwise().sum();
	Matrix<int, Dynamic, 1> Vdisplay = Visibilities->cast<int>() * VectorXi::Ones(Visibilities->cols()); // faster method for row-wise summation (rowwise().sum())

	Vdisplay = Vdisplay.cwiseMax(0);
	Vdisplay = Vdisplay.cwiseMin(num_in);
	//DisplayImages::DisplayGrayscaleImageTruncated(&Vdisplay, &sd_->masks_[cid_out_], sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_], "Visibilities");
	DisplayImages::DisplayGrayscaleImage(&Vdisplay, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_], "Visibilities");
	if (debug) DebugPrintMatrix(&Vdisplay, "Vdisplay");
	Vdisplay.resize(0, 1);

	// U_ = -accum(vals.SEI(2, :)', SE, [tp 1]);
	Matrix<int, Dynamic, 1> SEI_1_trans = SEI_.block(1, 0, 1, SEI_.cols()).transpose().cast<int>(); // use SEI_ rather than SEI_used_ because wil used by Accumarray to populate Udisplay, which is based on a full pixel image
	Udisplay = Uout->cast<double>() + EigenMatlab::Accumarray(&SEI_1_trans, SEout, num_pixels_out_, 1).cast<double>(); // use SEI_ rather than SEI_used_ because wil used by Accumarray to populate Udisplay, which is based on a full pixel image
	//DisplayImages::DisplayGrayscaleImageTruncated(&Udisplay, &sd_->masks_[cid_out_], sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_], "Object Disparity Edges");
	DisplayImages::DisplayGrayscaleImage(&Udisplay, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_], "Object Disparity Edges");
	if (debug) DebugPrintMatrix(&Udisplay, "Udisplay");
	Udisplay.resize(0, 1);
}

// computes SE (updating arg SE), an 8xR triple clique energy table, each column containing the energies[E000 E001 E010 E011 E100 E101 E110 E111] for a given triple clique (including auxiliary node).
// note that triple-clique smoothness should include triple-cliques that include high-confidence pixels as long as one unknown pixel is present in the clique.  Masked-out pixels, however, should not appear in any triple cliques.
// Dcurr and Dnew are truncated to used pixels only and SEI_used_ is used for indices
void Optimization::ComputeSE(const Eigen::Matrix<double, Dynamic, 1> *Dcurr, const Eigen::Matrix<double, Dynamic, 1> *Dnew, double scale_factor, Matrix<int, 8, Dynamic> *SE) {
	bool debug = false;

	cout << "Optimization::ComputeSE()" << endl;

	if (debug) {
		DebugPrintMatrix(Dcurr, "Dcurr");
		DebugPrintMatrix(Dnew, "Dnew");
		DebugPrintMatrix(&SEI_used_, "SEI_used_");
	}

	assert(Dcurr->rows() == num_used_pixels_out_ && Dnew->rows() == num_used_pixels_out_);

	// add surface smoothness constraints (planar == true ... so use a planar prior: finite differences 2nd derivative of disparity)
	// SE = (double([D1(:)'; D2(:)']) - vals.d_min) / vals.d_step; // creates a 2xnum_pixels matrix of quantized disparity labels, using disparity values in D1 and D2
	Matrix<double, Dynamic, Dynamic> SE1 = Dcurr->transpose();
	Matrix<double, 1, Dynamic> tmp_Dnewt = Dnew->transpose();
	EigenMatlab::ConcatVertically(&SE1, &tmp_Dnewt);
	tmp_Dnewt.resize(1, 0);
	SE1 = SE1.array() - (double)sd_->min_disps_[cid_out_];
	double range_factor = 1 / (double)(sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_]);
	SE1 = SE1 * range_factor;
	if (debug) DebugPrintMatrix(&SE1, "SE = (double([D1(:)'; D2(:)']) - vals.d_min) / vals.d_step;");

	// SE = SE(:,vals.SEI);
	Matrix<double, Dynamic, Dynamic> SE2 = EigenMatlab::AccessByIndicesColumns(&SE1, &SEI_used_); // SE is 2xnum_pixels and SEI_ is 3xm (see header file for description of m); returns SE.rows() x (SEI_.rows()*SEI_.cols()) matrix, so it's 2x(3*m);  // dynamic row setting for resizing later; first row is for Dcurr and second for Dnew, with columns ordering disparity access from CM order of SEI_used_ values
	SE1.resize(2, 0);
	if (debug) DebugPrintMatrix(&SE2, "SE = SE(:,vals.SEI);");

	// SE = reshape(SE, 6, []); // rows alternate Dcurr then Dnew across triple clique p,q,r, so it's pixel p with disparity from Dcurr, then pixel p with disparity from Dnew, then pixel q with disparity from Dcurr, etc.
	assert(SE2.cols() % 3 == 0);
	SE2.resize(6, SE2.cols() / 3); // first 3 rows are triple-clique for Dcurr, and second 3 rows are triple-clique for Dnew; the number of columns equals the number of columns in SEI_
	if (debug) DebugPrintMatrix(&SE2, "SE = reshape(SE, 6, []);");

	// take the 2nd derivate of the disparity; Dcurr and Dnew values are interleaved in the rows of SE (first row is Dcurr, second is Dnew, third is Dcurr, fourth is Dnew, etc.) and we need 2nd derivative approximations of each combination of Dcurr and Dnew disparities for triple-clique nodes p, q, and r - there are 8 combinations of 2 values over 3 nodes.
	// SE = reshape(SE([1 3 5; 1 3 6; 1 4 5; 1 4 6; 2 3 5; 2 3 6; 2 4 5; 2 4 6]',:), 3, 8, []); // setup for the 2nd derivate approximation calculation
	// SE = diff(SE, 2); // 2-level diff - approximates a 2nd derivative
	Eigen::Matrix<int, 24, 1> row_indices;
	row_indices << 1, 3, 5, 1, 3, 6, 1, 4, 5, 1, 4, 6, 2, 3, 5, 2, 3, 6, 2, 4, 5, 2, 4, 6; // 1-indexed from ojw
	row_indices = row_indices.array() - 1; // adjust from 1-indexed row numbers to 0-indexed row numbers
	Matrix<double, 24, Dynamic> SE3(24, SE2.cols());
	int r2, r3, r4;
	int ri_rows = row_indices.rows();
	for (int ri = 0; ri < ri_rows; ri++) {
		r2 = row_indices(ri, 0);
		SE3.row(ri) = SE2.row(r2);
	}
	SE2.resize(0, 0);
	Matrix<double, Dynamic, Dynamic> SE4(8, SE3.cols()); // need dynamic row setting for resizing later
	r4 = 0;
	for (r3 = 2; r3 < 24; r3 += 3) {
		SE4.row(r4) = SE3.row(r3) - 2 * SE3.row(r3 - 1) + SE3.row(r3 - 2); // approximating the differential
		r4++;
	}
	SE3.resize(24, 0);
	if (debug) DebugPrintMatrix(&SE4, "SE = diff(SE, 2); // 2-level diff");

	// apply our smoothness weighting - takes the difference in second derivative of disparity across neighboring pixels and applies the weighting factor to it from EW_, with a minimum weight assigned by disp_thresh_
	// SE = reshape(cast(scale_factor * vals.esmooth(SE(:)), class(E)), [], size(vals.SEI, 2));
	SE4.resize(SE4.rows()*SE4.cols(), 1);
	FuseProposals_eSmooth(&SE4);
	SE4 *= scale_factor;
	Matrix<int, Dynamic, Dynamic> SE5 = SE4.cast<int>(); // this truncates rather than rounds to match outcome in Matlab for this line, even though have had to use rounding elsewhere //EigenMatlab::cwiseRound(&SE4);
	SE4.resize(0, 0);
	assert((SE5.rows() * SE5.cols()) % SE5.cols() == 0);
	int SE5_rows = SE5.rows() * SE5.cols() / SEI_used_.cols();
	SE5.resize(SE5_rows, SEI_used_.cols());
	(*SE) = SE5;
	if (debug) DebugPrintMatrix(SE, "SE = reshape(cast(scale_factor * vals.esmooth(SE(:)), class(E)), [], size(vals.SEI, 2));"); // first time I notice SE_ being off by 1 for some values (not seemingly most common 51937 and 623244)

	// MAL - replace smoothness energies by Kinf_ where the difference in disparity between neighboring pixels is greater than a threshold for neighboring pixels within a label segment as given by GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT; actually, make the penalty bigger the larger the disparity difference is
	unsigned int *pS1 = SEI_used_.row(0).data();
	unsigned int *pS2 = SEI_used_.row(1).data();
	unsigned int *pS3 = SEI_used_.row(2).data();
	unsigned int idx1, idx2, idx3;
	double disp1, disp2, disp3;
	int tcr1, tcr2, tcr3;
	double depth1, depth2, depth3, diff1, diff2, penalty;
	bool depth_edge;
	double depth_thresh = GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT / static_cast<double>(sd_->agisoft_to_world_scales_[cid_out_]);
	for (int SE_col = 0; SE_col < SEI_.cols(); SE_col++) { // note that SEI_ and SEI_used_ have the same number of columns - just different index representations
		idx1 = *pS1++;
		idx2 = *pS2++;
		idx3 = *pS3++;

		int SE_row = 0;
		for (int row_idx = 2; row_idx < row_indices.rows(); row_idx += 3) { // 24 entries in row_indices matrix; values of 0,2,4 correspond to Dcurr and values of 1,3,5 correspond to Dnew
			tcr1 = row_idx - 2;
			tcr2 = row_idx - 1;
			tcr3 = row_idx;

			// Dcurr and Dnew in this function have been truncated to used pixels only
			if ((row_indices(tcr1, 0) % 2) == 0) disp1 = (*Dcurr)(idx1, 0);
			else disp1 = (*Dnew)(idx1, 0);
			if ((row_indices(tcr2, 0) % 2) == 0) disp2 = (*Dcurr)(idx2, 0);
			else disp2 = (*Dnew)(idx2, 0);
			if ((row_indices(tcr3, 0) % 2) == 0) disp3 = (*Dcurr)(idx3, 0);
			else disp3 = (*Dnew)(idx3, 0);

			if (disp1 != 0) depth1 = 1. / disp1;
			else depth1 = 0;
			if (disp2 != 0) depth2 = 1. / disp2;
			else depth2 = 0;
			if (disp3 != 0) depth3 = 1. / disp3;
			else depth3 = 0;

			diff1 = abs(depth1 - depth2);
			diff2 = abs(depth2 - depth3);

			depth_edge = false;
			if ((diff1 > depth_thresh) ||
				(diff2 > depth_thresh))
				depth_edge = true;

			if (depth_edge) {
				penalty = round(static_cast<double>(Kinf_)* max(diff1, diff2) / depth_thresh);
				(*SE)(SE_row, SE_col) = penalty;
			}

			SE_row++;
		}
	}
}

// iter is the 0-indexed number of the current iteration
void Optimization::QPBO_eval(Matrix<int, 2, Dynamic> *UE, Eigen::Matrix<unsigned int, 2, Dynamic> *PI, Matrix<int, 4, Dynamic> *PE, Matrix<unsigned int, 3, Dynamic> *TI, Matrix<int, 8, Dynamic> *TE, Matrix<int, Dynamic, 1> *Lout, int iter, bool include_smoothness_terms) {
	/*
	if (GLOBAL_QPBO_USE_LARGER_INTERNAL) {
		QPBO_wrapper_func<int64_t>(UE, PI, PE, TI, TE, Lout, iter, (int64_t)1 << 47, include_smoothness_terms);
	}
	else {
		QPBO_wrapper_func<int32_t>(UE, PI, PE, TI, TE, Lout, iter, (int32_t)1 << 20, include_smoothness_terms);
	}
	return;
	*/
	bool debug = false;

	if (debug) cout << "Optimization::QPBO_eval()" << endl;

	Lout->setZero();
	
#pragma omp parallel for num_threads(omp_get_num_procs()) // not working yet
	// separate by segment and optimize segment-by-segment to reduce optimization type
	for (map<unsigned int, int>::iterator it = sd_->seglabel_counts_[cid_out_].begin(); it != sd_->seglabel_counts_[cid_out_].end(); ++it) {
		unsigned int seglabel = (*it).first;
		int labelcount = (*it).second;

		if (seglabel == 0) continue; // skip seglabel 0, which denotes the background segment

		if (debug) cout << "computing label " << seglabel << endl;

		//Matrix<int, 2, Dynamic> UE_seg;
		Matrix<unsigned int, 2, Dynamic> PI_seg;
		Matrix<int, 4, Dynamic> PE_seg;
		Matrix<unsigned int, 3, Dynamic> TI_seg;
		Matrix<int, 8, Dynamic> TE_seg;
		Matrix<int, Dynamic, 1> Lout_seg = (*Lout);
		
		// variables for pairwise and triple-clique reductions
		int idx_used, idx_full;
		Point p;
		unsigned int node_label;
		int r;
		bool passes;
		int col_seg;

		// determine number of input images to use for this segment; reduce number of input images participating in the optimization according to the number of pixels in the segment to prevent extremely long computation times for large segments since large segments generally also need fewer input image pairwise ePhoto comparisons to arrive at the correct answer; formula is: number of input images to use (always pick closest to virtual camera angle) = round(exp(max(0, GLOBAL_PIXEL_THRESHOLD_FOR_MIN_INPUT_CAMS - num_unk_pixels_in_segment) / 180000), 0); this yields 1 input image at num_unk_pixels_in_segment >= 500,000 pixels and exponentially increasing number of input images below that, with a factor of 1/180,000 pixels
		double factor = 180000.; //170000.;
		int num_cams_to_use = min(11, round(exp(max(0., static_cast<double>(GLOBAL_PIXEL_THRESHOLD_FOR_MIN_INPUT_CAMS - labelcount)) / factor), 0));
		if (debug) cout << "for segment " << seglabel << " with " << labelcount << " pixels, using num_cams_to_use " << num_cams_to_use << endl;

		// pairwise terms (PI, PE) - reduce to where both pixels in pair must be of current segment of interest; also reduce number of input images participating in the optimization according to the number of pixels in the segment to prevent extremely long computation times for large segments since large segments generally also need fewer input image pairwise ePhoto comparisons to arrive at the correct answer; they were added in order from closest to camera angle to farthest from cid_out_
		col_seg = 0;
		Matrix<unsigned int, 2, Dynamic> PI_seg_tmp(2, PI->cols());
		Matrix<int, 4, Dynamic> PE_seg_tmp(4, PE->cols());
		int num_curr_input_cam;
		for (int c = 0; c < PI->cols(); c++) {
			// ignore cameras after first num_cams_to_use (assumes they are ordered closest to farthest from cid_out_)
			num_curr_input_cam = ceil(static_cast<double>((*PI)(1, c)) / static_cast<double>(num_used_pixels_out_));
			if (num_curr_input_cam > (num_cams_to_use + 1)) // Dcurr is cam 0, Dnew is cam 1, and first input cam is cam 2
				continue;

			r = 0;
			passes = true;
			while ((r < 2) && (passes)) {
				idx_used = (*PI)(r, c) % num_used_pixels_out_; // first row should be in range [0, num_used_pixels_out_) but second should be in range [num_used_pixels_out_, (num_used_pixels_out_*(1+2*num_in)) since the first row corresponds to pixels in the reference image while the second row corresponds to pixels in one of the input images
				idx_full = sd_->used_maps_bwd_[cid_out_](idx_used, 0);
				p = PixIndexBwdCM(idx_full, sd_->heights_[cid_out_]);
				node_label = sd_->segs_[cid_out_](p.y, p.x);
				if (node_label != seglabel) passes = false;
				r++;
			}
			if (passes) {
				PI_seg_tmp.col(col_seg) = PI->col(c);
				PE_seg_tmp.col(col_seg) = PE->col(c);
				col_seg++;
			}
		}
		PI_seg.resize(2, col_seg);
		PI_seg = PI_seg_tmp.block(0, 0, 2, col_seg);
		PE_seg.resize(4, col_seg);
		PE_seg = PE_seg_tmp.block(0, 0, 4, col_seg);

		if ((debug) && (PI_seg.cols() > 0)) DebugPrintMatrix(&PI_seg, "PI_seg");
		if ((debug) && (PE_seg.cols() > 0)) DebugPrintMatrix(&PE_seg, "PE_seg");

		// triple clique terms (TI, TE) - reduce to where all pixels in triple clique must be of current segment of interest
		col_seg = 0;
		Matrix<unsigned int, 3, Dynamic> TI_seg_tmp(3, TI->cols());
		Matrix<int, 8, Dynamic> TE_seg_tmp(8, TE->cols());
		for (int c = 0; c < TI->cols(); c++) {
			//cout << "c " << c << endl;
			r = 0;
			passes = true;
			while ((r < 3) && (passes)) {
				idx_used = (*TI)(r, c); // all three rows should be in range [0, num_used_pixels_out_) since all rows correspond to pixels in the referenc image
				idx_full = sd_->used_maps_bwd_[cid_out_](idx_used, 0);
				p = PixIndexBwdCM(idx_full, sd_->heights_[cid_out_]);
				node_label = sd_->segs_[cid_out_](p.y, p.x);
				if (node_label != seglabel) passes = false;
				//cout << "r " << r << ", idx_used " << idx_used << ", idx_full " << idx_full << ", p (" << p.x << ", " << p.y << "), nodel_label " << node_label << ", seglabel " << seglabel << endl;
				r++;
				//if (debug) {
				//	if (passes) {
				//		cout << "r " << r << ", idx_used " << idx_used << ", idx_full " << idx_full << ", p (" << p.x << ", " << p.y << "), nodel_label " << node_label << ", seglabel " << seglabel << /endl;
				//		cin.ignore();
				//	}
				//}
			}
			if (passes) {
				TI_seg_tmp.col(col_seg) = TI->col(c);
				TE_seg_tmp.col(col_seg) = TE->col(c);
				col_seg++;
			}
		}
		TI_seg.resize(3, col_seg);
		if (col_seg > 0) TI_seg = TI_seg_tmp.block(0, 0, 3, col_seg);
		TE_seg.resize(8, col_seg);
		if (col_seg > 0) TE_seg = TE_seg_tmp.block(0, 0, 8, col_seg);

		if ((debug) && (TI_seg.cols() > 0)) DebugPrintMatrix(&TI_seg, "TI_seg");
		if ((debug) && (TE_seg.cols() > 0)) DebugPrintMatrix(&TE_seg, "TE_seg");

		if (GLOBAL_QPBO_USE_LARGER_INTERNAL) {
			QPBO_wrapper_func<int64_t>(UE, &PI_seg, &PE_seg, &TI_seg, &TE_seg, &Lout_seg, iter, (int64_t)1 << 47, include_smoothness_terms);
		}
		else {
			QPBO_wrapper_func<int32_t>(UE, &PI_seg, &PE_seg, &TI_seg, &TE_seg, &Lout_seg, iter, (int32_t)1 << 20, include_smoothness_terms);
		}
		/*
		// copy results from Lout_seg (num_used_pixels_out_seg spots) into full lists at appropriate spots in Lout (num_used_pixels_out_ spots)
		for (map<int, int>::iterator it = map_Lout.begin(); it != map_Lout.end(); ++it) {
			int idx_seg = (*it).first;
			int idx_orig = (*it).second;
			(*Lout)(idx_orig, 0) = Lout_seg(idx_seg, 0);
		}
		*/

		// copy results from Lout_seg into Lout
		int *pLout = Lout->data();
		int *pLout_seg = Lout_seg.data();
		unsigned int *pS = sd_->segs_[cid_out_].data();
		bool *pM = sd_->masks_[cid_out_].data();
		int used_idx = 0;
		for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
			for (int r = 0; r < sd_->widths_[cid_out_]; r++) {
				int idx_full = PixIndexFwdCM(Point(c, r), sd_->heights_[cid_out_]);
				if (sd_->masks_[cid_out_](idx_full, 0)) {
					if (seglabel == sd_->segs_[cid_out_](r, c)) {
						int idx_used = sd_->used_maps_fwd_[cid_out_](idx_full, 0);
						(*Lout)(idx_used, 0) = Lout_seg(idx_used, 0);
					}
				}

				/*
				if (*pM) {
					if (seglabel == *pS)
						*pLout = *pLout_seg;
					pLout++;
					pLout_seg++;
					used_idx++;
				}
				pM++;
				pS++;
				*/
			}
		}

		if (debug) DebugPrintMatrix(Lout, "Lout");
	}
}

// updates Uout, Eout, SEout, Vout
// L is our Dswap
void Optimization::QPBO_CalcVisEnergy(Matrix<bool, Dynamic, 1> *L, Matrix<int, 2, Dynamic> *U, Matrix<int, 4, Dynamic> *E, Matrix<unsigned int, 2, Dynamic> *EI, Matrix<int, 8, Dynamic> *SE, Matrix<unsigned int, 3, Dynamic> *SEI, Matrix<int, 1, Dynamic> *TE, Matrix<unsigned int, 2, Dynamic> *TEI, int num_in, Matrix<int, Dynamic, 1> *Uout, Matrix<int, Dynamic, 1> *Eout, Matrix<int, Dynamic, 1> *SEout, Matrix<bool, Dynamic, Dynamic> *Vout) {
	assert(L->rows() == num_used_pixels_out_);
	//assert(U->cols() == (num_used_pixels_out_ + (num_used_pixels_out_ * 2 * num_in))); // not necessarily true when compress the graph
	//assert(EI->cols() == (num_unknown_pixels_out_ * 2 * num_in)); // not necessarily true when compress the graph
	assert(TE->cols() == TEI->cols());
	assert(Uout->rows() == num_used_pixels_out_);

	bool debug = false;

	Matrix<int, Dynamic, 1> L_int = L->cast<int>();
	
	// generate visibility maps (Vout)
	// V = true(2*tp, num_in);
	(*Vout) = Matrix<bool, Dynamic, Dynamic>(2 * num_used_pixels_out_, num_in);
	Vout->setConstant(true);
	// V(TEI(2, L(TEI(1, :))~= TE')-tp) = false;
	bool *pVout = Vout->data();
	unsigned int *pTEI = TEI->data();
	bool *pL = L->data();
	int *pTE = TE->data();
	unsigned int Lidx, TEIidx;
	for (int i = 0; i < TEI->cols(); i++) {
		Lidx = pTEI[2 * i];
		if (pL[Lidx] != pTE[i]) {
			TEIidx = pTEI[2 * i + 1] - num_used_pixels_out_;
			pVout[TEIidx] = false;
		}
	}

	// calculate energies (Uout, Eout, SEout)
	
	// U = U((0:tp-1)'*2+L+1);
	Matrix<unsigned int, Dynamic, 1> U_idx(num_used_pixels_out_, 1); // incorporate the transpose from the beginning
	U_idx.setLinSpaced(num_used_pixels_out_, 0, (num_used_pixels_out_ - 1));
	U_idx = U_idx * (unsigned int)2;
	U_idx = U_idx + L->cast<unsigned int>();
	//U_idx = U_idx.array() + 1; // skip since want 0-indexed, not 1-indexed
	(*Uout) = EigenMatlab::AccessByIndices(U, &U_idx);
	
	// set up Ltmp for calculations of Eout and SEout
	Matrix<int, Dynamic, Dynamic> Vtmp = Vout->cast<int>();
	Vtmp.resize(Vtmp.rows()*Vtmp.cols(), 1);
	Matrix<int, Dynamic, Dynamic> Ltmp = L_int; // make columns Dynamic even though we know it's 1 so that it matches Vtmp going into call ConcatVertically()
	EigenMatlab::ConcatVertically(&Ltmp, &Vtmp);
	Vtmp.resize(0, 0);
	
	// E = E((0:size(EI, 2) - 1)'*4+L(EI(1,:))*2+L(EI(2,:))+1); // skip last +1 since want 0-indexed, not 1-indexed
	(*Eout) = Matrix<int, Dynamic, 1>(EI->cols(), 1);
	int *pEout = Eout->data();
	int *pLtmp = Ltmp.data();
	unsigned int *pEI = EI->data();
	int *pE = E->data();
	int idxE;
	for (int i = 0; i < EI->cols(); i++) {
		idxE = i * 4 + pLtmp[(pEI[0])] * 2 + pLtmp[(pEI[1])];
		*pEout = pE[idxE];
		pEI+=2;
		pEout++;
	}

	//SE = SE((0:size(SEI, 2) - 1)'*8+L(SEI(1,:))*4+L(SEI(2,:))*2+L(SEI(3,:))+1); // skip last +1 since want 0-indexed, not 1-indexed
	(*SEout) = Matrix<int, Dynamic, 1>(SEI->cols(), 1);
	int *pSEout = SEout->data();
	pLtmp = Ltmp.data();
	unsigned int *pSEI = SEI->data();
	int *pSE = SE->data();
	int idxSE;
	for (int i = 0; i < SEI->cols(); i++) {
		idxSE = i * 8 + pLtmp[(pSEI[0])] * 4 + pLtmp[(pSEI[1])] * 2 + pLtmp[(pSEI[2])];
		*pSEout = pSE[idxSE];
		pSEI+=3;
		pSEout++;
	}

	if (debug) DebugPrintMatrix(SE, "SE");
	if (debug) DebugPrintMatrix(SEout, "SEout");
	
}

// given a set of 3d image coordinates(pixel coordinates plus depth), ordered such that the x coordinates are monotonically increasing, as are the y coordinates within each block of identical x coordinates, finds any occluding / occluded pairs in this set.
// arg V is an Mx3 list of 3d image points, defined as[x y Z], which has been sorted using sortrows(V).
// updates arg P, a 2xN list of vectors of indices for [occluding occluded]' pixel pairs.  First row gets index of occluding pixel and second gets index of occluded pixel.
// assumes V is a compact used pixel representation according to mask, having skipped pixel indices for mask values of false; will return indices for a full pixel image, not a compact representation
void Optimization::FindInteractions(const Eigen::Matrix<double, Dynamic, 3> *V, double dist, Eigen::Matrix<unsigned int, 2, Dynamic> *P) {
	bool debug = false;

	int length = V->rows();

	// // get pointers to input arrays
	const double *Xa = V->data();
	const double *Ya = Xa + length;
	const double *Za = Ya + length;
	const double *Xb, *Yb, *Zb;

	// Create an output buffer
	Matrix<unsigned int, 2, Dynamic> Pext(2, GLOBAL_MAX_MEAN_INTERACTIONS*length);
	unsigned int *Istart = Pext.data();
	unsigned int *I = Istart;

	// Find interactions
	double xa, xb, yb, yal, yah, za, zb;
	int n = 0;
	for (int a = 0; a < (length - 1); a++) {
		xa = *Xa + dist;
		yal = *Ya - dist;
		yah = *Ya + dist;
		za = *Za;
		Xb = ++Xa; // increment Xa first
		Yb = ++Ya; // increment Ya first
		Zb = ++Za; // increment Za first
		for (int b = a + 1; b < length; b++) {
			xb = *Xb++;
			yb = *Yb++;
			zb = *Zb++;
			if (xb > xa)
				break; // if X[b] is not the same as X[a], then we've passed all pixels b that match pixel a in x position, so no more occlusions to pixel a
			if (yb < yal || yb > yah)
				continue; // if y value Y[b] is more than dist pixels away from Y[a], then skip it because no occlusion occurring (we're using dist = 0.5 pixels)
			
			assert(I < &Istart[GLOBAL_MAX_MEAN_INTERACTIONS*length]);

			if (za < zb) { // pixel a occludes pixel b
				*I++ = a;
				*I++ = b;
			}
			else { // pixel b occludes pixel a
				*I++ = b;
				*I++ = a;
			}
			
			n++;
		}
	}

	(*P) = Pext.block(0, 0, 2, n);
}