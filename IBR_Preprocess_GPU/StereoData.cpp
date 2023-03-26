#include "StereoData.h"

StereoData::StereoData() {
	init_to_middlebury = false;
}

StereoData::~StereoData() {
}

void StereoData::Init(std::map<int, Mat> imgsT, std::map<int, Mat> imgMasks, std::map<int, Mat> imgMasks_valid, std::map<int, MatrixXf> depth_maps, std::map<int, Matrix3d> Ks, std::map<int, Matrix3d> Kinvs, std::map<int, Matrix4d> RTs, std::map<int, Matrix4d> RTinvs, std::map<int, Matrix<double, 3, 4>> Ps, std::map<int, Matrix<double, 4, 3>> Pinvs, std::map<int, float> min_depths, std::map<int, float> max_depths, std::map<int, std::map<unsigned int, Matrix<unsigned int, Dynamic, 1>>> unknown_segs, std::map<int, float> agisoft_to_world_scales, Matrix4d AgisoftToWorld, Matrix4d WorldToAgisoft, std::vector<int> exclude_cam_ids, int max_num_cams, Scene *scene) {

	// copy args locally

	min_depths_ = min_depths; // copies elements over, but preserves elements already existing, so must delete those first if desired
	max_depths_ = max_depths; // copies elements over, but preserves elements already existing, so must delete those first if desired

	use_cids_.erase(use_cids_.begin(), use_cids_.end());

	AgisoftToWorld_ = AgisoftToWorld;
	WorldToAgisoft_ = WorldToAgisoft;
	for (std::map<int, float>::iterator it = agisoft_to_world_scales.begin(); it != agisoft_to_world_scales.end(); ++it) {
		int cid = (*it).first;
		agisoft_to_world_scales_[cid] = (*it).second;
	}

	for (map<int, Camera*>::iterator it = scene->cameras_.begin(); it != scene->cameras_.end(); ++it) {
		int cid = (*it).first;
		fn_imgs_[cid] = (*it).second->fn_;
		orientations_[cid] = (*it).second->orientation_;
	}

	for (std::map<int, Mat>::iterator it = imgsT.begin(); it != imgsT.end(); ++it) {
		int cid = (*it).first;

		assert(RTs.find(cid) != RTs.end());
		Mat img = cv::Mat::zeros((*it).second.size(), (*it).second.type());
		(*it).second.copyTo(img);
		imgsT_[cid] = img;
		valid_cam_poses_[cid] = true; // initialize to true, then update later with FilterCamerasByPoseAccuracy()
		heights_[cid] = imgsT_[cid].rows;
		widths_[cid] = imgsT_[cid].cols;
		stereo_computed_[cid] = false;
	}

	for (std::map<int, Mat>::iterator it = imgMasks.begin(); it != imgMasks.end(); ++it) {
		int cid = (*it).first;
		assert(RTs.find(cid) != RTs.end());
		
		Mat imc = Mat::zeros((*it).second.rows, (*it).second.cols, CV_8UC3);
		scene->cameras_[cid]->imgMask_color_.copyTo(imc);
		imgMasks_color_[cid] = imc;

		Mat mask = Mat::zeros((*it).second.rows, (*it).second.cols, CV_8UC1);
		(*it).second.copyTo(mask);
		imgMasks_[cid] = mask;

		Matrix<uchar, Dynamic, Dynamic> mask1((*it).second.rows, (*it).second.cols);
		EigenOpenCV::cv2eigen((*it).second, mask1);
		mask1.resize((*it).second.rows*(*it).second.cols, 1);
		Matrix<bool, Dynamic, 1> mask2 = mask1.array() > GLOBAL_MIN_MASKSEG_LINEVAL;
		masks_[cid] = mask2;
		Matrix<int, Dynamic, 1> mask3 = mask1.cast<int>();
		masks_int_[cid] = mask3;

		segs_[cid] = scene->cameras_[cid]->seg_;
		seglabel_counts_[cid] = scene->cameras_[cid]->seglabel_counts_;
	}

	for (std::map<int, Mat>::iterator it = imgMasks_valid.begin(); it != imgMasks_valid.end(); ++it) {
		int cid = (*it).first;
		assert(RTs.find(cid) != RTs.end());

		Mat mask = Mat::zeros((*it).second.rows, (*it).second.cols, CV_8UC1);
		(*it).second.copyTo(mask);
		imgMasks_valid_[cid] = mask;

		Matrix<uchar, Dynamic, Dynamic> mask1((*it).second.rows, (*it).second.cols);
		EigenOpenCV::cv2eigen((*it).second, mask1);
		mask1.resize((*it).second.rows*(*it).second.cols, 1);

		Matrix<bool, Dynamic, 1> mask2 = mask1.array() > GLOBAL_MIN_MASKSEG_LINEVAL;

		masks_valid_[cid] = mask2;

		// closeup booleans were originally set using imgMask_valid_ for each camera
		closeup_xmins_[cid] = scene->cameras_[cid]->closeup_xmin_;
		closeup_xmaxs_[cid] = scene->cameras_[cid]->closeup_xmax_;
		closeup_ymins_[cid] = scene->cameras_[cid]->closeup_ymin_;
		closeup_ymaxs_[cid] = scene->cameras_[cid]->closeup_ymax_;
	}

	// create dilated masks for use as approximate masks to accomodate error in camera pose when testing reprojection against masks
	for (std::map<int, Mat>::iterator it = imgMasks.begin(); it != imgMasks.end(); ++it) {
		int cid = (*it).first;
		DilateMask(cid);
	}

	for (std::map<int, MatrixXf>::iterator it = depth_maps.begin(); it != depth_maps.end(); ++it) {
		int cid = (*it).first;
		assert(RTs.find(cid) != RTs.end());
		Eigen::MatrixXf dm((*it).second.rows(), (*it).second.cols());
		dm = (*it).second;
		depth_maps_[cid] = dm;
		Matrix<bool, Dynamic, Dynamic> kd = dm.array() != 0.;
		kd.resize(kd.rows()*kd.cols(), 1);
		known_depths_[cid] = kd;
		num_known_pixels_[cid] = kd.count();

		MatrixXf disparity_mapf = DepthMap::ConvertDepthMapToDisparityMap(&(*it).second);
		disparity_mapf.resize(disparity_mapf.rows()*disparity_mapf.cols(), 1);
		Matrix<double, Dynamic, 1> disparity_mapd = disparity_mapf.cast<double>();
		disparity_maps_[cid] = disparity_mapd;
	}

	// any camera that do not already have a depth map should be given one, albeit with no data
	for (std::map<int, Mat>::iterator it = imgsT.begin(); it != imgsT.end(); ++it) {
		int cid = (*it).first;
		if (depth_maps_.find(cid) != depth_maps_.end()) continue;
		MatrixXf dm((*it).second.rows, (*it).second.cols);
		dm.setZero();
		depth_maps_[cid] = dm;
	}

	for (std::map<int, Matrix3d>::iterator it = Ks.begin(); it != Ks.end(); ++it) {
		int cid = (*it).first;
		assert(imgsT.find(cid) != imgsT.end());
		Matrix3f K = (*it).second.cast<float>();
		Ks_[cid] = K;
	}

	for (std::map<int, Matrix3d>::iterator it = Kinvs.begin(); it != Kinvs.end(); ++it) {
		int cid = (*it).first;
		assert(imgsT.find(cid) != imgsT.end());
		Matrix3f Kinv = (*it).second.cast<float>();
		Kinvs_[cid] = Kinv;
	}

	for (std::map<int, Matrix4d>::iterator it = RTs.begin(); it != RTs.end(); ++it) {
		int cid = (*it).first;
		assert(imgsT.find(cid) != imgsT.end());
		Matrix4f RT = (*it).second.cast<float>();
		RTs_[cid] = RT;
	}

	for (std::map<int, Matrix4d>::iterator it = RTinvs.begin(); it != RTinvs.end(); ++it) {
		int cid = (*it).first;
		assert(imgsT.find(cid) != imgsT.end());
		Matrix4f RTinv = (*it).second.cast<float>();
		RTinvs_[cid] = RTinv;

		// set up use_cids, which require RTinvs_
		Matrix4d RTinv_out = RTinvs_[cid].cast<double>();
		Point3d view_pos = Camera::GetCameraPositionWS(&RTinv_out);
		Point3d view_dir = Camera::GetCameraViewDirectionWS(&RTinv_out);
		use_cids_[cid] = scene->GetClosestCams(view_pos, view_dir, exclude_cam_ids, max_num_cams); // only use relevant cameras
	}

	for (std::map<int, Matrix<double, 3, 4>>::iterator it = Ps.begin(); it != Ps.end(); ++it) {
		int cid = (*it).first;
		assert(imgsT.find(cid) != imgsT.end());
		Matrix<float, 3, 4> P = (*it).second.cast<float>();
		Ps_[cid] = P;
	}

	for (std::map<int, Matrix<double, 4, 3>>::iterator it = Pinvs.begin(); it != Pinvs.end(); ++it) {
		int cid = (*it).first;
		assert(imgsT.find(cid) != imgsT.end());
		Matrix<float, 4, 3> Pinv = (*it).second.cast<float>();
		Pinvs_[cid] = Pinv;
	}

	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		Eigen::Matrix<float, Dynamic, 3> A;
		EigenOpenCV::cv2eigenImage<float>(&(*it).second, &A);
		As_[cid] = A;
	}

	InitPout2ins(); // create projection matrices that transform from each "reference" camera SS to each "input" camera's SS
	InitPixelData();
	UpdateDisps();
	//BuildAllValidDisparityRanges();

	for (map<int, Matrix<bool, Dynamic, 1>>::iterator it = masks_unknowns_.begin(); it != masks_unknowns_.end(); ++it) {
		int cid = (*it).first;
		masks_unknowns_orig_[cid] = (*it).second;
	}

	cout << "StereoData::Init() complete" << endl;
}

void StereoData::RevertOrigUnknowns(int cid) {
	Matrix<bool, Dynamic, 1> m = masks_unknowns_orig_[cid];
	SpecifyPixelData(cid, &m);
}

// dilates camera cid's mask to update an entry in masks_dilated_; requires that masks_ and heights_ and widths_ are set
void StereoData::DilateMask(int cid) {
	assert(masks_.find(cid) != masks_.end());

	if (GLOBAL_MASK_DILATION == 0) {
		masks_dilated_[cid] = masks_[cid];
		return;
	}

	// Set up morphological operator
	int morph_type = MORPH_RECT; // MORPH_ELLIPSE
	int morph_size = GLOBAL_MASK_DILATION; // get rid of small regional markings
	Mat element = getStructuringElement(morph_type,
		Size(2 * morph_size + 1, 2 * morph_size + 1),
		Point(morph_size, morph_size));
	
	Matrix<int, Dynamic, Dynamic> maskEig = masks_[cid].cast<int>();
	maskEig.resize(heights_[cid], widths_[cid]);
	Mat maskCV = Mat::zeros(heights_[cid], widths_[cid], CV_8UC1);
	EigenOpenCV::eigen2cv(maskEig, maskCV);
	Mat mask_dilated = cv::Mat::zeros(heights_[cid], widths_[cid], CV_8UC1);
	cv::dilate(maskCV, mask_dilated, element);
	Matrix<uchar, Dynamic, Dynamic> mask1(heights_[cid], widths_[cid]);
	EigenOpenCV::cv2eigen(mask_dilated, mask1);
	mask1.resize(mask1.rows()*mask1.cols(), 1);
	Matrix<bool, Dynamic, 1> mask2 = mask1.cast<bool>();
	masks_dilated_[cid] = mask2;
}

// for camera with ID cid, updates Xunknowns and Yunknowns to hold coordinates of a grid of size height x width that only includes positions for which Mask is true (the count of these positions is also given in num_mask_true to speed the function
// updates Iunknowns to hold coordinate index in CM order corresponding to X and Y
// num_unknown_pixels_ must be initialized first, along with masks_ and width and height for image with ID cid
void StereoData::InitUnknownAndUsedDisparityCoordinates(int cid, Matrix<double, Dynamic, 1> *Xuseds, Matrix<double, Dynamic, 1> *Yuseds, Matrix<double, Dynamic, 1> *Xunknowns, Matrix<double, Dynamic, 1> *Yunknowns, Matrix<unsigned int, Dynamic, 1> *Iunknowns) {
	bool debug = false;

	Xuseds->resize(num_used_pixels_[cid], 1);
	Yuseds->resize(num_used_pixels_[cid], 1);
	Xunknowns->resize(num_unknown_pixels_[cid], 1);
	Yunknowns->resize(num_unknown_pixels_[cid], 1);
	Iunknowns->resize(num_unknown_pixels_[cid], 1);

	bool *pMunk = masks_unknowns_[cid].data();
	bool *pMused = masks_[cid].data();
	double *pXused = Xuseds->data();
	double *pYused = Yuseds->data();
	double *pXunk = Xunknowns->data();
	double *pYunk = Yunknowns->data();
	unsigned int *pIunk = Iunknowns->data();
	unsigned int i = 0;
	for (int c = 0; c < widths_[cid]; c++) {
		for (int r = 0; r < heights_[cid]; r++) {
			if (*pMunk++) {
				*pXunk++ = (double)c;
				*pYunk++ = (double)r;
				*pIunk++ = i;
				i++;
			}
			if (*pMused++) {
				*pXused++ = (double)c;
				*pYused++ = (double)r;
			}
		}
	}

	if (debug) {
		DebugPrintMatrix(Xuseds, "Xuseds");
		DebugPrintMatrix(Yuseds, "Yuseds");
		DebugPrintMatrix(Xunknowns, "Xunknowns");
		DebugPrintMatrix(Yunknowns, "Yunknowns");
		DebugPrintMatrix(Iunknowns, "Iunknowns");
	}
}

// initializes values in num_used_pixels_ and used_pixels_
void StereoData::InitPixelData(int cid_specific) {// std::map<int, std::map<unsigned int, Matrix<unsigned int, Dynamic, 1>>> unknown_segs) {
	bool debug = false;

	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid = (*it).first;
		if (!valid_cam_poses_[cid]) continue;
		if ((cid_specific != -1) &&
			(cid != cid_specific))
			continue;

		num_pixels_[cid] = imgsT_[cid].rows * imgsT_[cid].cols;

		//num_used_pixels_[cid] = masks_[cid].count(); // only works if only use mask data and not high/low confidence data

		known_depths_[cid].resize(0, 1);
		known_depths_[cid].resize(num_pixels_[cid], 1);
		known_depths_[cid].setConstant(false);

		bool *pM = masks_[cid].data();
		float *pD = depth_maps_[cid].data();
		bool *pK = known_depths_[cid].data();
		int num_used = 0;
		int num_unknown = 0;
		int num_known = 0;
		for (int c = 0; c < masks_[cid].cols(); c++) {
			for (int r = 0; r < masks_[cid].rows(); r++) {
				if (*pM) { // skip pixels that are masked out
					num_used++;
					if ((!GLOBAL_TRUST_AGISOFT_DEPTHS) ||
						(*pD == 0.)) // skip pixels for which depth data is already available (assumes this data has already been scrubbed, which is performed in a Scene::Init() sub-call); don't use GLOBAL_FLOAT_ERROR because depth 0. exactly signifies no trustworthy data
						num_unknown++;
					else {
						*pK = true;
						num_known++;
					}
				}
				pM++;
				pD++;
				pK++;
			}
		}

		num_used_pixels_[cid] = num_used;
		num_unknown_pixels_[cid] = num_unknown;
		num_known_pixels_[cid] = num_known;

		// wipe any previous data for camera cid
		used_maps_fwd_[cid].resize(0, 1);
		used_maps_bwd_[cid].resize(0, 1);
		unknown_maps_fwd_[cid].resize(0, 1);
		unknown_maps_bwd_[cid].resize(0, 1);
		Aunknowns_[cid].resize(0, 3);
		masks_unknowns_[cid].resize(0, 1);

		// fill new data for camera cid
		used_maps_fwd_[cid].resize(num_pixels_[cid], 1);
		used_maps_fwd_[cid].setConstant(-1);
		used_maps_bwd_[cid].resize(num_used, 1);
		unknown_maps_fwd_[cid].resize(num_pixels_[cid], 1);
		unknown_maps_fwd_[cid].setConstant(-1);
		unknown_maps_bwd_[cid].resize(num_unknown, 1);
		Aunknowns_[cid].resize(num_unknown, 3);
		masks_unknowns_[cid].resize(num_pixels_[cid], 1);
		pM = masks_[cid].data();
		pD = depth_maps_[cid].data();
		bool *pMu = masks_unknowns_[cid].data();
		int *pUsedf = used_maps_fwd_[cid].data();
		int *pUsedb = used_maps_bwd_[cid].data();
		int *pUnkf = unknown_maps_fwd_[cid].data();
		int *pUnkb = unknown_maps_bwd_[cid].data();
		int idx_used = 0;
		int idx_unknown = 0;
		int idx_all = 0;
		for (int c = 0; c < masks_[cid].cols(); c++) {
			for (int r = 0; r < masks_[cid].rows(); r++) {
				if (*pM) { // skip pixels that are masked out
					*pUsedf = idx_used;
					*pUsedb++ = idx_all;
					idx_used++;

					if ((!GLOBAL_TRUST_AGISOFT_DEPTHS) ||
						(*pD == 0.)) { // skip pixels for which depth data is already available (assumes this data has already been scrubbed, which is performed in a Scene::Init() sub-call)
						Aunknowns_[cid].row(idx_unknown) = As_[cid].row(idx_all);
						*pMu = true;
						*pUnkf = idx_unknown;
						*pUnkb++ = idx_all;
						idx_unknown++;
					}
					else
						*pMu = false;
				}
				else
					*pMu = false;

				pM++;
				pD++;
				pMu++;
				pUsedf++;
				pUnkf++;
				idx_all++;
			}
		}

		if (debug) {
			Matrix<float, Dynamic, 3> Arecon(As_[cid].rows(), As_[cid].cols());
			EigenMatlab::AssignByIndicesRows(&Arecon, &unknown_maps_bwd_[cid], &Aunknowns_[cid]);
			Matrix<float, Dynamic, 3> Amasked = As_[cid];
			EigenMatlab::MaskRows(&Amasked, &masks_unknowns_[cid]);
			bool equal = EigenMatlab::TestEqual(&Amasked, &Arecon);
			if (equal) cout << "StereoData::InitPixelData() used_maps_bwd[" << cid << "] passed test" << endl;
			else cout << "StereoData::InitPixelData() used_maps_bwd[" << cid << "] failed test" << endl;
			cin.ignore();
		}
	}

	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid = (*it).first;
		if (!valid_cam_poses_[cid]) continue;
		if ((cid_specific != -1) &&
			(cid != cid_specific))
			continue;

		InitUnknownAndUsedDisparityCoordinates(cid, &Xuseds_[cid], &Yuseds_[cid], &Xunknowns_[cid], &Yunknowns_[cid], &Iunknowns_[cid]);
	}
}

void StereoData::SpecifyPixelData(int cid, Matrix<bool, Dynamic, 1> *mask_unknowns) {// std::map<int, std::map<unsigned int, Matrix<unsigned int, Dynamic, 1>>> unknown_segs) {
	bool debug = false;

	num_pixels_[cid] = imgsT_[cid].rows * imgsT_[cid].cols;
	masks_unknowns_[cid] = (*mask_unknowns);
	num_unknown_pixels_[cid] = mask_unknowns->count();
	num_used_pixels_[cid] = masks_[cid].count();
	Matrix<bool, Dynamic, 1> mask_notunknown = (*mask_unknowns);
	EigenMatlab::CwiseNot(&mask_notunknown);
	known_depths_[cid] = EigenMatlab::CwiseAnd(&masks_[cid], &mask_notunknown);
	num_known_pixels_[cid] = num_used_pixels_[cid] - num_unknown_pixels_[cid];

	// wipe any previous data for camera cid
	used_maps_fwd_[cid].resize(0, 1);
	used_maps_bwd_[cid].resize(0, 1);
	unknown_maps_fwd_[cid].resize(0, 1);
	unknown_maps_bwd_[cid].resize(0, 1);
	Aunknowns_[cid].resize(0, 3);

	// fill new data for camera cid
	used_maps_fwd_[cid].resize(num_pixels_[cid], 1);
	used_maps_fwd_[cid].setConstant(-1);
	used_maps_bwd_[cid].resize(num_used_pixels_[cid], 1);
	unknown_maps_fwd_[cid].resize(num_pixels_[cid], 1);
	unknown_maps_fwd_[cid].setConstant(-1);
	unknown_maps_bwd_[cid].resize(num_unknown_pixels_[cid], 1);
	Aunknowns_[cid].resize(num_unknown_pixels_[cid], 3);
	masks_unknowns_[cid].resize(num_pixels_[cid], 1);
	bool *pM = masks_[cid].data();
	bool *pMunk = masks_unknowns_[cid].data();
	bool *pMu = masks_unknowns_[cid].data();
	int *pUsedf = used_maps_fwd_[cid].data();
	int *pUsedb = used_maps_bwd_[cid].data();
	int *pUnkf = unknown_maps_fwd_[cid].data();
	int *pUnkb = unknown_maps_bwd_[cid].data();
	int idx_used = 0;
	int idx_unknown = 0;
	int idx_all = 0;
	for (int c = 0; c < masks_[cid].cols(); c++) {
		for (int r = 0; r < masks_[cid].rows(); r++) {
			if (*pM) { // skip pixels that are masked out
				*pUsedf = idx_used;
				*pUsedb++ = idx_all;
				idx_used++;

				if (*pMunk) { // skip pixels that are known
					Aunknowns_[cid].row(idx_unknown) = As_[cid].row(idx_all);
					*pMu = true;
					*pUnkf = idx_unknown;
					*pUnkb++ = idx_all;
					idx_unknown++;
				}
				else
					*pMu = false;
			}
			else
				*pMu = false;

			pM++;
			pMunk++;
			pMu++;
			pUsedf++;
			pUnkf++;
			idx_all++;
		}
	}

	if (debug) {
		Matrix<float, Dynamic, 3> Arecon(As_[cid].rows(), As_[cid].cols());
		EigenMatlab::AssignByIndicesRows(&Arecon, &unknown_maps_bwd_[cid], &Aunknowns_[cid]);
		Matrix<float, Dynamic, 3> Amasked = As_[cid];
		EigenMatlab::MaskRows(&Amasked, &masks_unknowns_[cid]);
		bool equal = EigenMatlab::TestEqual(&Amasked, &Arecon);
		if (equal) cout << "StereoData::InitPixelData() unknown_maps_bwd_[" << cid << "] passed test" << endl;
		else cout << "StereoData::InitPixelData() unknown_maps_bwd_[" << cid << "] failed test" << endl;
		cin.ignore();
	}

	InitUnknownAndUsedDisparityCoordinates(cid, &Xuseds_[cid], &Yuseds_[cid], &Xunknowns_[cid], &Yunknowns_[cid], &Iunknowns_[cid]);
}

// returns closest label to given disparity value
// labels are truncated to between 0 and the maximum label number - 1
int StereoData::DispValueToLabel(int cid, float disp_val) {
	int label = round((disp_val - min_disps_[cid]) / disp_steps_[cid], 0);
	if (label < 0) label = 0;
	else if (label >(disps_[cid].size() - 1)) label = (int)disps_[cid].size() - 1;
	return label;
}

// updates disps_, nums_disps_, min_disps_, max_disps_, and disp_steps_ members based on use_cids_ and other data
void StereoData::UpdateDisps(int cid_specific) {
	bool debug = false;

	cout << "StereoData::UpdateDisps()" << endl;

	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid_ref = (*it).first;
		if ((cid_specific != -1) &&
			(cid_ref != cid_specific)) continue;

		cout << "computing for cid " << cid_ref << endl;

		if (init_to_middlebury) {
			min_disps_[cid_ref] = 0;
			max_disps_[cid_ref] = 236;
			disp_steps_[cid_ref] = 1;// ojw's disp_step_ = -236, which differs from mine, because his is the difference between min and max, while mine is the difference between adjacent disparity labelsf
			nums_disps_[cid_ref] = 237;
			disps_[cid_ref] = ArrayXd(nums_disps_[cid_ref]);
			int d = max_disps_[cid_ref];
			for (int i = 0; i < nums_disps_[cid_ref]; i++) {
				disps_[cid_ref](i) = d;
				d -= static_cast<int>(disp_steps_[cid_ref]);
			}
			return;
		}

		// extend range by GLOBAL_EXTEND_DEPTH_RANGE front and back
		float ext = (max_depths_[cid_ref] - min_depths_[cid_ref]) * GLOBAL_EXTEND_DEPTH_RANGE;
		float min_depth_ext = min_depths_[cid_ref] - ext; // this is in camera space for cid_out
		if (min_depth_ext <= 0.) min_depth_ext = GLOBAL_MIN_CS_DEPTH;
		float max_depth_ext = max_depths_[cid_ref] + ext; // this is in camera space for cid_out

		Matrix<float, 4, 2> depth_extremes_CSout; // 2 reference camera space points (0,0,min_depth,1) and (0,0,max_depth,1)
		depth_extremes_CSout.setZero();
		depth_extremes_CSout(2, 0) = min_depth_ext;
		depth_extremes_CSout(2, 1) = max_depth_ext;
		depth_extremes_CSout(3, 0) = 1.;
		depth_extremes_CSout(3, 1) = 1.;

		float disp_vals = 0; // to hold the number of disparity levels; to be set by projecting points (0,0,min_depth,1) and (0,0,max_depth,1) into each reference camera, finding the pixel distance between projected near and far points, and ensuring the minimum spacing between disparity samples is 0.5 pixels in screen space

		for (map<int, Matrix<float, 3, 4>>::iterator it = Ps_.begin(); it != Ps_.end(); ++it) {
			int cid = (*it).first;
			if (std::find(use_cids_[cid_ref].begin(), use_cids_[cid_ref].end(), cid) == use_cids_[cid_ref].end()) continue;
			if (cid == cid_ref) continue;

			Matrix<float, 3, 2> depth_extremes_SS = (*it).second * RTinvs_[cid_ref] * depth_extremes_CSout; // project world space points (0,0,min_depth,1) and (0,0,max_depth,1) into screen space of this camera
			// divide the screen space points through by the homogeneous coordinates to normalize them
			float h0 = depth_extremes_SS(2, 0);
			depth_extremes_SS.col(0) = depth_extremes_SS.col(0) / h0;
			float h1 = depth_extremes_SS(2, 1);
			depth_extremes_SS.col(1) = depth_extremes_SS.col(1) / h1;
			// find the maximum pixel distance among x and y dimensions for the two projected pixels
			float xdist = abs(depth_extremes_SS(0, 0) - depth_extremes_SS(0, 1));
			float ydist = abs(depth_extremes_SS(1, 0) - depth_extremes_SS(1, 1));
			disp_vals = max(disp_vals, max(xdist, ydist));
		}
		nums_disps_[cid_ref] = ceil(disp_vals * 2.); // ensure minimum spacing is 0.5 pixels, so multiply minimum 1 pixel spacing by 2 to get minimum 0.5 pixel spacing and round up to ensure integer number of disparity levels is over the threshold

		//num_disps_ = 3; // for debugging only so later functions run more quickly - must remove this line

		// Calculate disparities - disps ends up as range from 1/min_depth to 1/max_depth with disp_vals number of steps, evenly spaced
		disps_[cid_ref] = ArrayXd(nums_disps_[cid_ref]); // set up disparities from 0 through (num_disps - 1)
		for (int i = 0; i < nums_disps_[cid_ref]; i++) {
			disps_[cid_ref](i) = static_cast<float>(i);
		}
		// taking disp_vals to be 10, min depth to be 2m and max to be 8m, take it through the calcs...
		disps_[cid_ref] *= (1. - (min_depth_ext / max_depth_ext)) / static_cast<float>((nums_disps_[cid_ref] - 1)); // disps: 0, 3/36, 6/36, 9/36, ... , 27/36
		disps_[cid_ref] = (1. - disps_[cid_ref]) / min_depth_ext; // disps: 1/2, 5/12, 3/8, ... , 1/8

		// order disparities from foreground to background => descending
		std::sort(disps_[cid_ref].data(), disps_[cid_ref].data() + disps_[cid_ref].size(), std::greater<float>()); // sorts values in descending order, but are already in that order at this point...uncomment if want to make doubly-sure
		//igl::sort(X, dim, mode, Y, IX); // igl method for sorting values in descending order

		max_disps_[cid_ref] = static_cast<float>(disps_[cid_ref](0));
		min_disps_[cid_ref] = static_cast<float>(disps_[cid_ref](nums_disps_[cid_ref] - 1)); // max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp
		disp_steps_[cid_ref] = static_cast<float>(disps_[cid_ref](0)) - static_cast<float>(disps_[cid_ref](1)); // max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp

		if (debug) {
			cout << "updating for cid " << cid_ref << endl;
			cout << "Depth extended min and max in the camera space of the reference camera: " << min_depth_ext << ", " << max_depth_ext << endl;
			cout << "Max disparity (closest): " << max_disps_[cid_ref] << endl;
			cout << "Min disparity (farthest): " << min_disps_[cid_ref] << endl;
			cout << "Number of disparities: " << nums_disps_[cid_ref] << endl;
			cout << "Disparity step: " << disp_steps_[cid_ref] << endl;
			cout << endl;
			cin.ignore();
		}
	}
}

// transforms the reference frame and sets the extrinsics matrices and output extrinsics matrix such that the output camera extrinsics matrix is the identity
// RTins is a map of camera ID => pointer to camera extrinsics matrix
// transforms all cameras, regardless of whether used, each time is called
void StereoData::InitPout2ins() {

	if (init_to_middlebury) return;

	for (std::map<int, Matrix<float, 3, 4>>::iterator it = Ps_.begin(); it != Ps_.end(); ++it) { // transforms all cameras, regardless of whether used, each time is called
		int cid_src = (*it).first;
		for (std::map<int, Matrix<float, 3, 4>>::iterator it = Ps_.begin(); it != Ps_.end(); ++it) { // transforms all cameras, regardless of whether used, each time is called
			int cid_dest = (*it).first;
			Pout2ins_[cid_src][cid_dest] = Pss1Toss2(cid_src, cid_dest);
		}
	}
}

Matrix<float, 3, 4> StereoData::Pss1Toss2(int cid1, int cid2) {
	Matrix<float, 1, 4> ext;
	ext << 0., 0., 0., 1.;
	Matrix4f Pout_ext;
	Pout_ext << Ps_[cid1], ext;
	Matrix4f Pout_ext_inv = Pout_ext.inverse();

	Matrix<float, 3, 4> Pss = Ps_[cid2] * Pout_ext_inv;
	return Pss;
}

// initialize using Middlebury's cones dataset
void StereoData::Init_MiddleburyCones(int cid_ref) {
	use_cids_[cid_ref].erase(use_cids_[cid_ref].begin(), use_cids_[cid_ref].end());
	use_cids_[cid_ref].push_back(2);
	use_cids_[cid_ref].push_back(6);

	// load color images
	for (int i = 0; i < 9; i++) {
		std::string sfn = GLOBAL_FILEPATH_DATA + "cones\\im" + std::to_string(i) + ".ppm";
		const char* fncolor = sfn.c_str();
		cv::Mat imgcolor = readPPM(fncolor);
		imgsT_[i] = imgcolor;
	}
	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		Eigen::Matrix<float, Dynamic, 3> A;
		EigenOpenCV::cv2eigenImage<float>(&(*it).second, &A);
		As_[cid] = A;

		std::string fn_mask_full = GLOBAL_FILEPATH_DATA + +"cones\\input" + std::to_string(cid) + "_mask.png";
		Mat imgMask = imread(fn_mask_full, IMREAD_GRAYSCALE);
		if (imgMask.rows == 0 || imgMask.cols == 0)
			imgMask = cv::Mat((*it).second.rows, (*it).second.cols, CV_8UC1, 255); // if mask not found, assume all pixels are "opaque" foreground pixels
		Matrix<int, Dynamic, Dynamic> mask_tmp((*it).second.rows, (*it).second.cols);
		EigenOpenCV::cv2eigen(imgMask, mask_tmp);
		mask_tmp.resize((*it).second.rows*(*it).second.cols, 1);
		Matrix<bool, Dynamic, 1> mask = mask_tmp.cast<bool>();
		masks_[cid] = mask;
		MatrixXf dm((*it).second.rows, (*it).second.cols);
		dm.setZero();
		depth_maps_[cid] = dm;
		Matrix<bool, Dynamic, 1> kd((*it).second.rows*(*it).second.cols, 1);
		kd.setConstant(false);
		known_depths_[cid] = kd;
	}
	for (std::map<int, Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid = (*it).first;
		Matrix<float, Dynamic, 3> Aunknown = EigenMatlab::TruncateByBooleansRows(&As_[cid], &masks_[cid]);
		Aunknowns_[cid] = Aunknown;
	}
	
	// load depth maps
	//for (int i = 2; i <= 6; i+=4) {
	//std::string sfn = GLOBAL_FILEPATH_DATA + "cones\\disp" + std::to_string(i) + ".pgm";
	//const char* fndepth = sfn.c_str();
	//Eigen::MatrixXf dm = readPGM(fndepth);
	//depth_maps_[i] = dm;
	//}

	// set projection matrices
	for (int i = 0; i < 9; i++) {
		Eigen::Matrix<float, 3, 4> P;
		P.setZero();
		P(0, 0) = 1.;
		P(1, 1) = 1.;
		P(2, 2) = 1.;
		int disparity_factor = 4;
		int im_space = i - cid_ref;
		//P(0, 3) = -i / (disparity_factor * im_space);
		Pout2ins_[cid_ref][i] = P;
	}
	Pout2ins_[cid_ref][6](0, 3) = -0.25;

	init_to_middlebury = true;

	InitPixelData();
}

// returns true if successful, false otherwise (one reason for false occurs when reference camera is not accurately posed)
void StereoData::ClearStatistics(int cid_ref) {
	bool debug = false;

	map_ = Matrix<unsigned short, Dynamic, 1>(num_pixels_[cid_ref], 1); // num_pixels_[cid_out] x 1; contains iteration number at which each disparity map coefficient was set during optimization

	// optimization settings
	average_over_ = GLOBAL_OPTIMIZATION_AVERAGE_OVER;
	max_iters_ = GLOBAL_OPTIMIZATION_MAX_ITERS + average_over_ + 1; // iter starts at options.average_over + 1 and increases by 1 with each iteration in Stereo_optim()
	converge_ = GLOBAL_OPTIMIZATION_CONVERGE * 0.01 * static_cast<double>(average_over_); // loop condition-check compares current energy to energy average_over iterations earlier, so checking for meeting minimum average convergence necessitates multiplying single-iteration change in convergence by the number of iterations of which we are averaging.  There is an additional factor of 0.01 because options.converge is given as a percentage and we need to convert it to a decimal value here.

	// optimization results containers
	count_updated_vals_.resize(1, max_iters_);
	count_unlabelled_vals_.resize(1, max_iters_);
	count_unlabelled_regions_.resize(1, max_iters_);
	count_unlabelled_after_QPBOP_.resize(1, max_iters_);
	timings_data_term_eval_.resize(1, max_iters_);
	timings_smoothness_term_eval_.resize(1, max_iters_);
	timings_qpbo_fuse_time_.resize(1, max_iters_);
	timings_iteration_time_.resize(1, max_iters_);
}

void StereoData::MaskImg(Mat *imgT, Matrix<bool, Dynamic, Dynamic> *mask, Mat *imgT_masked) {
	assert(imgT->rows == mask->rows() && imgT->cols == mask->cols());
	assert(imgT->rows == imgT_masked->rows && imgT->cols == imgT_masked->cols);
	imgT->copyTo(*imgT_masked);
	bool* pM = mask->data();
	Point pt;
	int h = imgT->rows;
	for (int i = 0; i < mask->rows(); i++) {
		pt = PixIndexBwdCM(i, h);
		if (*pM++ == 0) imgT_masked->at<Vec3b>(pt.y, pt.x) = Vec3b(0, 0, 0);
	}
}

// updates disp_val to snap it to the next farther valid value in the reference image for the unknown pixel with contracted unknown space pixel index idx_unk, if one exists; if one doesn't exist, no change is made
// a 0 disparity value (undefined/unknown depth) is snapped to the largest valid disparity value, indicating the smallest valid depth
void StereoData::PushToSnapNextFartherValidDisparity(int cid, int idx_unk, double &disp_val) {
	bool debug = false;
	bool debug_test = true;

	int disp_label = DispValueToLabel(cid, static_cast<float>(disp_val));

	// a 0 disparity value (undefined/unknown depth) is snapped to the largest valid disparity value, indicating the smallest valid depth
	if (disp_val == 0) {
		Matrix<bool, 1, Dynamic> udv = unknown_disps_valid_[cid].row(idx_unk);
		bool *p;
		bool valid;
		int curr_label = disp_label;
		p = udv.data() + curr_label;
		valid = *p;
		while ((!valid) &&
			(curr_label > 0)) {
			p--;
			curr_label--;
			valid = *p;
		}
		if (valid) disp_val = static_cast<double>(DispLabelToValue(cid, curr_label));
		return;
	}

	float disp_val_float, snapped_disp_val;

	disp_val_float = static_cast<float>(disp_val);

	disp_label = DispValueToLabel(cid, disp_val_float);
	snapped_disp_val = DispLabelToValue(cid, disp_label);
	disp_val = static_cast<double>(snapped_disp_val);

	if (debug) cout << "StereoData::PushToSnapNextFartherValidDisparity() invalid disp_val " << disp_val << " at disp_label " << disp_label << " and snapped_disp_val " << snapped_disp_val << " being examined for idx_unk " << idx_unk << " because is not currently valid" << endl;

	Matrix<bool, 1, Dynamic> udv = unknown_disps_valid_[cid].row(idx_unk);
	bool *p;
	int valid_disp_label; // the new, valid disparity label
	bool valid;
	int curr_label;

	// count forward from current label, find first valid label, and record distance
	curr_label = disp_label;
	p = udv.data() + curr_label;
	valid = *p;
	while ((!valid) &&
		(curr_label < (nums_disps_[cid] - 1))) {
		p++;
		curr_label++;
		valid = *p;
	}
	if (valid) valid_disp_label = curr_label;
	else valid_disp_label = -1;

	// chose first valid label with shorter distance, or closer of the two to the camera (larger disparity value, which is larger disparity label) if they're equidistant 
	if (valid_disp_label == -1) return; // no valid label exists, so don't change disp_val

	if (debug_test) {
		cout << "StereoData::PushToSnapNextFartherValidDisparity() valid_disp_label " << valid_disp_label << endl;
		if (!unknown_disps_valid_[cid](idx_unk, valid_disp_label)) {
			cout << "StereoData::PushToSnapNextFartherValidDisparity() valid_disp_label not actually valid" << endl;
			cin.ignore();
		}
	}

	if (debug) {
		cout << "StereoData::PushToSnapNextFartherValidDisparity() valid_disp_label " << valid_disp_label << endl;
		cin.ignore();
	}

	disp_val = static_cast<double>(DispLabelToValue(cid, valid_disp_label));
}

// updates disp_val to snap it to the closest valid value in the reference image for the unknown pixel with contracted unknown space pixel index idx_unk, if one exists; if one doesn't exist, no change is made; if possible, also snaps to within GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT WS units of all immediate segment label neighbors
// if the closest valid range value is a tie, the tie goes to the value closer to the camera (a higher disparity)
// a 0 disparity value (undefined/unknown depth) is snapped to the largest valid disparity value, indicating the smallest valid depth
// disps are all disps for the image (full indices), but only those for the segment label including idx_unk are used
void StereoData::SnapDisparityToValidRange(int cid, int idx_unk, double &disp_val) {
	bool debug = false;
	bool debug_test = false;

	int disp_label;
	float disp_val_float, snapped_disp_val;

	disp_val_float = static_cast<float>(disp_val);

	// a 0 disparity value (undefined/unknown depth) is snapped to the largest valid disparity value, indicating the smallest valid depth
	if (disp_val == 0) {
		Matrix<bool, 1, Dynamic> udv = unknown_disps_valid_[cid].row(idx_unk);
		bool *p;
		bool valid;
		int curr_label = nums_disps_[cid] - 1;
		p = udv.data() + curr_label;
		curr_label = disp_label;
		valid = *p;
		while ((!valid) &&
			(curr_label > 0)) {
			p--;
			curr_label--;
			valid = *p;
		}
		if (valid) disp_val = static_cast<double>(DispLabelToValue(cid, curr_label));
		return;
	}

	disp_label = DispValueToLabel(cid, disp_val_float);
	snapped_disp_val = DispLabelToValue(cid, disp_label);
	disp_val = static_cast<double>(snapped_disp_val);
	if (unknown_disps_valid_[cid](idx_unk, disp_label)) return; // already checks out as valid

	if (debug) cout << "StereoData::SnapDisparityToValidRange() invalid disp_val " << disp_val << " at disp_label " << disp_label << " and snapped_disp_val " << snapped_disp_val << " being examined for idx_unk " << idx_unk << " because is not currently valid" << endl;

	Matrix<bool, 1, Dynamic> udv = unknown_disps_valid_[cid].row(idx_unk);
	bool *p;
	int valid_disp_label; // the new, valid disparity label
	bool valid;
	int curr_label, bwd_label, fwd_label;
	// count backward from current label, find first valid label, and record distance
	curr_label = disp_label;
	p = udv.data() + curr_label;
	valid = *p;
	while ((!valid) &&
		(curr_label > 0)) {
		p--;
		curr_label--;
		valid = *p;
	}
	if (valid) bwd_label = curr_label;
	else bwd_label = -1;

	if (debug) cout << "StereoData::SnapDisparityToValidRange() bwd_label " << bwd_label << endl;

	// count forward from current label, find first valid label, and record distance
	curr_label = disp_label;
	p = udv.data() + curr_label;
	valid = *p;
	while ((!valid) &&
		(curr_label < (nums_disps_[cid] - 1))) {
		p++;
		curr_label++;
		valid = *p;
	}
	if (valid) fwd_label = curr_label;
	else fwd_label = -1;

	if (debug) cout << "StereoData::SnapDisparityToValidRange() fwd_label " << fwd_label << endl;

	// chose first valid label with shorter distance, or closer of the two to the camera (larger disparity value, which is larger disparity label) if they're equidistant 
	if ((bwd_label == -1) &&
		(fwd_label == -1)) return; // no valid label exists, so don't change disp_val
	else if (bwd_label == -1) valid_disp_label = fwd_label;
	else if (fwd_label == -1) valid_disp_label = bwd_label;
	else {
		int bwd_dist = curr_label - bwd_label;
		int fwd_dist = fwd_label - curr_label;
		if (bwd_dist == fwd_dist) valid_disp_label = fwd_label;
		else if (bwd_dist < fwd_dist) valid_disp_label = bwd_label;
		else valid_disp_label = fwd_label;
	}

	if (debug_test) {
		if (!unknown_disps_valid_[cid](idx_unk, valid_disp_label)) {
			cout << "StereoData::SnapDisparityToValidRange() valid_disp_label not actually valid" << endl;
			cin.ignore();
		}
	}

	if (debug) {
		cout << "StereoData::SnapDisparityToValidRange() valid_disp_label " << valid_disp_label << endl;
		cin.ignore();
	}

	disp_val = static_cast<double>(DispLabelToValue(cid, valid_disp_label));
}

// updates disp_val to snap it to the closest valid value in the reference image for the unknown pixel with contracted unknown space pixel index idx_unk, if one exists; if one doesn't exist, no change is made; if possible, also snaps to within GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT WS units of all immediate segment label neighbors
// if the closest valid range value is a tie, the tie goes to the value closer to the camera (a higher disparity)
// a 0 disparity value (undefined/unknown depth) is snapped to the largest valid disparity value, indicating the smallest valid depth
// disps are all disps for the image (full indices), but only those for the segment label including idx_unk are used
// a pixel with segment label 0 should be masked out since it's in the background segment - if we encounter that here for idx_unk, return without any change to disp_val
void StereoData::SnapDisparityToValidRangeAndSmooth(int cid, int idx_unk, double &disp_val, Matrix<double, Dynamic, 1> *disps) {
	bool debug = false;
	bool debug_test = false;

	if (debug) cout << "StereoData::SnapDisparityToValidRangeAndSmooth()" << endl;

	if (debug) {
		if ((idx_unk < 0) ||
			(idx_unk >= num_unknown_pixels_[cid]))
			cout << "idx_unk out of bounds at " << idx_unk << " whereas should be in range [0, " << num_unknown_pixels_[cid] << ")" << endl;
	}
	assert((idx_unk >= 0) && (idx_unk < num_unknown_pixels_[cid]));

	// set vals for neighbor_disps; neighbor_disps are disparities of immediate neighbors that are in the same segment or 0 if don't exist in the relative position, where (1,1) is ignored as the current pixel...neighboring pixels within the same segment are required to be within GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT WS units of each other (must convert WS to CS for comparison with disp vals); if no valid label for this pixel satisfies this condition, then it is ignored; otherwise, it is followed, as well
	double depth_thresh = GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT / agisoft_to_world_scales_[cid];
	int idx_neighbor;
	unsigned int label, label_neighbor;
	int idx_full = unknown_maps_bwd_[cid](idx_unk, 0);
	Point pt = PixIndexBwdCM(idx_full, heights_[cid]);
	label = segs_[cid](pt.y, pt.x);
	if (debug) cout << "segment label " << label << endl;
	if (label == 0) { // a pixel with segment label 0 should be masked out since it's in the background segment - if we encounter that here for idx_unk, return without any change to disp_val
		if (debug) cout << "pixel has segment label 0, which should not occur - returning with no change to disp_val" << endl;
		return;
	}
	double ndisp, ndisp_high, ndisp_low, ndepth, ndepth_high, ndepth_low;
	int ndisp_label_low, ndisp_label_high;
	int min_label = 0;
	int max_label = nums_disps_[cid] - 1;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			//if (debug) cout << "examining point (" << (pt.x + i) << ", " << (pt.y + j) << ")" << endl;
			if (((pt.y + j) < 0) ||
				((pt.x + i) < 0) ||
				((pt.y + j) >= heights_[cid]) ||
				((pt.x + i) >= widths_[cid]))
				continue;
			//if (debug) cout << "passed screen space bounds check" << endl;
			label_neighbor = segs_[cid](pt.y + j, pt.x + i);
			//if (debug) cout << "label_neighbor " << label_neighbor << endl;
			if (label_neighbor != label) continue;
			//if (debug) cout << "passed segment label check" << endl;
			idx_neighbor = PixIndexFwdCM(Point(pt.x + i, pt.y + j), heights_[cid]);
			//if (debug) cout << "idx_neighbor " << idx_neighbor << endl;
			ndisp = (*disps)(idx_neighbor, 0);
			//if (debug) cout << "ndisp " << ndisp << endl;
			if (ndisp == 0) continue;
			ndepth = 1. / ndisp;
			ndepth_high = ndepth + depth_thresh;
			ndepth_low = ndepth - depth_thresh;
			if ((ndepth_high == 0) ||
				(ndepth_low == 0)) continue;
			ndisp_high = 1. / ndepth_low;
			ndisp_low = 1. / ndepth_high;
			ndisp_label_low = DispValueToLabel(cid, ndisp_low);
			ndisp_label_high = DispValueToLabel(cid, ndisp_high);
			if (ndisp_label_low > min_label) // constrain range
				min_label = ndisp_label_low;
			if (ndisp_label_high < max_label) // constrain range
				max_label = ndisp_label_high;
		}
	}

	if (debug) cout << "min_label " << min_label << ", max_label " << max_label << endl;

	int disp_label;
	float disp_val_float, snapped_disp_val;
	Matrix<bool, 1, Dynamic> udv = unknown_disps_valid_[cid].row(idx_unk);
	bool *p;
	int valid_disp_label; // the new, valid disparity label
	bool valid;
	int curr_label, bwd_label, fwd_label;

	// determine whether a valid label exists between min and max according to neighbors' disparities
	if (min_label > max_label)
		valid = false;
	else {
		curr_label = min_label;
		p = udv.data() + curr_label;
		valid = *p;
		while ((!valid) &&
			(curr_label <= max_label)) {
			p++;
			curr_label++;
			valid = *p;
		}
	}

	// if there is no valid label for meeting neighbor disparity ranges, default to snapping to a generally valid label, even though it will fail neighbor test, since no label will pass label test anyway
	if (!valid) {
		if (debug) cout << "no valid label meets neighbor disparity range constrictions" << endl;
		min_label = 0;
		max_label = nums_disps_[cid] - 1;
	}
	else {
		if (debug) cout << "valid label found for min_label " << min_label << ", max_label " << max_label << endl;
		//if (debug) cin.ignore();
	}

	disp_val_float = static_cast<float>(disp_val);

	// a 0 disparity value (undefined/unknown depth) is snapped to the largest valid disparity value, indicating the smallest valid depth
	if (disp_val == 0) {
		Matrix<bool, 1, Dynamic> udv = unknown_disps_valid_[cid].row(idx_unk);
		bool *p;
		bool valid;
		int curr_label = min(max_label, nums_disps_[cid] - 1);
		p = udv.data() + curr_label;
		curr_label = disp_label;
		valid = *p;
		while ((!valid) &&
			(curr_label > min_label)) {
			p--;
			curr_label--;
			valid = *p;
		}
		if (valid) disp_val = static_cast<double>(DispLabelToValue(cid, curr_label));
		return;
	}
	
	disp_label = DispValueToLabel(cid, disp_val_float);
	snapped_disp_val = DispLabelToValue(cid, disp_label);
	disp_val = static_cast<double>(snapped_disp_val);
	if (unknown_disps_valid_[cid](idx_unk, disp_label)) return; // already checks out as valid

	if (debug) cout << "StereoData::SnapDisparityToValidRange() invalid disp_val " << disp_val << " at disp_label " << disp_label << " and snapped_disp_val " << snapped_disp_val << " being examined for idx_unk " << idx_unk << " because is not currently valid" << endl;

	// count backward from current label, find first valid label, and record distance
	curr_label = min(disp_label, max_label);
	p = udv.data() + curr_label;
	valid = *p;
	while ((!valid) &&
		(curr_label > min_label)) {
		p--;
		curr_label--;
		valid = *p;
	}
	if (valid) bwd_label = curr_label;
	else bwd_label = -1;

	if (debug) cout << "StereoData::SnapDisparityToValidRange() bwd_label " << bwd_label << endl;

	// count forward from current label, find first valid label, and record distance
	curr_label = max(disp_label, min_label);
	p = udv.data() + curr_label;
	valid = *p;
	while ((!valid) &&
		(curr_label < max_label)) {
		p++;
		curr_label++;
		valid = *p;
	}
	if (valid) fwd_label = curr_label;
	else fwd_label = -1;

	if (debug) cout << "StereoData::SnapDisparityToValidRange() fwd_label " << fwd_label << endl;

	// chose first valid label with shorter distance, or closer of the two to the camera (larger disparity value, which is larger disparity label) if they're equidistant 
	if ((bwd_label == -1) &&
		(fwd_label == -1)) return; // no valid label exists, so don't change disp_val
	else if (bwd_label == -1) valid_disp_label = fwd_label;
	else if (fwd_label == -1) valid_disp_label = bwd_label;
	else {
		int bwd_dist = curr_label - bwd_label;
		int fwd_dist = fwd_label - curr_label;
		if (bwd_dist == fwd_dist) valid_disp_label = fwd_label;
		else if (bwd_dist < fwd_dist) valid_disp_label = bwd_label;
		else valid_disp_label = fwd_label;
	}

	if (debug_test) {
		if (!unknown_disps_valid_[cid](idx_unk, valid_disp_label)) {
			cout << "StereoData::SnapDisparityToValidRange() valid_disp_label not actually valid" << endl;
			cin.ignore();
		}
	}

	if (debug) {
		cout << "StereoData::SnapDisparityToValidRange() valid_disp_label " << valid_disp_label << endl;
		//cin.ignore();
	}

	disp_val = static_cast<double>(DispLabelToValue(cid, valid_disp_label));
}

// updates disps so that each relevant value is snapped to the closest valid value for the pixel in camera cid, if one exists; if one doesn't exist, no change is made
// if the closest valid range value is a tie, the tie goes to the value closer to the camera (a higher disparity)
// disps are disparities for all pixels, but the algorithm only considers changing disparities of unknown pixels, assuming known pixels are fine as is
// a 0 disparity value (undefined/unknown depth) is snapped to the largest valid disparity value, indicating the smallest valid depth
void StereoData::SnapDisparitiesToValidRanges(int cid, Eigen::Matrix<double, Dynamic, 1> *disps) {
	bool debug = false;

	if (debug) cout << "StereoData::SnapDisparitiesToValidRanges()" << endl;

	int idx_unk = 0;
	double disp_val;
	double *pD = disps->data();
	bool *pM = masks_unknowns_[cid].data();
	for (int idx_full = 0; idx_full < disps->rows(); idx_full++) {
		if (*pM) {
			disp_val = *pD;
			if (debug) cout << "StereoData::SnapDisparitiesToValidRanges() examining idx_unk " << idx_unk << " with disp val " << disp_val << endl;

			//SnapDisparityToValidRange(cid, idx_unk, disp_val);
			SnapDisparityToValidRangeAndSmooth(cid, idx_unk, disp_val, disps);
			*pD = disp_val;
			idx_unk++;
		}
		pM++;
		pD++;
	}
	if (debug) cout << "StereoData::SnapDisparitiesToValidRanges() completed" << endl;
}

// builds data structure crowd_disparity_proposal_
// Matrix<double, Dynamic, 1> crowd_disparity_proposal_: disparity proposal for unknown pixels only of reference camera; known pixels from all other cameras are projected into the reference screen space. All surrounding integer pixel coordinates that are unknown and masked-in receive disparity information from projected pixels, with the closest camera view direction to cid_ref winning races; all other pixels receive a value of 0
// must have built unknown_disps_valid_ before calling this function
void StereoData::BuildCrowdDisparityProposal(int cid_ref) {
	bool debug = false;
	double t;
	bool timing = true;
	if (timing) t = (double)getTickCount();

	crowd_disparity_proposal_[cid_ref].resize(num_unknown_pixels_[cid_ref], 1);
	crowd_disparity_proposal_[cid_ref].setZero();

	Matrix<double, Dynamic, 1> view_dotprods(num_unknown_pixels_[cid_ref], 1);
	view_dotprods.setConstant(-1);

	Point3f view_dir_ref = Camera::GetCameraViewDirectionWS(&RTinvs_[cid_ref]);

	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid = (*it).first;
		//if (cid == cid_ref) continue; // also do for cid_ref
		if (!valid_cam_poses_[cid]) continue; // cams with inaccurate poses are not included in mask-checking
		int num_nonzero_depths = depth_maps_[cid].count();
		if (num_nonzero_depths == 0)
			continue; // must have some depth information

		Point3f view_dir = Camera::GetCameraViewDirectionWS(&RTinvs_[cid]);
		double vdot = view_dir_ref.ddot(view_dir);

		Matrix<double, Dynamic, 4> WC(num_nonzero_depths, 4); // data structure containing homogeneous pixel positions across columns (u,v,1)
		WC.col(2).setOnes();

		Matrix<double, 1, Dynamic> depths(1, num_nonzero_depths); // for I below
		double *pDepths = depths.data();
		float *pDepth = depth_maps_[cid].data();
		double *pX = WC.col(0).data();
		double *pY = WC.col(1).data();
		double *pDisp = WC.col(3).data();
		int h = imgsT_[cid].rows;
		int w = imgsT_[cid].cols;
		float depth;
		for (int c = 0; c < w; c++) {
			for (int r = 0; r < h; r++) {
				depth = *pDepth++;
				if (depth == 0.)
					continue;
				*pX++ = static_cast<double>(c);
				*pY++ = static_cast<double>(r);
				*pDisp++ = 1. / static_cast<double>(depth);

				*pDepths++ = static_cast<double>(depth);
			}
		}
		
		// project points from cid_ref screen space to cid screen space to get depths from cid
		Matrix<float, 3, 4> Pint2out = Pss1Toss2(cid, cid_ref);
		Matrix<double, Dynamic, 3> T = WC * Pint2out.transpose().cast<double>();
		Matrix<double, Dynamic, 1> N = T.col(2).array().inverse(); // determine homogeneous coordinates to divide by
		T.col(0) = T.col(0).cwiseProduct(N); // divide by homogeneous coordinates
		T.col(1) = T.col(1).cwiseProduct(N); // divide by homogeneous coordinates

		// project points from cid screen space to cid_ref camera space to get cid_ref camera space disparity information
		Matrix<float, 1, 4> ext;
		ext << 0., 0., 0., 1.;
		Matrix4f Pout_ext;
		Pout_ext << Ps_[cid], ext;
		Matrix4f Pout_ext_inv = Pout_ext.inverse();

		Matrix<float, 4, 4> Pcs = RTs_[cid_ref] * Pout_ext_inv;
		Matrix<double, Dynamic, 4> Tcs = WC * Pcs.transpose().cast<double>();
		Tcs.col(2) = Tcs.col(2).cwiseQuotient(Tcs.col(3)); // divide by homogeneous coordinates

		double depth_new;
		double disp_new, disp_curr;
		int idx_full_out, idx_unk_out;
		for (int idx_known = 0; idx_known < T.rows(); idx_known++) {
			int x = floor(T(idx_known, 0));
			int y = floor(T(idx_known, 1));
			if ((x < 0) ||
				(y < 0) ||
				(x >= (widths_[cid_ref] - 1)) ||
				(y >= (heights_[cid_ref] - 1))) // projected location must be within valid screen space of output image
				continue;
			depth_new = Tcs(idx_known, 2); // new disparity information is taken from cid_ref camera space
			if (depth_new != 0) {
				disp_new = 1. / depth_new;
				// update appropriate output pixels that surround floating point projected location
				for (int i = 0; i <= 1; i++) {
					for (int j = 0; j <= 1; j++) {
						idx_full_out = (x + i)*heights_[cid_ref] + (y + j);
						idx_unk_out = unknown_maps_fwd_[cid_ref](idx_full_out, 0);
						if (idx_unk_out == -1) continue; // not a pixel that is masked-in with unknown depth, so don't attempt to update it
						SnapDisparityToValidRange(cid_ref, idx_unk_out, disp_new); // must snap it to ensure compliance with valid ranges
						if (disp_new != 0) { // can occur on the edges of masks due to rounding in this function when the masks are not being dilated
							disp_curr = crowd_disparity_proposal_[cid_ref](idx_unk_out, 0);
							if ((disp_curr == 0) ||
								(vdot > view_dotprods(idx_unk_out, 0))) { // the closest camera viewing direction to cid_ref wins races
								crowd_disparity_proposal_[cid_ref](idx_unk_out, 0) = disp_new;
								view_dotprods(idx_unk_out, 0) = vdot;
							}
						}
					}
				}
			}
		}
		
		if (debug) {
			cout << "StereoData::BuildCrowdDisparityProposal() cumulative result up to and including input cid " << cid << endl;
			DisplayImages::DisplayGrayscaleImageTruncated(&crowd_disparity_proposal_[cid_ref], &masks_unknowns_[cid_ref], heights_[cid_ref], widths_[cid_ref], orientations_[cid_ref]);
		}
	}

	if (debug) {
		cout << "StereoData::BuildCrowdDisparityProposal() result" << endl;
		DisplayImages::DisplayGrayscaleImageTruncated(&crowd_disparity_proposal_[cid_ref], &masks_unknowns_[cid_ref], heights_[cid_ref], widths_[cid_ref], orientations_[cid_ref]);
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "StereoData::BuildCrowdDisparityProposal() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// project points with non-zero, masked-in depth values from cid_src to cid_dest screen space and give each pixel a "vote" for its label
// Majority vote wins on label mapping, but there is a minimum threshold percentage vote.
// Also, if two labels in cid1 both claim to map to the same label in cid2, the one with more votes wins.
// Not all labels will be mapped from or to.
// The mapping can be used to help determine occlusions when calculating photoconsistency in optimization.
// Returns map of label_src => label_dest
map<int, int> StereoData::MapSegmentationLabelsAcrossImages(int cid_src, int cid_dest, Matrix<double, Dynamic, 1> *disparity_map_src) {
	bool debug = false;

	// project points
	Matrix<double, Dynamic, 3> Iss_src;
	Matrix<double, Dynamic, 3> Iss_dest;
	ProjectSS1toSS2(cid_src, cid_dest, disparity_map_src, &Iss_src, &Iss_dest);
	assert(Iss_src.rows() == Iss_dest.rows());

	// accumulate votes
	map<int, map<int, int>> votes; // map of src label => map of dest label => vote count
	double *pXsrc = Iss_src.col(0).data();
	double *pYsrc = Iss_src.col(1).data();
	double *pXdest = Iss_dest.col(0).data();
	double *pYdest = Iss_dest.col(1).data();
	double x_srcd, y_srcd, x_destd, y_destd;
	int x_src, y_src, x_dest, y_dest;
	unsigned int label_src, label_dest;
	for (int i = 0; i < Iss_src.rows(); i++) {
		x_srcd = *pXsrc++;
		y_srcd = *pYsrc++;
		x_destd = *pXdest++;
		y_destd = *pYdest++;
		x_src = round(x_srcd);
		y_src = round(y_srcd);
		x_dest = round(x_destd);
		y_dest = round(y_destd);

		if ((x_src < 0) ||
			(x_src >= widths_[cid_src]) ||
			(y_src < 0) ||
			(y_src >= heights_[cid_src]) ||
			(x_dest < 0) ||
			(x_dest >= widths_[cid_dest]) ||
			(y_dest < 0) ||
			(y_dest >= heights_[cid_dest]))
			continue;

		label_src = segs_[cid_src](y_src, x_src);
		label_dest = segs_[cid_dest](y_dest, x_dest);
		if (votes.find(label_src) == votes.end())
			votes[label_src][label_dest] = 1;
		else if (votes[label_src].find(label_dest) == votes[label_src].end())
			votes[label_src][label_dest] = 1;
		else {
			int v = votes[label_src][label_dest];
			votes[label_src][label_dest] = v + 1;
		}
	}

	// tally results
	map<int, int> mapping;
	map<int, pair<int, int>> mapping_given; // label_src => pair of label_dest and votes without comparison
	map<int, pair<int, int>> mapping_received; // label_dest => pair of label_src and votes with comparison

	for (map<int, map<int, int>>::iterator it1 = votes.begin(); it1 != votes.end(); ++it1) {
		int label_src = (*it1).first;
		int highest_vote = 0;
		unsigned int best_label_dest;
		for (map<int, int>::iterator it2 = (*it1).second.begin(); it2 != (*it1).second.end(); ++it2) {
			int label_dest = (*it2).first;
			int vote = (*it2).second;
			if (vote > highest_vote) {
				highest_vote = vote;
				best_label_dest = label_dest;
			}
		}
		pair<int, int> p;
		p.first = best_label_dest;
		p.second = highest_vote;
		mapping_given[label_src] = p;
	}

	for (map<int, pair<int, int>>::iterator it = mapping_given.begin(); it != mapping_given.end(); ++it) {
		int label_src = (*it).first;
		int label_dest = (*it).second.first;
		int vote = (*it).second.second;
		if (mapping_received.find(label_dest) == mapping_received.end()) {
			pair<int, int> p;
			p.first = label_src;
			p.second = vote;
			mapping_received[label_dest] = p;
		}
		else {
			if (vote > mapping_received[label_dest].second) {
				pair<int, int> p;
				p.first = label_src;
				p.second = vote;
				mapping_received[label_dest] = p;
			}
		}
	}

	for (map<int, pair<int, int>>::iterator it = mapping_received.begin(); it != mapping_received.end(); ++it) {
		int label_dest = (*it).first;
		int label_src = (*it).second.first;
		mapping[label_src] = label_dest;
	}

	if (debug) {
		cout << "Segment label mapping from cid " << cid_src << " to cid " << cid_dest << endl;
		for (map<int, int>::iterator it = mapping.begin(); it != mapping.end(); ++it) {
			int label1 = (*it).first;
			int label2 = (*it).second;
			cout << "label " << label1 << " is mapped to label " << label2 << endl;
		}
		cout << endl;
		cin.ignore();
	}

	return mapping;
}

// project points from cid_src's SS to cid_dest's SS, using the disparity ap for cid_src disparity_map_src
// results are stored with homogeneous SS coordinates in Iss_src and Iss_dest for all masked-in coordinates for which there is disparity information
void StereoData::ProjectSS1toSS2(int cid_src, int cid_dest, Matrix<double, Dynamic, 1> *disparity_map_src, Matrix<double, Dynamic, 3> *Iss_src, Matrix<double, Dynamic, 3> *Iss_dest) {
	// get known cid_src SS coords, but make 4D and use depth_map_src to change homogeneous values
	Matrix<double, Dynamic, 4> WC_src_tmp(heights_[cid_src] * widths_[cid_src], 4);
	WC_src_tmp.col(2).setOnes();
	double *pX = WC_src_tmp.col(0).data();
	double *pY = WC_src_tmp.col(1).data();
	double *pDisp = WC_src_tmp.col(3).data();
	bool *pM = masks_[cid_src].data();
	Matrix<bool, Dynamic, 1> known_mask(heights_[cid_src] * widths_[cid_src], 1);
	bool *pMk = known_mask.data();
	double *pDM = disparity_map_src->data();
	int num_known = 0;
	float d;
	int idx_full = 0;
	for (int c = 0; c < widths_[cid_src]; c++) {
		for (int r = 0; r < heights_[cid_src]; r++) {
			d = *pDM++;
			if ((d != 0) &&
				(*pM)) {
				*pX++ = c;
				*pY++ = r;
				*pDisp++ = d;
				num_known++;
				*pMk++ = true;
			}
			else *pMk++ = false;
			pM++;
			idx_full++;
		}
	}
	Matrix<double, Dynamic, 4> WC_src(num_known, 4);
	WC_src.block(0, 0, num_known, 4) = WC_src_tmp.block(0, 0, num_known, 4);
	Iss_src->resize(num_known, 3);
	(*Iss_src) = WC_src.block(0, 0, num_known, 3);

	Matrix<float, 1, 4> ext;
	ext << 0., 0., 0., 1.;
	Matrix4f Psrc_ext;
	Psrc_ext << Ps_[cid_src].cast<float>(), ext;
	Matrix4f Psrc_ext_inv = Psrc_ext.inverse();
	Matrix<float, 3, 4> Psrc2dest = Ps_[cid_dest].cast<float>() * Psrc_ext_inv;
	(*Iss_dest) = WC_src * Psrc2dest.transpose().cast<double>();
	Matrix<double, Dynamic, 1> H = Iss_dest->col(2).array().inverse(); // determine homogeneous coordinates to divide by
	Iss_dest->col(0) = Iss_dest->col(0).cwiseProduct(H); // divide by homogeneous coordinates
	Iss_dest->col(1) = Iss_dest->col(1).cwiseProduct(H); // divide by homogeneous coordinates
}

// tests world space coordinates from each camera against masks from all other input cameras.  any world space point that, when reprojected into another camera space, does not fall on a pixel with a true value in that camera space is considered to have a rejected depth value.  the corresponding pixel in the corresponding depth map is assigned a depth of 0 to signify that we have no valid information for that pixel
void StereoData::CleanDepths(int cid_ref, Matrix<float, Dynamic, 1> *depth_map) {
	bool debug = false;
	bool debug_display = false;

	if (debug) cout << "StereoData::CleanDepths() for image " << cid_ref << endl;

	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid = (*it).first;
		if (cid == cid_ref) continue;
		if (!valid_cam_poses_[cid]) continue; // cams with inaccurate poses are not included in mask-checking

		if (debug) cout << "...against image " << cid << endl;

		Matrix<double, Dynamic, 4> WC_tmp(heights_[cid_ref] * widths_[cid_ref], 4);
		WC_tmp.col(2).setOnes();
		double *pX = WC_tmp.col(0).data();
		double *pY = WC_tmp.col(1).data();
		double *pDisp = WC_tmp.col(3).data();
		bool *pM = masks_[cid_ref].data();
		Matrix<bool, Dynamic, 1> known_mask(heights_[cid_ref] * widths_[cid_ref], 1);
		bool *pMk = known_mask.data();
		float *pDM = depth_map->data();
		int num_known = 0;
		float z;
		int idx_full = 0;
		for (int c = 0; c < widths_[cid_ref]; c++) {
			for (int r = 0; r < heights_[cid_ref]; r++) {
				z = *pDM++;
				if ((z != 0) &&
					(*pM)) {
					*pX++ = c;
					*pY++ = r;
					*pDisp++ = 1. / static_cast<double>(z);
					num_known++;
					*pMk++ = true;
				}
				else *pMk++ = false;
				pM++;
				idx_full++;
			}
		}
		Matrix<double, Dynamic, 4> WC(num_known, 4);
		WC.block(0, 0, num_known, 4) = WC_tmp.block(0, 0, num_known, 4);

		CleanDepths_Pair(cid_ref, depth_map, &WC, &known_mask, cid);
	}

	if (debug_display) DisplayImages::DisplayGrayscaleImage(depth_map, heights_[cid_ref], widths_[cid_ref], orientations_[cid_ref]);
}

// given source and destination camera IDs, updates the source camera's depth map so that any of the source camera's world space points, when reprojected into the destination camera's screen space, that falls on a masked out pixel has its corresponding depth value in the source camera's depth map set to zero to signify an unknown depth (since the depth we had for that pixel was found to be incorrect)
// WC_known is data structure with SS coordinates of known pixels with column 0 containing X coords, col 1 containing Y coord, col 2 containing constant 1., and col 3 containing disparity (not depth) value in camera space
// mask known has a row for every pixel in the image, with true values where the pixel is "known"
void StereoData::CleanDepths_Pair(int cid_src, Matrix<float, Dynamic, 1> *src_depth_map, Matrix<double, Dynamic, 4> *WC_known, Matrix<bool, Dynamic, 1> *known_mask, int cid_dest) {
	bool debug = false;
	bool debug_tmp = false;
	bool debug_tmp2 = false;

	if (debug) cout << "Scene::CleanDepths_Pair() for image " << cid_src << " against image " << cid_dest << endl;

	// reproject known screen space coordinates of cid_src to screen space of cid_dest and normalize by homogeneous coordinates
	Matrix<float, 1, 4> ext;
	ext << 0., 0., 0., 1.;
	Matrix4f Psrc_ext;
	Psrc_ext << Ps_[cid_src].cast<float>(), ext;
	Matrix4f Psrc_ext_inv = Psrc_ext.inverse();
	Matrix<float, 3, 4> Psrc2dest = Ps_[cid_dest].cast<float>() * Psrc_ext_inv;
	Matrix<double, Dynamic, 3> T = (*WC_known) * Psrc2dest.transpose().cast<double>();
	Matrix<double, Dynamic, 1> H = T.col(2).array().inverse(); // determine homogeneous coordinates to divide by
	T.col(0) = T.col(0).cwiseProduct(H); // divide by homogeneous coordinates
	T.col(1) = T.col(1).cwiseProduct(H); // divide by homogeneous coordinates

	int h_src = heights_[cid_src];
	int w_src = widths_[cid_src];
	int h_dest = heights_[cid_dest];
	int w_dest = widths_[cid_dest];

	// interpolate depths from floating point coordinates in destination screen space
	Matrix<double, Dynamic, 1> X = T.col(0);
	Matrix<double, Dynamic, 1> Y = T.col(1);
	Matrix<bool, Dynamic, 1> inbound_mask(T.rows(), 1);
	inbound_mask.setConstant(true);
	Interpolation::InterpolateAgainstMask<double>(w_dest, h_dest, &X, &Y, &masks_dilated_[cid_dest], &inbound_mask, closeup_xmins_[cid_dest], closeup_xmaxs_[cid_dest], closeup_ymins_[cid_dest], closeup_ymaxs_[cid_dest]);

	if (debug_tmp) {
		cout << "StereoData::CleanDepths_Pair() masks_dilated_[ " << cid_dest << "]" << endl;
		DisplayImages::DisplayGrayscaleImage(&masks_dilated_[cid_dest], h_dest, w_dest, orientations_[cid_dest]);

		cout << "StereoData::CleanDepths_Pair() reprojection" << endl;
		Matrix<bool, Dynamic, Dynamic> view_reproj(h_dest, w_dest);
		view_reproj.setConstant(false);
		int x, y;
		for (int r = 0; r < T.rows(); r++) {
			x = round(T(r, 0));
			y = round(T(r, 1));
			if ((x < 0) || (x >= w_dest) || (y < 0) || (y >= h_dest)) continue;
			view_reproj(y, x) = true;
		}
		DisplayImages::DisplayGrayscaleImage(&view_reproj, h_dest, w_dest, orientations_[cid_dest]);

		cout << "StereoData::CleanDepths_Pair() inbound mask with count " << inbound_mask.count() << endl;
		DisplayImages::DisplayGrayscaleImageTruncated(&inbound_mask, known_mask, h_src, w_src, orientations_[cid_src]);
	}

	if (debug_tmp2) {
		if (inbound_mask.count() != inbound_mask.rows())
			cout << "Scene::CleanDepths_Pair() for cid_src " << cid_src << ", cid_dest " << cid_dest << " inbound_mask has " << inbound_mask.count() << " of " << inbound_mask.rows() << endl;
		else cout << "Scene::CleanDepths_Pair() for cid_src " << cid_src << ", cid_dest " << cid_dest << " inbound_mask for all pixels checked" << endl;
	}

	Matrix<bool, Dynamic, 1> inbound_mask_full(h_src*w_src, 1);
	inbound_mask_full.setZero();
	EigenMatlab::AssignByTruncatedBooleans(&inbound_mask_full, known_mask, &inbound_mask);
	float val_zero = 0.;
	EigenMatlab::AssignByBooleansNot(src_depth_map, &inbound_mask_full, val_zero);

	if (debug_tmp) {
		cout << "inbound_mask rows " << inbound_mask.rows() << ", count " << inbound_mask.count() << endl;
		cout << "inbound_mask_full rows " << inbound_mask_full.rows() << ", count " << inbound_mask_full.count() << endl;
		cout << "Scene::CleanDepths_Pair() after for src " << cid_src << " with non-zero depth count of " << src_depth_map->count() << endl;
		DisplayImages::DisplayGrayscaleImage(src_depth_map, h_src, w_src, orientations_[cid_src]);
	}
}

// tests a set of unknown pixel disparity proposals for reference camera cid_ref against the masks for all other cameras to ensure the values are not out of bounds
// disps are disparities for all pixels
// cid_ref is the camera ID to which the disparities apply; for each pixel, pass is updated to true if a disparity passes the test, false if it fails
// returns true if all pass, false if any fail; pass must be same size as disps
// requires all initialization complete
bool StereoData::TestDisparitiesAgainstMasks(const int cid_ref, Eigen::Matrix<double, Dynamic, 1> *disps, Matrix<bool, Dynamic, 1> *pass) {
	assert(disps->rows() == pass->rows());

	bool debug = false;

	Matrix<bool, Dynamic, 1> pass_used = ContractFullToUsedSize(cid_ref, pass);
	
	int num_pixels_ref = As_[cid_ref].rows();
	int num_used_ref = num_used_pixels_[cid_ref];
	Matrix<double, Dynamic, 4> WC(num_used_ref, 4);
	WC.col(2).setOnes();
	double *pWCx = WC.col(0).data();
	double *pWCy = WC.col(1).data();
	double *pWCdisp = WC.col(3).data();
	double *pDisp = disps->data();
	bool *pM = masks_[cid_ref].data();
	Point pt;
	int h = imgsT_[cid_ref].rows;
	for (int idx = 0; idx < num_pixels_ref; idx++) {
		if (!*pM++) {
			pDisp++;
			continue;
		}
		pt = PixIndexBwdCM(idx, h);
		*pWCx++ = pt.x;
		*pWCy++ = pt.y;
		*pWCdisp++ = *pDisp++;
	}

	Matrix<double, Dynamic, 3> T(num_used_ref, 3); // reprojected SS coords
	Matrix<double, Dynamic, 1> X(num_used_ref, 1); // SS X coords after reprojection
	Matrix<double, Dynamic, 1> Y(num_used_ref, 1); // SS Y coords after reprojection
	Matrix<double, Dynamic, 1> H; // to hold homogeneous coordinates after reprojection
	Matrix<int, Dynamic, 1> X_round(num_used_ref, 1);
	Matrix<int, Dynamic, 1> Y_round(num_used_ref, 1);

	bool success = true;
	pass->setConstant(true);
	pass_used.setConstant(true);
	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid = (*it).first;
		if (cid == cid_ref) continue; // ojw_stereo_optim.m sets num_in to the number of images excluding the reference image (numel(vals.I) where vals.I=images(2:end) from ojw_stereo.m), then sets vals.P to P(2:end), but takes the transpose of each by permuting rows
		if (!valid_cam_poses_[cid]) continue; // cams with inaccurate poses are not included in mask-checking

		// calculate the coordinates in the input image
		Matrix<float, 3, 4> Pout2in = Pss1Toss2(cid_ref, cid);
		T = WC * Pout2in.transpose().cast<double>();
		H = T.col(2).array().inverse(); // determine homogeneous coordinates to divide by
		T.col(0) = T.col(0).cwiseProduct(H); // divide by homogeneous coordinates
		T.col(1) = T.col(1).cwiseProduct(H); // divide by homogeneous coordinates

		X = T.col(0);
		Y = T.col(1);
		Interpolation::InterpolateAgainstMask<double>(imgsT_[cid].cols, imgsT_[cid].rows, &X, &Y, &masks_dilated_[cid], &pass_used, closeup_xmins_[cid], closeup_xmaxs_[cid], closeup_ymins_[cid], closeup_ymaxs_[cid]);

		// update pass from pass_used
		EigenMatlab::AssignByTruncatedBooleans(pass, &masks_[cid_ref], &pass_used); // pass_used only adds falses with each iteration

		if (debug) {
			if (pass_used.count() != pass_used.rows())
				cout << "for cid_ref " << cid_ref << ", cid " << cid << " pass_used has " << pass_used.count() << " of " << pass_used.rows() << endl;
			else cout << "for cid_ref " << cid_ref << ", cid " << cid << " passes for all pixels checked" << endl;


			bool *pP = pass_used.data();
			bool known;
			int num_known_failed = 0;
			int num_unknown_failed = 0;
			int idx_full;
			for (int idx_used = 0; idx_used < pass_used.rows(); idx_used++) {
				if (*pP++) continue;
				idx_full = used_maps_bwd_[cid_ref](idx_used, 0);
				known = known_depths_[cid_ref](idx_full, 0);
				if (known) num_known_failed++;
				else {
					int idx_unk = unknown_maps_fwd_[cid_ref](idx_full, 0);
					cout << "failure at unknown ID " << idx_unk << " for idx_full " << idx_full << " and cid_ref " << cid_ref << " and idx_used " << idx_used << endl;
					float disp_val = static_cast<float>((*disps)(idx_full, 0));
					cout << "disp val is " << disp_val << endl;
					int disp_label = DispValueToLabel(cid_ref, disp_val);
					cout << "disp_label is " << disp_label << endl;
					cout << "boolean validity for it is " << unknown_disps_valid_[cid_ref](idx_unk, disp_label) << endl;
					num_unknown_failed++;

					if (unknown_disps_valid_[cid_ref](idx_unk, disp_label)) { // if is marked as should be valid even though it failed here, get more info
						Point pt = PixIndexBwdCM(idx_full, h);
						Matrix<double, 1, 4> WC1(1, 4);
						WC1(0, 0) = pt.x;
						WC1(0, 1) = pt.y;
						WC1(0, 2) = 1.;
						WC1(0, 3) = (*disps)(idx_full, 0);
						
						Matrix<double, 1, 3> T1 = WC1 * Pout2in.transpose().cast<double>(); // vals.P is transpose of P; project WC into screen space of current input image; note that WC is in screen space of reference image and Ps have been altered to transform directly from reference image screen space to input image screen space
						// N = 1 ./ T(:,3);
						double x_proj = T1(0, 0) / T1(0, 2);
						double y_proj = T1(0, 1) / T1(0, 2);
						cout << "projected position of questionable point is (" << x_proj << ", " << y_proj << ")" << endl;
						int x = (int)x_proj;
						int y = (int)y_proj;
						int h = imgsT_[cid].rows;
						int k = h * x + y; // col major index of current position
						cout << "TestDisparitiesAgainstMasks() Mask[k] " << masks_dilated_[cid](k, 0) << ", Mask[k + h] " << masks_dilated_[cid](k + h, 0) << ", Mask[k+1] " << masks_dilated_[cid](k + 1, 0) << ", Mask[k+h+1] " << masks_dilated_[cid](k + h + 1, 0) << endl;

						// view reprojected pixel in dilated image for reference
						Mat img = Mat::zeros(imgsT_[cid].rows, imgsT_[cid].cols, CV_8UC3);
						imgsT_[cid].copyTo(img);

						if (GLOBAL_MASK_DILATION > 0) {
							int morph_type = MORPH_RECT; // MORPH_ELLIPSE
							int morph_size = GLOBAL_MASK_DILATION; // get rid of small regional markings
							Mat element = getStructuringElement(morph_type,
								Size(2 * morph_size + 1, 2 * morph_size + 1),
								Point(morph_size, morph_size));
							dilate(img, img, element);
						}

						Vec3b v(0, 0, 255);
						img.at<Vec3b>(y, x) = v;

						display_mat(&img, "img with reprojected point that fails test despite having valid flag in unknown_disps_valid_", orientations_[cid]);
					}
					else { // if invalid, check to ensure that snapping to a valid disparity value would make it valid as expected
						double disp_vald = static_cast<double>(disp_val);
						SnapDisparityToValidRange(cid_ref, idx_unk, disp_vald);
						float disp_valf = static_cast<float>(disp_vald);
						int disp_label_new = DispValueToLabel(cid_ref, disp_valf);

						if (unknown_disps_valid_[cid_ref](idx_unk, disp_label_new))
							cout << "After snapping the value to a valid disparity value for the unknown pixel, it checks out" << endl;
						else
							cout << "After snapping the value to a valid disparity value for the unknown pixel, it still does not check out" << endl;

						cin.ignore();
					}
				}
			}

			cout << "num_known_failed " << num_known_failed << endl;
			cout << "num_unknown_failed " << num_unknown_failed << endl;
			cin.ignore();
			
		}
	}
	
	if (pass->count() == pass->rows()) success = true;
	else success = false;

	return success;
}

// view epilines of unknown pixels from reference image in each other image
void StereoData::TestProjectionMatricesByEpilines(int cid_ref) {

	Matrix<float, 1, 4> WC(1, 4); // data structure containing homogeneous pixel positions across columns (u,v,1)
	Matrix<float, Dynamic, 3> T(1, 3);
	float x, y, h;
	float disp_val;
	cv::Scalar color(0, 0, 255);

	
	Matrix<bool, Dynamic, 1> inbounds(num_unknown_pixels_[cid_ref], 1);
	inbounds.setConstant(false);
	for (int r = 0; r < inbounds.rows(); r++) {
		inbounds(r, 0) = unknown_disps_valid_[cid_ref].row(r).any();
	}
	
	for (int idx = 0; idx < num_unknown_pixels_[cid_ref]; idx++) {
		if (inbounds(idx, 0)) continue; // only display for unknown pixels considered out of bounds

		int idx_full = unknown_maps_bwd_[cid_ref](idx, 0);
		Point pt = PixIndexBwdCM(idx_full, heights_[cid_ref]);

		// draw the point in question on image cid_ref and display it
		cv::Scalar color(0, 0, 255);
		cv::Mat outImg1(heights_[cid_ref], widths_[cid_ref], CV_8UC3);
		imgsT_[cid_ref].copyTo(outImg1);
		cv::circle(outImg1, Point(pt.x, pt.y), 3, color, -1, CV_AA);
		display_mat(&outImg1, "out of bounds unknown ref image pixel to project", orientations_[cid_ref]);
		
		WC(0, 0) = pt.x;
		WC(0, 1) = pt.y;
		WC(0, 2) = 1.;

		for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
			int cid = (*it).first;
			if (cid == cid_ref) continue; // notes in ojw_segpln.m say the images list used here excludes the reference image, but his code doesn't actually exclude it

			// ready new image for epiline of current unknown pixel
			cv::Mat outImg2(imgsT_[cid].rows, imgsT_[cid].cols, CV_8UC3);
			imgsT_[cid].copyTo(outImg2);

			for (int disp_label = 0; disp_label < disps_[cid_ref].size(); disp_label++) {
				// Vary image coordinates according to disparity
				disp_val = DispLabelToValue(cid_ref, disp_label);
				WC(0, 3) = disp_val;

				T = WC * Pout2ins_[cid_ref][cid].transpose(); // vals.P is transpose of P; project WC into screen space of current input image; note that WC is in screen space of reference image and Ps have been altered to transform directly from reference image screen space to input image screen space
				// N = 1 ./ T(:,3);
				h = T(0, 2);
				x = T(0, 0) / h;
				y = T(0, 1) / h;

				cv::circle(outImg2, Point(x, y), 1, color, -1, CV_AA);
			}

			cout << "displaying for image " << cid << endl;
			display_mat(&outImg2, "epiline from StereoData method", orientations_[cid]);
		}
	}
}

// in order to determine the proper dilation element size for masks when constraining depth values, view projections of known pixels from reference image in each other image after dilation to discern overlap visually
// inputs and outputs must be set first
void StereoData::DebugViewPoseAccuracy(int cid_ref) {
	// project known pixels into each other image and change destination pixels' red component to 255 to see accuracy of projection matrices
	Matrix<bool, Dynamic, Dynamic> known = depth_maps_[cid_ref].array() > 0.;
	int num_known = known.count();
	Matrix<float, Dynamic, 4> WCkd(num_known, 4);
	float *pWCkd_x = WCkd.data();
	float *pWCkd_y = WCkd.data() + num_known;
	float *pWCkd_z = WCkd.data() + 3 * num_known;
	float depth;
	for (int i = 0; i < num_pixels_[cid_ref]; i++) {
		Point pt = PixIndexBwdCM(i, heights_[cid_ref]);
		depth = depth_maps_[cid_ref](pt.y, pt.x);
		if (depth <= 0.) continue;
		*pWCkd_x++ = pt.x;
		*pWCkd_y++ = pt.y;
		*pWCkd_z++ = 1 / depth;
	}
	WCkd.col(2).setOnes();
	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		if (cid == cid_ref) continue;

		Matrix<float, Dynamic, 3> T2 = WCkd * Pout2ins_[cid_ref][cid].transpose();
		Matrix<float, Dynamic, 1> N2 = T2.col(2).array().inverse();
		T2.col(0) = T2.col(0).cwiseProduct(N2);
		T2.col(1) = T2.col(1).cwiseProduct(N2);

		Mat img = Mat::zeros(imgsT_[cid].rows, imgsT_[cid].cols, CV_8UC3);
		imgsT_[cid].copyTo(img);

		if (GLOBAL_MASK_DILATION > 0) {
			int morph_type = MORPH_RECT; // MORPH_ELLIPSE
			int morph_size = GLOBAL_MASK_DILATION; // get rid of small regional markings
			Mat element = getStructuringElement(morph_type,
				Size(2 * morph_size + 1, 2 * morph_size + 1),
				Point(morph_size, morph_size));
			dilate(img, img, element);
		}

		Matrix<int, Dynamic, 1> X_round2 = T2.col(0).cast<int>();
		Matrix<int, Dynamic, 1> Y_round2 = T2.col(1).cast<int>();
		int *pX = X_round2.data();
		int *pY = Y_round2.data();
		int x, y, k;
		int h = imgsT_[cid].rows;
		int w = imgsT_[cid].cols;
		for (int i = 0; i < T2.rows(); i++) {
			x = *pX++;
			y = *pY++;
			if ((x >= 0) &&
				(y >= 0) &&
				(x < w) &&
				(y < h)) {
				k = h * x + y; // col major index of current position
				Vec3b pix = img.at<Vec3b>(y, x);
				pix[2] = 255;
				img.at<Vec3b>(y, x) = pix;
			}
		}
		display_mat(&img, "test", orientations_[cid]);
	}
}

// uses unknown_disps_valid_ to find the maximum valid disparity value for an unknown pixel
// returns 0 if no valid disparity exists for the unknown pixel whose index is given
float StereoData::GetMaxValidDisparity(int cid, int unk_pix_idx) {
	int label = GetMaxValidDisparityLabel(cid, unk_pix_idx);
	if (label != 0)
		return DispLabelToValue(cid, label);
	else
		return 0.;
}

// uses unknown_disps_valid_ to find the maximum valid disparity label for an unknown pixel
// returns 0 if no valid disparity label exists for the unknown pixel whose index is given
int StereoData::GetMaxValidDisparityLabel(int cid, int unk_pix_idx) {
	Matrix<bool, 1, Dynamic> unknown_disps_valid_pixel = unknown_disps_valid_[cid].row(unk_pix_idx);
	bool *pV = unknown_disps_valid_pixel.data() + unknown_disps_valid_[cid].cols() - 1;
	int disp_label;
	bool found = false;
	for (int c = (unknown_disps_valid_[cid].cols() - 1); c >= 0; c--) { // start at max disparity, which is min depth
		if (*pV--) {
			disp_label = c;
			found = true;
			break;
		}
	}
	if (found)
		return disp_label;
	else
		return 0;
}

// uses unknown_disps_valid_ to find the maximum valid disparity value for an unknown pixel
// returns 0 if no valid disparity exists for the unknown pixel whose index is given
float StereoData::GetMinValidDisparity(int cid, int unk_pix_idx) {
	int label = GetMinValidDisparityLabel(cid, unk_pix_idx);
	if (label != 0)
		return DispLabelToValue(cid, label);
	else
		return 0.;
}

// uses unknown_disps_valid_ to find the minimum valid disparity label for an unknown pixel
// returns 0 if no valid disparity exists for the unknown pixel whose index is given
int StereoData::GetMinValidDisparityLabel(int cid, int unk_pix_idx) {
	Matrix<bool, 1, Dynamic> unknown_disps_valid_pixel = unknown_disps_valid_[cid].row(unk_pix_idx);
	bool *pV = unknown_disps_valid_pixel.data();
	int disp_label;
	bool found = false;
	for (int c = 0; c < unknown_disps_valid_[cid].cols(); c++) { // start at min disparity, which is max depth
		if (*pV++) {
			disp_label = c;
			found = true;
			break;
		}
	}
	if (found)
		return disp_label;
	else
		return 0;
}

// uses unknown_disps_valid_ to find the first max and last value before a false for an unknown pixel, where there may be multiple additional valid ranges not accounted for
// returns 0 if no valid disparity exists for the unknown pixel whose index is given
void StereoData::GetFirstMinMaxValidDisparity(int cid, int unk_pix_idx, float &max, float &min) {
	Matrix<bool, 1, Dynamic> unknown_disps_valid_pixel = unknown_disps_valid_[cid].row(unk_pix_idx);
	bool *pV = unknown_disps_valid_pixel.data() + unknown_disps_valid_[cid].cols() - 1;
	bool found_max = false;
	bool found_min = false;
	for (int c = (unknown_disps_valid_[cid].cols() - 1); c >= 0; c--) { // start at max disparity, which is min depth
		if ((*pV) &&
			(!found_max)) {
			max = DispLabelToValue(cid, c);
			found_max = true;
		}
		else if ((!*pV) &&
				 (found_max) &&
				 (!found_min)) {
			max = DispLabelToValue(cid, c + 1);
				 found_min = true;
				 break;
		}
		pV--;
	}
	if (!found_max) {
		max = 0.;
		min = 0.;
	}
	else if (!found_min) {
		min = 0.;
	}
}

void StereoData::SaveValidRangesPointCloud(string scene_name) {
	bool debug = true;

	cout << "StereoData::SaveValidRangesPointCloud()" << endl;

	std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\validranges_pointcloud.obj";
	ofstream myfile;
	myfile.open(fn);

	Matrix<float, 3, 1> Iss;
	Matrix<float, 4, 1> Iws;
	Matrix<float, 2, 1> Ics_xyonly;
	Matrix<float, 2, 3> Kinv_uvonly;
	float disp_val, depth_val, h;
	Point p_ss1;
	int idx_full_cid;

	for (map<int, Matrix<bool, Dynamic, Dynamic>>::iterator it = unknown_disps_valid_.begin(); it != unknown_disps_valid_.end(); ++it) {
		int cid = (*it).first;

		if (debug) cout << "saving valid range points for cid " << cid << endl;
		
		bool *pU = unknown_disps_valid_[cid].data();

		// for each disparity label according to unknown_disps_valid_
		for (int disp_label = 0; disp_label < unknown_disps_valid_[cid].cols(); disp_label++) {
			// for each screen space pixel in cid1's unknown_disps_valid_ that is currently valid at the current disparity label (state may change during loop, hence "currently valid")
			disp_val = DispLabelToValue(cid, disp_label);
			if (disp_val == 0) continue;
			depth_val = 1. / disp_val;

			for (int idx_unk_cid = 0; idx_unk_cid < unknown_disps_valid_[cid].rows(); idx_unk_cid++) {
				if (!*pU++) continue;

				// project the current pixel at the current disparity label into cid2's camera space and screen space

				// scale u,v,w by the desired depth amount to get homogeneous coordinates that reflect the depth after transformation
				idx_full_cid = unknown_maps_bwd_[cid][idx_unk_cid];
				p_ss1 = PixIndexBwdCM(idx_full_cid, heights_[cid]);
				Iss(0, 0) = p_ss1.x * depth_val;
				Iss(1, 0) = p_ss1.y * depth_val;
				Iss(2, 0) = depth_val;

				// transform screen space to camera space - transform u,v to x,y, then add rows for z (equal to each depth value) and w (equal to 1.0)
				Kinv_uvonly.row(0) = Kinvs_[cid].row(0);
				Kinv_uvonly.row(1) = Kinvs_[cid].row(1);
				Ics_xyonly = Kinv_uvonly * Iss; // Ics is homogeneous 4xn matrix of camera space points
				Iws(0, 0) = Ics_xyonly(0, 0);
				Iws(1, 0) = Ics_xyonly(1, 0);
				// in camera space, set z to depth value and w to 1 (already scaled x,y in homogeneous screen space)
				Iws(2, 0) = depth_val;
				Iws(4, 0) = 1.;

				// transform camera space positions to world space
				Iws = RTinvs_[cid] * Iws; // Iws is homogeneous 4xn matrix of world space points; RTinv includes transformation from Agisoft space to world space
				// normalize by homogeneous value
				h = 1 / Iws(3, 0);
				Iws *= h;
				
				myfile << "v " << Iws(0, 0) << " " << Iws(1, 0) << " " << Iws(2, 0) << endl;
			}
		}
	}

	myfile.close();
}

// for each camera, project its unknown pixels 
void StereoData::CrossCheckValidDisparityRanges(int cid_ref) {

	bool debug = true;

	Matrix<float, 3, Dynamic> Iss1;
	Matrix<float, 4, Dynamic> Iws;
	Matrix<float, 2, 1> Ics1_xyonly;
	Matrix<float, 2, 3> Kinv1_uvonly;
	Matrix<float, 4, Dynamic> Ics2;
	Matrix<float, 3, Dynamic> Iss2;
	float disp_val1, depth_val1, disp_val2, depth_val2, h;
	Point p_ss1, p_ss2;
	Point2f p_ss2f;
	int disp_label2, idx_full_cid1, idx_full_cid2, idx_unk_cid2;

	for (map<int, Matrix<bool, Dynamic, Dynamic>>::iterator it1 = unknown_disps_valid_.begin(); it1 != unknown_disps_valid_.end(); ++it1) {
		int cid1 = (*it1).first;

		Kinv1_uvonly.row(0) = Kinvs_[cid1].row(0);
		Kinv1_uvonly.row(1) = Kinvs_[cid1].row(1);

		for (map<int, Matrix<bool, Dynamic, Dynamic>>::iterator it2 = unknown_disps_valid_.begin(); it2 != unknown_disps_valid_.end(); ++it2) {
			int cid2 = (*it2).first;
			if (cid2 == cid1) continue;

			bool *pU = unknown_disps_valid_[cid1].data();
			// for each disparity label according to unknown_disps_valid_
			for (int disp_label1 = 0; disp_label1 < unknown_disps_valid_[cid1].cols(); disp_label1++) {
				// for each screen space pixel in cid1's unknown_disps_valid_ that is currently valid at the current disparity label (state may change during loop, hence "currently valid")
				disp_val1 = DispLabelToValue(cid1, disp_label1);
				if (disp_val1 == 0) continue;
				depth_val1 = 1. / disp_val1;

				for (int idx_unk_cid1 = 0; idx_unk_cid1 < unknown_disps_valid_[cid1].rows(); idx_unk_cid1++) {
					if (!*pU) continue; // ignore those already currently invalid (state may change during loop, hence "currently valid")

					// project the current pixel at the current disparity label into cid2's camera space and screen space

					// scale u,v,w by the desired depth amount to get homogeneous coordinates that reflect the depth after transformation
					idx_full_cid1 = unknown_maps_bwd_[cid1][idx_unk_cid1];
					p_ss1 = PixIndexBwdCM(idx_full_cid1, heights_[cid1]);
					Iss1(0, 0) = p_ss1.x * depth_val1;
					Iss1(1, 0) = p_ss1.y * depth_val1;
					Iss1(2, 0) = depth_val1;

					// transform screen space to camera space - transform u,v to x,y, then add rows for z (equal to each depth value) and w (equal to 1.0)
					Ics1_xyonly = Kinv1_uvonly * Iss1; // Ics is homogeneous 4xn matrix of camera space points
					Iws(0, 0) = Ics1_xyonly(0, 0);
					Iws(1, 0) = Ics1_xyonly(1, 0);
					// in camera space, set z to depth value and w to 1 (already scaled x,y in homogeneous screen space)
					Iws(2,0) = depth_val1;
					Iws(1, 0) = 1.;

					// transform camera space positions to world space
					Iws = RTinvs_[cid1] * Iws; // Iws is homogeneous 4xn matrix of world space points; RTinv includes transformation from Agisoft space to world space
					// normalize by homogeneous value
					h = 1 / Iws(3, 0);
					Iws *= h;

					// reproject world space coordinates to cid2's camera space so can get nearest cid2 disparity label
					Ics2 = RTs_[cid2] * Iws; // note the matrix multiplication property: Ainv * A = A * Ainv
					h = 1 / Ics2(3, 0);
					Ics2 *= h;
					depth_val2 = Ics2(2, 0);
					if (depth_val2 == 0) continue;
					disp_val2 = 1 / depth_val2;

					// reproject world space coordinates to cid2's screen space so can get nearest cid2 pixel
					Iss2 = Ps_[cid2] * Iws; // note the matrix multiplication property: Ainv * A = A * Ainv
					h = 1 / Iss2(2, 0);
					Iss2 *= h;
					p_ss2f.x = Iss2(0, 0);
					p_ss2f.y = Iss2(1, 0);

					// find the nearest pixel and the nearest disparity label
					disp_label2 = DispValueToLabel(cid2, disp_val2);
					p_ss2.x = round(p_ss2f.x);
					p_ss2.y = round(p_ss2f.y);

					// if the nearest pixel isn't valid at the nearest disparity label according to cid2, change the current pixel's status for cid1 at the current disparity label to invalid
					idx_full_cid2 = PixIndexFwdCM(p_ss2, heights_[cid2]);
					idx_unk_cid2 = unknown_maps_fwd_[cid2][idx_full_cid2];
					if (!unknown_disps_valid_[cid2](idx_unk_cid2, disp_label2))
						*pU = false;

					pU++; // advance to the next element of unknown_disps_valid_[cid1]
				}
			}
		}
	}
}

void StereoData::BuildAllValidDisparityRanges() {
	bool debug = true;

	cout << "StereoData::BuildAllValidDisparityRanges()" << endl;

	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid = (*it).first;
		if (!valid_cam_poses_[cid]) continue;

		BuildValidDisparityRanges(cid);

		if (debug) cout << "completed BuildValidDisparityRanges(cid) for cid " << cid << endl;
	}
	// once all are built, cross-check them
	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid = (*it).first;
		if (!valid_cam_poses_[cid]) continue;

		CrossCheckValidDisparityRanges(cid);

		if (debug) cout << "completed CrossCheckValidDisparityRanges(cid) for cid " << cid << endl;
	}
}

// builds data structure unknown_disps_valid_ : used to determine whether reprojection is masked in or out of destination screen space for cameras not otherwise included in stereo reconstruction; first dimension is unknown pixel's location index among unknown indexing coordinates; second dimension is the quantized disparity label; data structure applies to the reference image against all other images
// InitPixelData() and UpdateDisps() must be called before this function
// also project the other camera's known pixels into camera cid_ref's screen space and use the resulting camera space depths as maximum valid depths if are closer than current maximum valid depth (and convert to disparities appropriately, where max becomes min)
// cull the result set so that if there is a break in valid disparity labels for a pixel, only the nearest set (higher disparity) it retained as valid and the rest is set to invalid to appropriately model occlusions
// since camera poses are not exact, some disparities may fail a valid range on a camera due to pose estimation error for that camera.  This is especially common for thin areas of the mask.  To combat this issue, only require the pixel disparity to pass a certain percentage of cameras' masks, assuming it may fail some due to the error.  And if no disparities qualify for the masked-in pixel, assume the disparities that pass the most cameras must since it was masked-in.
void StereoData::BuildValidDisparityRanges(int cid_ref) {
	bool debug_display = false;
	bool debug = true;
	bool timing = true;
	double t_init, t, t_loop;
	if (timing) {
		t_init = (double)getTickCount();
		t = (double)getTickCount();
	}

	cout << "StereoData::BuildValidDisparityRanges()" << endl;

	if (debug) cout << "computing for " << disps_[cid_ref].size() << " disparities" << endl;

	Matrix<int, Dynamic, Dynamic> unknown_disps_valid_votes(num_unknown_pixels_[cid_ref], nums_disps_[cid_ref]);

	Matrix<double, Dynamic, 4> WC(num_unknown_pixels_[cid_ref], 4); // data structure containing homogeneous pixel positions across columns (u,v,1)
	WC.col(0) = Xunknowns_[cid_ref];
	WC.col(1) = Yunknowns_[cid_ref];
	WC.col(2).setOnes();
	Matrix<double, Dynamic, 3> T(num_unknown_pixels_[cid_ref], 3);
	Matrix<double, Dynamic, 1> N(num_unknown_pixels_[cid_ref], 1);
	Matrix<double, Dynamic, 1> X(num_unknown_pixels_[cid_ref], 1);
	Matrix<double, Dynamic, 1> Y(num_unknown_pixels_[cid_ref], 1);

	Matrix<bool, Dynamic, 1> InboundsMask(num_unknown_pixels_[cid_ref], 1);
	Matrix<int, Dynamic, 1> InboundsVotes(num_unknown_pixels_[cid_ref], 1); // each input image gets 1 vote per pixel as to whether it is inbounds or out of bounds; every masked-in pixel must be inbounds for InboundsMask somewhere, so in cases where no pixel gets positive votes from every input image, set the ones with the most votes to be true in InboundsMask
	double disp_val;

	int loop_count = 0;
	if (timing) t_loop = (double)getTickCount();
	for (int disp_label = 0; disp_label < disps_[cid_ref].size(); disp_label++) {
		// Vary image coordinates according to disparity
		disp_val = static_cast<double>(DispLabelToValue(cid_ref, disp_label));
		//if (debug) cout << "StereoData::BuildValidDisparityRanges() disp_val " << disp_val << " is label number " << disp_label << " of " << disps_[cid_ref].size() << endl;
		WC.col(3).setConstant(disp_val);

		InboundsVotes.setZero();

		for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
			int cid = (*it).first;
			if (cid == cid_ref) continue; // notes in ojw_segpln.m say the images list used here excludes the reference image, but his code doesn't actually exclude it
			if (!valid_cam_poses_[cid]) continue; // cams with inaccurate poses are not included in mask-checking

			//if (debug) cout << "StereoData::BuildValidDisparityRanges() disp_label " << disp_label << " projected into cid " << cid << endl;

			T = WC * Pout2ins_[cid_ref][cid].transpose().cast<double>();

			N = T.col(2).array().inverse(); // determine homogeneous coordinates to divide by
			T.col(0) = T.col(0).cwiseProduct(N); // divide by homogeneous coordinates
			T.col(1) = T.col(1).cwiseProduct(N); // divide by homogeneous coordinates

			X = T.col(0);
			Y = T.col(1);
			InboundsMask.setConstant(true); // assume true especially to consider out of bounds values in cases of photos that don't fully frame the object of interest
			Interpolation::InterpolateAgainstMask<double>(imgsT_[cid].cols, imgsT_[cid].rows, &X, &Y, &masks_valid_[cid], &InboundsMask, closeup_xmins_[cid], closeup_xmaxs_[cid], closeup_ymins_[cid], closeup_ymaxs_[cid]);

			InboundsVotes = InboundsVotes + InboundsMask.cast<int>();
		}

		unknown_disps_valid_votes.col(disp_label) = InboundsVotes;

		if ((timing) &&
			(debug)) {
			if ((loop_count % 100 == 0) &&
				(loop_count > 0)) {
				t_loop = (double)getTickCount() - t_loop;
				cout << "StereoData::BuildValidDisparityRange() loop of 100 disps ending at loop_count " << loop_count << " had running time = " << t_loop*1000. / getTickFrequency() << " ms" << endl;
				t_loop = (double)getTickCount();
			}
			loop_count++;
		}
	}

	// update unknown_disps_valid_[cid_ref]
	int vote_hurdle = round(static_cast<float>(As_.size()) * GLOBAL_RATIO_PASS_MASKS_TO_BE_VALID);
	unknown_disps_valid_[cid_ref].resize(num_unknown_pixels_[cid_ref], nums_disps_[cid_ref]);
	unknown_disps_valid_[cid_ref].setConstant(false);
	for (int r = 0; r < unknown_disps_valid_[cid_ref].rows(); r++) { // each row represents an unknown pixel
		int max = unknown_disps_valid_votes.row(r).maxCoeff();
		unknown_disps_valid_[cid_ref].row(r) = (unknown_disps_valid_votes.row(r).array() >= vote_hurdle).select(Matrix<bool, Dynamic, Dynamic>::Constant(unknown_disps_valid_[cid_ref].rows(), unknown_disps_valid_[cid_ref].cols(), true), unknown_disps_valid_[cid_ref].row(r));
	}
	Matrix<bool, 1, Dynamic> missingValids(1, unknown_disps_valid_[cid_ref].cols());
	for (int r = 0; r < unknown_disps_valid_[cid_ref].rows(); r++) { // each row represents an unknown pixel
		missingValids = unknown_disps_valid_[cid_ref].row(r).array() == false;
		int max = unknown_disps_valid_votes.row(r).maxCoeff();
		unknown_disps_valid_[cid_ref].row(r) = ((unknown_disps_valid_votes.row(r).array() == max) && (missingValids.array() == true)).select(Matrix<bool, Dynamic, Dynamic>::Constant(unknown_disps_valid_[cid_ref].rows(), unknown_disps_valid_[cid_ref].cols(), true), unknown_disps_valid_[cid_ref].row(r));
	}

	if ((timing) &&
		(debug)) {
		t = (double)getTickCount() - t;
		cout << "StereoData::BuildValidDisparityRange() determination running time = " << t*1000. / getTickFrequency() << " ms" << endl;
		t = (double)getTickCount();
	}

	/*
	// cull the result set so that if there is a break in valid disparity labels for a pixel, only the nearest set (higher disparity) it retained as valid and the rest is set to invalid to appropriately model occlusions
	for (int r = 0; r < unknown_disps_valid_[cid_ref].rows(); r++) { // for each pixel
		if (unknown_disps_valid_[cid_ref].row(r).count() == 0) continue;
		bool *pV = unknown_disps_valid_[cid_ref].row(r).data() + (unknown_disps_valid_[cid_ref].cols() - 1);
		bool found_first_valid = false;
		for (int c = (unknown_disps_valid_[cid_ref].cols() - 1); c >= 0; c--) { // for each disparity label, in reverse order to start with max disparity / min depth
			if (*pV) found_first_valid = true;
			else if (found_first_valid) { // invalid and already found first valid
				unknown_disps_valid_[cid_ref].block(r, 0, 1, c).setConstant(false); // excludes column c (only sets values for columns 0 through c-1), but that's ok because column c is already false
				break; // move on to next pixel
			}
			pV--;
		}
	}
	*/

	if (debug_display) {
		//cout << "cid_restrict " << cid_restrict << endl; // uncomment if want to debug restriction method
		Matrix<bool, Dynamic, 1> inbounds_all(num_unknown_pixels_[cid_ref], 1);
		//inbounds_all.setConstant(true);
		//cout << "Displaying all unknown pixels" << endl;
		//DisplayImages::DisplayGrayscaleImageTruncated(&inbounds_all, &masks_unknowns_[cid_ref], heights_[cid_ref], widths_[cid_ref], orientations_[cid_ref]);
		inbounds_all.setConstant(false);
		for (int r = 0; r < inbounds_all.rows(); r++) {
			inbounds_all(r, 0) = unknown_disps_valid_[cid_ref].row(r).any();
		}
		cout << "Displaying unknown pixels with at least one valid disparity" << endl;
		DisplayImages::DisplayGrayscaleImageTruncated(&inbounds_all, &masks_unknowns_[cid_ref], heights_[cid_ref], widths_[cid_ref], orientations_[cid_ref]);
	}
}

// builds data structure unknown_disps_valid_ : used to determine whether reprojection is masked in or out of destination screen space for cameras not otherwise included in stereo reconstruction; first dimension is unknown pixel's location index among unknown indexing coordinates; second dimension is the quantized disparity label; data structure applies to the reference image against all other images
// InitPixelData() and UpdateDisps() must be called before this function
// also project the other camera's known pixels into camera cid_ref's screen space and use the resulting camera space depths as maximum valid depths if are closer than current maximum valid depth (and convert to disparities appropriately, where max becomes min)
// cull the result set so that if there is a break in valid disparity labels for a pixel, only the nearest set (higher disparity) it retained as valid and the rest is set to invalid to appropriately model occlusions
// compare only to cameras within GLOBAL_MAX_ANGLE_DEGREES_BTWN_CAM_VIEW_DIRS_FOR_POSE_TRUST degrees in view direction
// Algorithm:
// I. Project rays from cid_ref through all unknown pixels into WS (also accumulate unknown points in data structure for use in computing epilines later)
// II. For each input image,
//		1. Compute epilines for all unknown pixels in cid_ref
//		2. For each epiline / unknown pixel in cid_ref,
//			a. Find where epiline intersects screen space bounds and only continue if line passes through screen space
//			b. Find all pairs of start/stop points for masked-in segments of the epiline by iterating along line segment pixels between screen space endpoints of the epiline
//			c. For each masked-in epiline line segment:
//				i. Cast rays from cid through start and stop points into WS
//				ii. Intersect cid_ref ray with start and stop rays, adding a tolerance for error
//				iii. Convert start and stop intersect points from WS to cid_ref CS using RTs_[cid_ref]
//				iv. Find cid_ref CS disparity labels of start and stop intersect points
//				v. Block assign valid = true for all disparity labels between start and stop intersection point disparity labels
void StereoData::BuildValidDisparityRanges_alt(int cid_ref) {
	bool debug_static = true;
	bool debug_tmp = true;
	bool debug_lineiter = false;
	bool timing = true;
	double t_init, t, t_loop;
	if (timing) {
		t_init = (double)getTickCount();
		t = (double)getTickCount();
	}

	cout << "StereoData::BuildValidDisparityRanges()" << endl;

	unknown_disps_valid_[cid_ref].resize(num_unknown_pixels_[cid_ref], nums_disps_[cid_ref]);
	unknown_disps_valid_[cid_ref].setConstant(false);

	// set up some variables and data structures
	Matrix<double, 4, 3> Pinv_ref = Pinvs_[cid_ref].cast<double>();
	Matrix4d RTinv_ref = RTinvs_[cid_ref].cast<double>();
	Point3d cam_pos_ref = Camera::GetCameraPositionWS(&RTinv_ref);
	float a, b, c; // ax + by + c = 0
	Point2f top, left, right, bottom;
	top.y = 0.;
	left.x = 0.;
	Point endpoint1, endpoint2, linepoint;
	Matrix<bool, Dynamic, 1> InboundsMask(1, disps_[cid_ref].size()); // mask for a pixel in cid_ref across all disparity labels
	Point2d pt_disp0, pt_disp1;
	Point pt_disp0r; // rounded version of pt_disp0

	// I. Project rays from cid_ref through all unknown pixels into WS (also accumulate unknown points in data structure for use in computing epilines later)
	vector<Point2f> unk_points;
	vector<Point3d> unk_point_ray_dirs;
	double *pX = Xunknowns_[cid_ref].data();
	double *pY = Yunknowns_[cid_ref].data();
	for (int r = 0; r < num_unknown_pixels_[cid_ref]; r++) {
		Point2f pf; 
		pf.x = static_cast<float>(*pX);
		pf.y = static_cast<float>(*pY);
		unk_points.push_back(pf);

		Point3d ray_dir;
		Point p = pf;
		ray_dir = Math3d::CastRayThroughPixel(cam_pos_ref, p, Pinv_ref);
		unk_point_ray_dirs.push_back(ray_dir);

		pX++;
		pY++;
	}
	
	// II. For each input image,
	if (timing) t_loop = (double)getTickCount();
	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		if (cid == cid_ref) continue; // notes in ojw_segpln.m say the images list used here excludes the reference image, but his code doesn't actually exclude it
		if (!valid_cam_poses_[cid]) continue; // cams with inaccurate poses are not included in mask-checking
		
		Matrix<double, 4, 3> Pinv = Pinvs_[cid].cast<double>();
		Matrix4d RTinv = RTinvs_[cid].cast<double>();
		Point3d cam_pos = Camera::GetCameraPositionWS(&RTinv);

		// 1. Compute epilines for all unknown pixels in cid_ref
		// compute all epiline equations Ax+By+C=0
		Matrix3f F = Math3d::ComputeFundamentalMatrix(cam_pos_ref, Pinvs_[cid_ref], Ps_[cid]);
		//if (debug_lineiter) DebugPrintMatrix(&F, "F");
		Mat Fcv = Mat::zeros(3, 3, CV_32F);
		EigenOpenCV::eigen2cv(F, Fcv);
		vector<cv::Vec3f> epilines;
		computeCorrespondEpilines(
			unk_points, //cv::Mat(unk_points), // image points : N x 1 or  1 x N matrix of type CV_32FC2 or vector<Point2f> .
			1,                   // in image 1 (can also be 2)
			Fcv, // F matrix
			epilines);     // vector of epipolar lines

		
		// 2. For each epiline / unknown pixel in cid_ref,
		int idx_unk = 0; // unk_points that generated epilines were in compact unknown pixel index order for cid_ref
		//if (debug_lineiter) cout << "investigating " << epilines.size() << " epilines in cid for " << num_unknown_pixels_[cid_ref] << " unknown pixels in cid_ref" << endl;
		for (vector<Vec3f>::iterator ite = epilines.begin(); ite != epilines.end(); ++ite) {
			InboundsMask.setConstant(false);
			
			Vec3f epiline = (*ite);
			a = epiline[0];
			b = epiline[1];
			c = epiline[2];

			//if (debug_lineiter) {
				//cout << "investigating epiline a " << a << ", b " << b << ", c " << c << endl;
				//cout << "associated unknown point in cid_ref is (" << unk_points[idx_unk].x << ", " << unk_points[idx_unk].y << ")" << endl;
			//}

			// a. Find where epiline intersects screen space bounds and only continue if line passes through screen space
			vector<Point2f> endpoints;
			right.x = static_cast<float>(widths_[cid] - 1);
			bottom.y = static_cast<float>(heights_[cid] - 1);
			top.x = -1 * (c + (b*top.y)) / a;
			bottom.x = -1 * (c + (b*bottom.y)) / a;
			left.y = -1 * (c + (a*left.x)) / b;
			right.y = -1 * (c + (a*right.x)) / b;

			//if (debug_lineiter) {
				//cout << "top " << top << ", bottom " << bottom << ", left " << left << ", right " << right << endl;
				//cin.ignore();
			//}

			if ((top.x >= 0) &&
				(top.x < widths_[cid]))
				endpoints.push_back(top);
			if ((bottom.x >= 0) &&
				(bottom.x < widths_[cid]))
				endpoints.push_back(bottom);
			if ((left.y >= 0) &&
				(left.y < heights_[cid]) &&
				(endpoints.size() < 2))
				endpoints.push_back(left);
			if ((right.y >= 0) &&
				(right.y < heights_[cid]) &&
				(endpoints.size() < 2))
				endpoints.push_back(right);

			if (endpoints.size() != 2) { // line does not go through image cid and will not intersect masked-in pixels
				idx_unk++;
				continue;
			}

			// line goes through image cid and may intersect masked-in pixels
			endpoint1.x = round(endpoints[0].x);
			endpoint1.y = round(endpoints[0].y);
			endpoint2.x = round(endpoints[1].x);
			endpoint2.y = round(endpoints[1].y);

			//if (debug_lineiter) cout << "2 endpoints " << endpoint1 << " and " << endpoint2 << endl;

			// b. Find all pairs of start/stop points for masked-in segments of the epiline by iterating along line segment pixels between screen space endpoints of the epiline
			map<int, Point> lineseg_starts;
			map<int, Point> lineseg_ends;
			int lineseg_num = 0;
			Point last_linepoint = endpoint1;
			bool inside = false;
			LineIterator itl(imgMasks_valid_[cid], endpoint1, endpoint2, 8); // create line iterator to compute pixels endpoint2 the line defined by endpoints
			for (int i = 0; i < itl.count; i++, ++itl) {
				linepoint = itl.pos();
				int val = imgMasks_valid_[cid].at<uchar>(linepoint.y, linepoint.x);
				//if (debug_lineiter) cout << "investigating pixel " << linepoint << " at mask value " << val << endl;

				if ((val != 0) &&
					(!inside)) { // just found start point of masked-in segment
					lineseg_starts[lineseg_num] = linepoint;
				}
				else if ((val == 0) &&
					(inside))  { // just passed end point of masked-in segment
					lineseg_ends[lineseg_num] = last_linepoint;
					lineseg_num++;
				}

				if (val != 0) inside = true;
				else inside = false;

				last_linepoint = linepoint;
			}

			// c. For each masked-in epiline line segment:
			for (map<int, Point>::iterator itls = lineseg_starts.begin(); itls != lineseg_starts.end(); ++itls) {
				// i. Cast rays from cid through start and stop points into WS
				int curr_lineseg = (*itls).first;
				Point startpt, endpt;
				startpt = (*itls).second;
				if (lineseg_ends.find(curr_lineseg) == lineseg_ends.end())
					endpt = endpoint2;
				else endpt = lineseg_ends[curr_lineseg];
				if (debug_lineiter) cout << "startpt " << startpt << ", endpt " << endpt << endl;
				Point3d start_ray_dir = Math3d::CastRayThroughPixel(cam_pos, startpt, Pinv);
				Point3d end_ray_dir = Math3d::CastRayThroughPixel(cam_pos, endpt, Pinv);
				if (debug_lineiter) cout << "start_ray_dir " << start_ray_dir << ", end_ray_dir " << end_ray_dir << endl;
				
				// ii. Intersect cid_ref ray with start and stop rays, adding a tolerance for error
				bool start_skew, end_skew;
				Point3d start_intersect_pt, end_intersect_pt;
				bool start_intersects, end_intersects;
				start_intersects = Math3d::IntersectionLines(cam_pos_ref, unk_point_ray_dirs[idx_unk], cam_pos, start_ray_dir, start_intersect_pt, start_skew, 0.001);
				end_intersects = Math3d::IntersectionLines(cam_pos_ref, unk_point_ray_dirs[idx_unk], cam_pos, end_ray_dir, end_intersect_pt, end_skew, 0.001);
				if (debug_lineiter) cout << "start_intersect_pt " << start_intersect_pt << ", end_intersect_pt " << end_intersect_pt << endl;
				if (debug_lineiter) cout << "start_intersects " << start_intersects << ", end_intersects " << end_intersects << endl;
				if (debug_lineiter) cout << "start_skew " << start_skew << ", end_skew " << end_skew << endl;

				// iii. Convert start and stop intersect points from WS to cid_ref CS using RTs_[cid_ref]
				Matrix<float, 4, 2> intersect_pts;
				intersect_pts.row(3).setConstant(1.);
				intersect_pts(0, 0) = start_intersect_pt.x;
				intersect_pts(1, 0) = start_intersect_pt.y;
				intersect_pts(2, 0) = start_intersect_pt.z;
				intersect_pts(0, 1) = end_intersect_pt.x;
				intersect_pts(1, 1) = end_intersect_pt.y;
				intersect_pts(2, 1) = end_intersect_pt.z;
				Matrix<float, 4, 2> intersect_pts_CS_ref = RTs_[cid_ref] * intersect_pts;

				// iv. Find cid_ref CS disparity labels of start and stop intersect points
				if (debug_lineiter) cout << "minimum cam depth " << min_depths_[cid_ref] << ", maximum cam depth " << max_depths_[cid_ref] << endl;
				if (debug_lineiter) cout << "minimum cam disparity " << min_disps_[cid_ref] << ", maximum cam disparity " << max_disps_[cid_ref] << endl;
				float start_depth_val = intersect_pts_CS_ref(2, 0) / intersect_pts_CS_ref(3, 0); // normalize depth in the process
				float end_depth_val = intersect_pts_CS_ref(2, 1) / intersect_pts_CS_ref(3, 1); // normalize depth in the process
				if (debug_lineiter) cout << "start_depth_val " << start_depth_val << ", end_depth_val " << end_depth_val << endl;
				if ((start_depth_val == 0) ||
					(end_depth_val == 0)) {
					if (debug_lineiter) {
						cout << "zero depth value present" << endl;
						cin.ignore();
					}
					idx_unk++;
					continue;
				}
				float start_disp_val = 1. / start_depth_val;
				float end_disp_val = 1. / end_depth_val;
				if (debug_lineiter) cout << "start_disp_val " << start_disp_val << ", end_disp_val " << end_disp_val << endl;
				int start_disp_label = DispValueToLabel(cid_ref, start_disp_val);
				int end_disp_label = DispValueToLabel(cid_ref, end_disp_val);
				if (debug_lineiter) cout << "start_disp_label " << start_disp_label << ", end_disp_label " << end_disp_label << endl;

				// v. Block assign valid = true for all disparity labels between start and stop intersection point disparity labels
				int min_label = min(start_disp_label, end_disp_label);
				int max_label = max(start_disp_label, end_disp_label);
				int num_labels = max_label - min_label + 1;
				if (debug_lineiter) cout << "min_label " << min_label << ", max_label " << max_label << ", num_labels " << num_labels << endl;
				InboundsMask.block(idx_unk, min_label, 1, num_labels).setConstant(true);
				if (debug_lineiter) cout << "succeeded" << endl;
			}
			
			if (debug_lineiter) cout << "about to finish pixel" << endl;
			unknown_disps_valid_[cid_ref].row(idx_unk) = InboundsMask;
			idx_unk++;
			if (debug_lineiter) {
				cout << "finished pixel idx_unk " << idx_unk << endl;
				//cin.ignore();
			}

			//if (idx_unk % 1000 == 0) cout << "finished pixel idx_unk " << idx_unk << endl;
		}
		if ((timing) &&
			(debug_tmp)) {
			t_loop = (double)getTickCount() - t_loop;
			cout << "StereoData::BuildValidDisparityRange() loop for cid " << cid << " running time = " << t_loop*1000. / getTickFrequency() << " ms" << endl;
			t_loop = (double)getTickCount();
			if (debug_lineiter) cin.ignore();
		}
	}
	if ((timing) &&
		(debug_tmp)) {
		t = (double)getTickCount() - t;
		cout << "StereoData::BuildValidDisparityRange() determination running time = " << t*1000. / getTickFrequency() << " ms" << endl;
		t = (double)getTickCount();
	}

	/*
	// cull the result set so that if there is a break in valid disparity labels for a pixel, only the nearest set (higher disparity) it retained as valid and the rest is set to invalid to appropriately model occlusions
	for (int r = 0; r < unknown_disps_valid_[cid_ref].rows(); r++) { // for each pixel
	if (unknown_disps_valid_[cid_ref].row(r).count() == 0) continue;
	bool *pV = unknown_disps_valid_[cid_ref].row(r).data() + (unknown_disps_valid_[cid_ref].cols() - 1);
	bool found_first_valid = false;
	for (int c = (unknown_disps_valid_[cid_ref].cols() - 1); c >= 0; c--) { // for each disparity label, in reverse order to start with max disparity / min depth
	if (*pV) found_first_valid = true;
	else if (found_first_valid) { // invalid and already found first valid
	unknown_disps_valid_[cid_ref].block(r, 0, 1, c).setConstant(false); // excludes column c (only sets values for columns 0 through c-1), but that's ok because column c is already false
	break; // move on to next pixel
	}
	pV--;
	}
	}
	*/

	// mask out unknown pixels with no valid range of quantized disparity labels and, if necessary, rebuild unknown_disps_valid_[cid_ref], pixel knowledge, and dilated masks for the camera
	Matrix<bool, Dynamic, 1> valid(unknown_disps_valid_[cid_ref].rows(), 1);
	Matrix<bool, Dynamic, 1> mask_unknowns = masks_unknowns_[cid_ref];
	valid.setConstant(true);
	bool *pV = valid.data();
	int idx_new = 0;
	int num_invalid = 0;
	Point p;
	for (int r = 0; r < unknown_disps_valid_[cid_ref].rows(); r++) {
		int num_valid = unknown_disps_valid_[cid_ref].row(r).count();
		if (num_valid == 0) {
			int idx_full = unknown_maps_bwd_[cid_ref](r, 0);
			masks_[cid_ref](idx_full, 0) = false;
			masks_int_[cid_ref](idx_full, 0) = 0;
			p = PixIndexBwdCM(idx_full, heights_[cid_ref]);
			imgMasks_[cid_ref].at<uchar>(p.y, p.x) = 0;
			mask_unknowns(idx_full, 0) = false;
			num_invalid++;
			*pV = false;
		}
		pV++;
	}

	if ((timing) &&
		(debug_tmp)) {
		t = (double)getTickCount() - t;
		cout << "StereoData::BuildValidDisparityRange() mask-outs running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}

	if (num_invalid > 0) { // rebuild unknown_disps_valid_[cid_ref], pixel knowledge, and dilated masks for the camera
		SpecifyPixelData(cid_ref, &mask_unknowns); // use this function instead of InitPixelData() in case are arriving here from StereoReconstruction::SmoothDisparityData() once called by StereoReconstruction::SyncDisparityData()
		DilateMask(cid_ref);
		Matrix<bool, Dynamic, Dynamic> udv_new(num_unknown_pixels_[cid_ref], nums_disps_[cid_ref]); // num_unknown_pixels_[cid_ref] has been recomputed in the InitPixelData(cid_ref) call above
		pV = valid.data();
		int idx_unk_new = 0;
		for (int r = 0; r < unknown_disps_valid_[cid_ref].rows(); r++) {
			if (*pV++) {
				udv_new.row(idx_unk_new) = unknown_disps_valid_[cid_ref].row(r);
				idx_unk_new++;
			}
		}
		unknown_disps_valid_[cid_ref] = udv_new;
	}

	if (timing) {
		t = (double)getTickCount() - t_init;
		cout << "StereoData::BuildValidDisparityRange() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}

	if (debug_static) {
		//cout << "cid_restrict " << cid_restrict << endl; // uncomment if want to debug restriction method
		Matrix<bool, Dynamic, 1> inbounds_all(num_unknown_pixels_[cid_ref], 1);
		//inbounds_all.setConstant(true);
		//cout << "Displaying all unknown pixels" << endl;
		//DisplayImages::DisplayGrayscaleImageTruncated(&inbounds_all, &masks_unknowns_[cid_ref], heights_[cid_ref], widths_[cid_ref], orientations_[cid_ref]);
		inbounds_all.setConstant(false);
		for (int r = 0; r < inbounds_all.rows(); r++) {
			inbounds_all(r, 0) = unknown_disps_valid_[cid_ref].row(r).any();
		}
		cout << "Displaying unknown pixels with at least one valid disparity" << endl;
		DisplayImages::DisplayGrayscaleImageTruncated(&inbounds_all, &masks_unknowns_[cid_ref], heights_[cid_ref], widths_[cid_ref], orientations_[cid_ref]);
	}
}

// for camera cid, given an unknown pixel's index (in unknown pixel indexing), generate a random floating point disparity value (snapped to quantized disparity label positions) that is valid with respect to image masks for all other cameras in the scene
// if no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
// requires that unknown_disps_valid_ is built first
float StereoData::GenerateRandomValidDisparity(int cid, int unk_pix_idx) {
	bool debug = false;

	int num = unknown_disps_valid_[cid].row(unk_pix_idx).count();
	if (debug) cout << "StereoData::GenerateRandomValidDisparity() num " << num << endl;
	if (num == 0) return min_disps_[cid] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / (max_disps_[cid] - min_disps_[cid])); // generates a random float in the range[sd_->min_disp_, sd_->max_disp_]; this approach may assign disparity values that do not conform to the masks of one or more other input images; // if no valid disparity exists (among quantized label positions), the function returns a rnadom disparity between the disparity min and max since this may occur when camera projection matrices have errors and we don't know enough to correct it
	int valid_disp_label = (rand() % (int)num); // index into the contracted space of valid disparity labels
	if (debug) cout << "StereoData::GenerateRandomValidDisparity() valid_disp_label " << valid_disp_label << endl;
	Matrix<bool, 1, Dynamic> udv = unknown_disps_valid_[cid].row(unk_pix_idx);
	bool *p = udv.data();
	int disp_label; // index into all disparity labels
	int i = 0; // current position within contracted space of valid disparity labels
	for (disp_label = 0; disp_label < unknown_disps_valid_[cid].cols(); disp_label++) {
		if (*p++) {
			if (i == valid_disp_label) break;
			i++;
		}
	}

	if (debug) cout << "StereoData::GenerateRandomValidDisparity() disp_label " << disp_label << endl;
	float disp_val = DispLabelToValue(cid, disp_label);
	if (debug) cout << "StereoData::GenerateRandomValidDisparity() min_disp_ " << min_disps_[cid] << endl;
	if (debug) cout << "StereoData::GenerateRandomValidDisparity() max_disp_ " << max_disps_[cid] << endl;
	if (debug) cout << "StereoData::GenerateRandomValidDisparity() disp_val " << disp_val << endl;
	if (debug) cin.ignore();
	return disp_val;
}

// for camera cid, given an unknown pixel's index (in unknown pixel indexing), generate a floating point disparity value (snapped to quantized disparity label positions) that is valid with respect to image masks for all other cameras in the scene and is perc decimal percentage of the way along all the valid disparities
// if no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
// requires that unknown_disps_valid_ is built first
// 100% generates max disparity / min depth
float StereoData::GenerateValidDisparityAtPerc(int cid, int unk_pix_idx, float perc) {
	bool debug = false;

	int num = unknown_disps_valid_[cid].row(unk_pix_idx).count();
	int valid_disp_label = floor(perc * static_cast<float>(num));
	if (valid_disp_label < 0) valid_disp_label = 0;
	if (valid_disp_label >(num - 1)) valid_disp_label = num - 1;
	if (debug) cout << "StereoData::GenerateValidDisparityAtPerc() num " << num << endl;
	if (debug) cout << "StereoData::GenerateValidDisparityAtPerc() valid_disp_label " << valid_disp_label << endl;
	Matrix<bool, 1, Dynamic> udv = unknown_disps_valid_[cid].row(unk_pix_idx);
	bool *p = udv.data();
	int disp_label; // index into all disparity labels
	int i = 0; // current position within contracted space of valid disparity labels
	for (disp_label = 0; disp_label < unknown_disps_valid_[cid].cols(); disp_label++) {
		if (*p++) {
			if (i == valid_disp_label) break;
			i++;
		}
	}

	if (debug) cout << "StereoData::GenerateValidDisparityAtPerc() disp_label " << disp_label << endl;
	float disp_val = DispLabelToValue(cid, disp_label);
	if (debug) cout << "StereoData::GenerateValidDisparityAtPerc() min_disp_ " << min_disps_[cid] << endl;
	if (debug) cout << "StereoData::GenerateValidDisparityAtPerc() max_disp_ " << max_disps_[cid] << endl;
	if (debug) cout << "StereoData::GenerateValidDisparityAtPerc() disp_val " << disp_val << endl;
	if (debug) cin.ignore();
	return disp_val;
}

void StereoData::InverseProjectSStoWS(int cid, Matrix<double, 3, Dynamic> *Iss, Matrix<double, 1, Dynamic> *depths, Matrix<double, 4, Dynamic> *Iws) {
	bool debug = false;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	// scale u,v,w by the desired depth amount to get homogeneous coordinates that reflect the depth after transformation
	// use depth values from depth_map
	Iss->row(0) = Iss->row(0).cwiseProduct(*depths);
	Iss->row(1) = Iss->row(1).cwiseProduct(*depths);
	Iss->row(2) = Iss->row(2).cwiseProduct(*depths);

	// transform screen space to camera space - transform u,v to x,y, then add rows for z (equal to each depth value) and w (equal to 1.0)
	Matrix<double, 2, 3> Kinv_uvonly;
	Kinv_uvonly.row(0) = Kinvs_[cid].row(0).cast<double>();
	Kinv_uvonly.row(1) = Kinvs_[cid].row(1).cast<double>();
	Matrix<double, 2, Dynamic> Ics_xyonly = Kinv_uvonly * (*Iss); // Ics is homogeneous 4xn matrix of camera space points
	Iws->setOnes(); // Iws at this point is still in camera space until we multiply it below by inverse extrinsics
	Iws->row(0) = Ics_xyonly.row(0);
	Iws->row(1) = Ics_xyonly.row(1);
	Ics_xyonly.resize(2, 0);

	// in camera space, set z to depth value and w to 1 (already scaled x,y in homogeneous screen space)
	// use depth values from dm
	Iws->row(2) = (*depths);

	// transform camera space positions to world space
	(*Iws) = RTinvs_[cid].cast<double>() * (*Iws); // Iws is homogeneous 4xn matrix of world space points; RTinv includes transformation from Agisoft space to world space
	Matrix<double, 1, Dynamic> H = Iws->row(3).array().inverse();

	// normalize by homogeneous value
	Iws->row(0) = Iws->row(0).cwiseProduct(H);
	Iws->row(1) = Iws->row(1).cwiseProduct(H);
	Iws->row(2) = Iws->row(2).cwiseProduct(H);
	Iws->row(3).setOnes();

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "StereoData::InverseProjectSStoWS() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// inverse projects screen space points (screen space dimensions ss_width x ss_height) with depths given by imgD from screen space to world space using Kinv and RTinv, updating a 4xn matrix of type float of the corresponding points in world space
// imgD is a 2D depth image matrix of size ss_height x ss_width (n points) whose depth values are in units that match Kinv
// Kinv is a 3x3 inverse calibration matrix of type CV_64F
// RTinv is a 4x4 inverse RT matrix of type CV_64F
// Iws must be a 4x(ss_width*ss_height) matrix
// updates Iws with homogeneous world space points as (x,y,z,w)
// expects everything in column-major
void StereoData::InverseProjectSStoWS(int ss_width, int ss_height, Matrix<float, Dynamic, Dynamic> *depth_map, Matrix3d *Kinv, Matrix4d *RTinv, Matrix<float, 4, Dynamic> *Iws) {
	bool debug = false;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	assert(Iws->cols() == ss_width*ss_height);
	assert(depth_map->rows() == ss_height && depth_map->cols() == ss_width);

	Matrix<float, Dynamic, Dynamic> dm(ss_height, ss_width); // Iws and related matrices are row-major interpretations of their 2D counterparts
	dm = (*depth_map);
	dm.resize(1, ss_width * ss_height);

	// scale u,v,w by the desired depth amount to get homogeneous coordinates that reflect the depth after transformation
	Matrix<float, 3, Dynamic> I = ConstructSSCoordsCM(ss_width, ss_height);
	// use depth values from depth_map
	I.row(0) = I.row(0).cwiseProduct(dm);
	I.row(1) = I.row(1).cwiseProduct(dm);
	I.row(2) = I.row(2).cwiseProduct(dm);

	// transform screen space to camera space - transform u,v to x,y, then add rows for z (equal to each depth value) and w (equal to 1.0)
	Matrix<float, 2, 3> Kinv_uvonly;
	Kinv_uvonly.row(0) = Kinv->row(0).cast<float>();
	Kinv_uvonly.row(1) = Kinv->row(1).cast<float>();
	Matrix<float, 2, Dynamic> Ics_xyonly = Kinv_uvonly * I; // Ics is homogeneous 4xn matrix of camera space points
	I.resize(3, 0);
	Iws->setOnes(); // Iws at this point is still in camera space until we multiply it below by inverse extrinsics
	Iws->row(0) = Ics_xyonly.row(0);
	Iws->row(1) = Ics_xyonly.row(1);
	Ics_xyonly.resize(2, 0);

	// in camera space, set z to depth value and w to 1 (already scaled x,y in homogeneous screen space)
	// use depth values from dm
	Iws->row(2) = dm;
	dm.resize(0, 0);

	// transform camera space positions to world space
	(*Iws) = (*RTinv).cast<float>() * (*Iws); // Iws is homogeneous 4xn matrix of world space points; RTinv includes transformation from Agisoft space to world space
	Matrix<float, 1, Dynamic> H = Iws->row(3).array().inverse();

	// normalize by homogeneous value
	Iws->row(0) = Iws->row(0).cwiseProduct(H);
	Iws->row(1) = Iws->row(1).cwiseProduct(H);
	Iws->row(2) = Iws->row(2).cwiseProduct(H);
	Iws->row(3).setOnes();

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "StereoData::InverseProjectSStoWS() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// returns 3 by (ss_w*ss_h) data structure with all homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h assuming row-major order of indices
Matrix<float, 3, Dynamic> StereoData::ConstructSSCoordsCM(int ss_w, int ss_h) {
	bool debug = false;

	Matrix<float, 3, Dynamic> I(3, ss_w*ss_h); // 3xn matrix of homogeneous screen space points where n is the number of pixels in imgT_
	I.row(2).setConstant(1.);

	int idx = 0;
	for (int c = 0; c < ss_w; c++) {
		for (int r = 0; r < ss_h; r++) {
			//idx = PixIndexFwdCM(Point(c, r), ss_h);
			I(0, idx) = static_cast<float>(c);
			I(1, idx) = static_cast<float>(r);
			idx++;
		}
	}

	return I;
}

// pixel positions are used to generate a regular lattice mesh for each camera, but the positions are at integer locations, resulting in sawtooth edges. The segment edges are smoothed to alleviate this effect. "Edge" pixels are identified as masked-in pixels that border at least one pixel of a different segment.  "Corner" pixels are identified as pixels with 2 neighbors in a 4-connected space that are part of the same segment, are "edge" pixels themselves, and are adjacent to each other (neighbors in an 8-connected sense).  "Corner" pixels are smoothed by setting their position to be the average of their current position and the average of the positions of their 2 identified edge neighbors.
// returns 3 by (ss_w*ss_h) data structure with all homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h assuming row-major order of indices
Matrix<double, 3, Dynamic> StereoData::ConstructSmoothedMaskedInSSCoordsCM(int cid, int ss_w, int ss_h) {
	bool debug = false;
	
	Matrix<float, 3, Dynamic> I = ConstructSSCoordsCM(ss_w, ss_h);

	// label each position as either on a segment edge or not
	Matrix<bool, Dynamic, Dynamic> edges(ss_h, ss_w);
	edges.setConstant(false);
	
	unsigned int curr_label, other_label;
	bool edge_pt;
	int idx, num_ens;
	for (int c = 1; c < (ss_w - 1); c++) {
		for (int r = 1; r < (ss_h - 1); r++) {
			curr_label = segs_[cid](r, c);
			if (curr_label == 0) continue;
			idx = PixIndexFwdCM(Point(c, r), ss_h);
			if (!masks_[cid](idx, 0)) continue;
			if (depth_maps_[cid](r, c) == 0.) continue;

			edge_pt = false;
			for (int j = -1; j <= 1; j++) {
				for (int i = -1; i <= 1; i++) {
					if ((i == 0) && (j == 0)) continue;
					if (segs_[cid](r + j, c + i) != curr_label)
						edge_pt = true;
				}
			}

			if (edge_pt) edges(r, c) = true;
		}
	}
	if (debug) DisplayImages::DisplayGrayscaleImage(&edges, ss_h, ss_w, orientations_[cid]);

	Matrix<bool, Dynamic, Dynamic> corners(ss_h, ss_w);
	corners.setConstant(false);
	for (int c = 1; c < (ss_w - 1); c++) {
		for (int r = 1; r < (ss_h - 1); r++) {
			if (!edges(r, c)) continue;
			idx = PixIndexFwdCM(Point(c, r), ss_h);

			num_ens = 0;
			if ((edges(r + 1, c)) &&
				(edges(r, c + 1))) num_ens++;
			if ((edges(r - 1, c)) &&
				(edges(r, c + 1))) num_ens++;
			if ((edges(r + 1, c)) &&
				(edges(r, c - 1))) num_ens++;
			if ((edges(r - 1, c)) &&
				(edges(r, c - 1))) num_ens++;

			if (num_ens == 1) corners(r, c) = true; // if has 2 edge neighbors at corner
		}
	}
	if (debug) DisplayImages::DisplayGrayscaleImage(&corners, ss_h, ss_w, orientations_[cid]);

	//Matrix<bool, Dynamic, Dynamic> changes(ss_h, ss_w);
	//changes.setConstant(false);

	// smooth edges
	Matrix<double, 3, Dynamic> Ismoothed = I.cast<double>();
	int idx_other;
	double curr_x, curr_y, new_x, new_y, num_crowd, crowd_x, crowd_y;
	for (int iter = 0; iter < GLOBAL_MESH_EDGE_SMOOTH_ITERS; iter++) {
		for (int c = 1; c < (ss_w - 1); c++) {
			for (int r = 1; r < (ss_h - 1); r++) {
				//if (!corners(r, c)) continue;
				if (!edges(r, c)) continue;

				idx = PixIndexFwdCM(Point(c, r), ss_h);
				curr_x = static_cast<double>(I(0, idx));
				curr_y = static_cast<double>(I(1, idx));
				curr_label = segs_[cid](r, c);
				if (curr_label == 0) continue; // actually shouldn't be needed because only non-zero labels should have edges(r,c)==true
				
				// each corner should be moved toward the average of its canonical (4-connected) neighbors of the same segment (mesh is built on 4-connected neighbor regular lattice) without crossing into neighboring pixel territory
				num_crowd = 0.;
				crowd_x = 0.;
				crowd_y = 0.;

				if ((edges(r + 1, c)) &&
					(segs_[cid](r + 1, c) == curr_label)) {
					idx_other = PixIndexFwdCM(Point(c, r + 1), heights_[cid]);
					crowd_x += I(0, idx_other);
					crowd_y += I(1, idx_other);
					num_crowd++;
				}
				if ((edges(r - 1, c)) &&
					(segs_[cid](r - 1, c) == curr_label)) {
					idx_other = PixIndexFwdCM(Point(c, r - 1), heights_[cid]);
					crowd_x += I(0, idx_other);
					crowd_y += I(1, idx_other);
					num_crowd++;
				}
				if ((edges(r, c + 1)) &&
					(segs_[cid](r, c + 1) == curr_label)) {
					idx_other = PixIndexFwdCM(Point(c + 1, r), heights_[cid]);
					crowd_x += I(0, idx_other);
					crowd_y += I(1, idx_other);
					num_crowd++;
				}
				if ((edges(r, c - 1)) &&
					(segs_[cid](r, c - 1) == curr_label)) {
					idx_other = PixIndexFwdCM(Point(c - 1, r), heights_[cid]);
					crowd_x += I(0, idx_other);
					crowd_y += I(1, idx_other);
					num_crowd++;
				}
				if (num_crowd != 2) continue;
				
				crowd_x /= num_crowd;
				crowd_y /= num_crowd;
				
				new_x = (GLOBAL_MESH_EDGE_SMOOTH_WEIGHT_CURR_POS * curr_x) + ((1 - GLOBAL_MESH_EDGE_SMOOTH_WEIGHT_CURR_POS) * crowd_x);
				new_y = (GLOBAL_MESH_EDGE_SMOOTH_WEIGHT_CURR_POS  *curr_y) + ((1 - GLOBAL_MESH_EDGE_SMOOTH_WEIGHT_CURR_POS) * crowd_y);
				
				new_x = max(new_x, static_cast<double>(c - 1) - GLOBAL_MESH_EDGE_SMOOTH_BUFFER_HALF);
				new_x = min(new_x, static_cast<double>(c + 1) - GLOBAL_MESH_EDGE_SMOOTH_BUFFER_HALF);
				new_y = max(new_y, static_cast<double>(r - 1) - GLOBAL_MESH_EDGE_SMOOTH_BUFFER_HALF);
				new_y = min(new_y, static_cast<double>(r + 1) - GLOBAL_MESH_EDGE_SMOOTH_BUFFER_HALF);

				if (new_x != curr_x) Ismoothed(0, idx) = new_x;
				if (new_y != curr_y) Ismoothed(1, idx) = new_y;

				//if (debug) {
				//	if ((new_x != curr_x) ||
				//		(new_y != curr_y)) {
				//		changes(r, c) = true;
				//	}
				//}
			}
		}
	}

	//if (debug) DisplayImages::DisplayGrayscaleImage(&changes, ss_h, ss_w, orientations_[cid]);

	return Ismoothed;
}

// builds data structures meshes_vertices_, meshes_faces_, and meshes_vertex_normals_
// set all vertex normals to be along the camera space ray through the corresponding screen space pixel, whereas set all face normals to be correct for the face; note that .obj file format does not require normals to be unit vectors, so we do not normalize them here
void StereoData::BuildMeshes(int cid_specific) {
	cout << "StereoData::BuildMeshes()" << endl;

	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		if ((cid_specific != -1) &&
			(cid != cid_specific)) continue;
		if (!valid_cam_poses_[cid]) continue;

		int h = heights_[cid];
		int w = widths_[cid];
		int num_points = h*w;
		Matrix<double, 4, Dynamic> Iws(4, h*w);
		
		Matrix<double, 3, Dynamic> Iss = ConstructSmoothedMaskedInSSCoordsCM(cid, w, h);
		//Matrix<float, 3, Dynamic> I = ConstructSSCoordsCM(w, h);;

		Matrix<double, Dynamic, Dynamic> dtmp = depth_maps_[cid].cast<double>();
		dtmp.resize(1, w*h);
		Matrix<double, 1, Dynamic> depths = dtmp;
		dtmp.resize(0, 0);	

		InverseProjectSStoWS(cid, &Iss, &depths, &Iws);
		//InverseProjectSStoWS(w, h, &depth_maps_[cid], &Kinv, &RTinv, &Iws);
		Matrix<float, Dynamic, 4> I = Iws.transpose().cast<float>();

		Matrix4d RTinv = RTinvs_[cid].cast<double>();
		Point3d cam_posd = Camera::GetCameraPositionWS(&RTinv);
		Point3f cam_pos;
		cam_pos.x = static_cast<float>(cam_posd.x);
		cam_pos.y = static_cast<float>(cam_posd.y);
		cam_pos.z = static_cast<float>(cam_posd.z);
		Point3f vnorm;

		meshes_vertices_[cid].erase(meshes_vertices_[cid].begin(), meshes_vertices_[cid].end());
		meshes_faces_[cid].erase(meshes_faces_[cid].begin(), meshes_faces_[cid].end());
		meshes_vertex_faces_[cid].erase(meshes_vertex_faces_[cid].begin(), meshes_vertex_faces_[cid].end());
		meshes_vertex_normals_[cid].erase(meshes_vertex_normals_[cid].begin(), meshes_vertex_normals_[cid].end());

		/*
		map<int, int> face_ccs; // map of face ID => connected component ID
		map<int, vector<int>> ccs; // map of connected component ID => vector of included face IDs
		int ccid;
		*/

		Matrix<float, Dynamic, Dynamic> depth_map_resized = depth_maps_[cid];
		depth_map_resized.resize(h*w, 1);

		// build mesh vertices
		float *pVd = depth_map_resized.data();
		bool *pVm = masks_[cid].data();
		float *pVx = I.col(0).data();
		float *pVy = I.col(1).data();
		float *pVz = I.col(2).data();
		unsigned int *pS = segs_[cid].data();
		// pre-fill first column of vertices
		for (int i = 0; i < num_points; i++) {
			// create vertices
			if ((*pVm) && // must be masked-in
				(*pS != 0) && // must not be member of segment label 0 (background segment)
				(*pVd != 0.)) { // must have non-zero camera space depth
				Point3f v(*pVx, *pVy, *pVz);
				meshes_vertices_[cid][i] = v;
				vector<int> empty_list_face_ids;
				meshes_vertex_faces_[cid][i] = empty_list_face_ids;
				vnorm = cam_pos - v;
				meshes_vertex_normals_[cid][i] = vnorm;
			}

			pVd++;
			pVm++;
			pVx++;
			pVy++;
			pVz++;
			pS++;
		}

		// build mesh faces
		float *pULd = depth_map_resized.data();
		float *pLLd = depth_map_resized.data() + 1;
		float *pURd = depth_map_resized.data() + h;
		float *pLRd = depth_map_resized.data() + h + 1;
		bool *pULm = masks_[cid].data();
		bool *pLLm = masks_[cid].data() + 1;
		bool *pURm = masks_[cid].data() + h;
		bool *pLRm = masks_[cid].data() + h + 1;
		unsigned int *pULs = segs_[cid].data();
		unsigned int *pLLs = segs_[cid].data() + 1;
		unsigned int *pURs = segs_[cid].data() + h;
		unsigned int *pLRs = segs_[cid].data() + h + 1;
		int idx_face = 0;
		for (int i = 0; i < (num_points - h - 1); i++) { // we are looking ahead in the data one column and row, so must stop one column and row short

			// create faces and normals; don't build faces where vertices are too far apart
			if ((*pULm) && (*pULd != 0.) && (*pULs != 0) &&
				(*pLRm) && (*pLRd != 0.) && (*pLRs != 0) &&
				(*pLLm) && (*pLLd != 0.) && (*pLLs != 0)) {

				Point3f e1 = meshes_vertices_[cid][i + 1] - meshes_vertices_[cid][i];
				Point3f e2 = meshes_vertices_[cid][i + h + 1] - meshes_vertices_[cid][i + 1];
				Point3f e3 = meshes_vertices_[cid][i] - meshes_vertices_[cid][i + h + 1];
				float d1 = vecdist(e1, e2);
				float d2 = vecdist(e1, e3);
				float d3 = vecdist(e2, e3);

				if ((d1 < GLOBAL_MESH_EDGE_DISTANCE_MAX) &&
					(d2 < GLOBAL_MESH_EDGE_DISTANCE_MAX) &&
					(d3 < GLOBAL_MESH_EDGE_DISTANCE_MAX)) {

					Vec3i f(i, i + 1, i + h + 1); // counter-clockwise vertex order
					meshes_faces_[cid][idx_face] = f;

					meshes_vertex_faces_[cid][i + h + 1].push_back(idx_face);
					meshes_vertex_faces_[cid][i + 1].push_back(idx_face);
					meshes_vertex_faces_[cid][i].push_back(idx_face);

					idx_face++;
				}
			}
			if ((*pULm) && (*pULd != 0.) && (*pULs != 0) &&
				(*pLRm) && (*pLRd != 0.) && (*pLRs != 0) &&
				(*pURm) && (*pURd != 0.) && (*pURs != 0)) {
				Point3f e1 = meshes_vertices_[cid][i + h + 1] - meshes_vertices_[cid][i];
				Point3f e2 = meshes_vertices_[cid][i + h] - meshes_vertices_[cid][i + h + 1];
				Point3f e3 = meshes_vertices_[cid][i] - meshes_vertices_[cid][i + h];
				float d1 = vecdist(e1, e2);
				float d2 = vecdist(e1, e3);
				float d3 = vecdist(e2, e3);

				if ((d1 < GLOBAL_MESH_EDGE_DISTANCE_MAX) &&
					(d2 < GLOBAL_MESH_EDGE_DISTANCE_MAX) &&
					(d3 < GLOBAL_MESH_EDGE_DISTANCE_MAX)) {

					Vec3i f(i, i + h + 1, i + h); // counter-clockwise vertex order
					meshes_faces_[cid][idx_face] = f;

					meshes_vertex_faces_[cid][i + h + 1].push_back(idx_face);
					meshes_vertex_faces_[cid][i + h].push_back(idx_face);
					meshes_vertex_faces_[cid][i].push_back(idx_face);

					idx_face++;
				}
			}

			pULd++;
			pLLd++;
			pURd++;
			pLRd++;
			pULm++;
			pLLm++;
			pURm++;
			pLRm++;
			pULs++;
			pLLs++;
			pURs++;
			pLRs++;
		}

		CleanFacesAgainstMasks(cid);
		RemoveIsolatedFacesFromMeshes(cid);
	}
}

// delete isolated connected components of meshes that have fewer than GLOBAL_MIN_CONNECTED_COMPONENT_FACES faces
// note: would like to make deletion apply as follows: if a segment has more than one group of connected faces, keep the largest and throw out any others since a segment should be fully connected; however, cannot since a single connected component may span multiple segments in order to have a cohesive model (since segmentation lines are often drawn inappropriately along edges that do not separate segments at disjoint disparities)
void StereoData::RemoveIsolatedFacesFromMeshes(int cid_specific) {
	bool debug = false;

	 cout << "StereoData::RemoveIsolatedFacesFromMeshes()" << endl;
	for (std::map<int, Mat>::iterator iti = imgsT_.begin(); iti != imgsT_.end(); ++iti) {
		int cid = (*iti).first;
		if (!valid_cam_poses_[cid]) continue;
		if ((cid_specific != -1) &&
			(cid != cid_specific))
			continue;

		// build connected components
		if (debug) cout << "build connected components" << endl;
		map<int, int> face_ccs; // map of face ID => connected component ID
		map<int, vector<int>> ccs; // map of connected component ID => vector of included face IDs
		int ccid = 0;
		for (map<int, Vec3i>::iterator it = meshes_faces_[cid].begin(); it != meshes_faces_[cid].end(); ++it) {
			int fid = (*it).first;

			// skip faces that have already been mapped to connected components
			if (face_ccs.find(fid) != face_ccs.end())
				continue;

			// create a new connected component for it
			face_ccs[fid] = ccid;
			ccs[ccid].push_back(fid);

			// now, travel along its connected faces and assign them to its connected component; for each vertex, find the other faces to which it belongs, assign those faces to this component, and recursively do the same for the vertices of those faces (implemented here as an iterative, rather than recursive, procedure to avoid overflowing the call stack with recursive calls
			vector<int> verts;
			for (int i = 0; i < 3; i++)
				verts.push_back(meshes_faces_[cid][fid][i]);
			while (verts.size() != 0) {
				int nvid = verts.back();
				verts.pop_back();
				for (vector<int>::iterator itvf = meshes_vertex_faces_[cid][nvid].begin(); itvf != meshes_vertex_faces_[cid][nvid].end(); ++itvf) {
					int nfid = (*itvf);
					if (nfid == fid) continue; // we're back at the original face (the next check would have also caught this, but it's faster to get these out of the way this way)
					if (face_ccs.find(nfid) != face_ccs.end()) { // we're at a face that's already mapped
						assert(face_ccs[nfid] == face_ccs[fid]);
						continue;
					}
					face_ccs[nfid] = ccid;
					ccs[ccid].push_back(nfid);
					for (int i = 0; i < 3; i++)
						verts.push_back(meshes_faces_[cid][nfid][i]);
				}
			}

			ccid++;
		}

		if (debug) {
			cout << "double-check that all faces are accounted for" << endl;
			bool passed = true;
			for (map<int, Vec3i>::iterator it = meshes_faces_[cid].begin(); it != meshes_faces_[cid].end(); ++it) {
				int fid = (*it).first;
				if (face_ccs.find(fid) == face_ccs.end()) {
					cout << "found a face that is missing" << endl;
					passed = false;
				}
			}
			if (passed)
				cout << "check passed" << endl << endl;
			else
				cout << "check failed" << endl << endl;
			cin.ignore();
		}

		if (debug) {
			cout << "number of connected components: " << ccs.size() << endl;
			cout << endl << "size of each connected component:" << endl;
			for (map<int, vector<int>>::iterator it = ccs.begin(); it != ccs.end(); ++it) {
				int ccid = (*it).first;
				int num = (*it).second.size();
				cout << "connected component " << ccid << " includes number of faces = " << num << endl;
			}
			cin.ignore();
		}

		/*
		// determine connected components that belong to each segment label
		if (debug) cout << "determine connected components that belong to each segment label" << endl;
		map<unsigned int, vector<int>> seg_ccs; // map of segment label => connected component ID
		for (map<int, vector<int>>::iterator it = ccs.begin(); it != ccs.end(); ++it) {
		int ccid = (*it).first;
		int fid = (*(*it).second.begin());
		int vid = meshes_faces_[cid][fid][0];
		Point3f wspt = meshes_vertices_[cid][vid];
		Matrix<float, 4, 1> WC;
		WC(0, 0) = wspt.x;
		WC(1, 0) = wspt.y;
		WC(2, 0) = wspt.z;
		WC(3, 0) = 1.;
		Matrix<float, 3, 1> T = Ps_[cid] * WC;
		float h = T(2, 0);
		int x = round(T(0, 0) / h);
		int y = round(T(1, 0) / h);
		if (x < 0) x = 0;
		if (y < 0) y = 0;
		if (x >= widths_[cid]) x = widths_[cid] - 1;
		if (y >= heights_[cid]) y = heights_[cid] - 1;
		unsigned int label = segs_[cid](y, x);
		seg_ccs[label].push_back(ccid);
		}

		if (debug) {
		for (map<unsigned int, vector<int>>::iterator it = seg_ccs.begin(); it != seg_ccs.end(); ++it) {
		unsigned int seg = (*it).first;
		cout << endl << "segment label " << seg << " includes the following connected components: " << endl;
		for (vector<int>::iterator it2 = (*it).second.begin(); it2 != (*it).second.end(); ++it2) {
		int sccid = (*it2);
		cout << "connected component " << sccid << endl;
		}
		}
		cin.ignore();
		}
		*/

		// determine connected components to delete
		if (debug) cout << "determine connected components to delete" << endl;
		vector<int> del_ccs;
		for (map<int, vector<int>>::iterator it = ccs.begin(); it != ccs.end(); ++it) {
			int ccid = (*it).first;
			int num = (*it).second.size();
			if (num < GLOBAL_MIN_CONNECTED_COMPONENT_FACES) del_ccs.push_back(ccid);
		}

		if (debug) {
			cout << endl << "connected components to delete:" << endl;
			for (vector<int>::iterator it = del_ccs.begin(); it != del_ccs.end(); ++it) {
				int ccid = (*it);
				cout << "connected component " << ccid << endl;
			}
			cin.ignore();
		}
			
		// delete appropriate connected components
		if (debug) cout << "delete appropriate connected components" << endl;
		for (vector<int>::iterator it = del_ccs.begin(); it != del_ccs.end(); ++it) {
			int ccid = (*it);

			// erase each face
			for (vector<int>::iterator itf = ccs[ccid].begin(); itf != ccs[ccid].end(); ++itf) {
				int fid = (*itf);

				for (int i = 0; i < 3; i++) {
					int vid = meshes_faces_[cid][fid][i];
					meshes_vertex_faces_[cid][vid].erase(find(meshes_vertex_faces_[cid][vid].begin(), meshes_vertex_faces_[cid][vid].end(), fid));
						
					// if the vertex no longer belongs to any face, erase the vertex as well
					if (meshes_vertex_faces_[cid][vid].size() == 0) {
						meshes_vertices_[cid].erase(vid);
						meshes_vertex_normals_[cid].erase(vid);
						meshes_texcoords_[cid].erase(vid);
					}
				}

				meshes_faces_[cid].erase(fid);

				if (debug) face_ccs.erase(fid);
			}

			if (debug) ccs.erase(ccid);
		}
			
		if (debug) {
			cout << endl << "double-check of connected components after deletion" << endl;
			cout << "number of connected components: " << ccs.size() << endl;
			cout << endl << "size of each connected component:" << endl;
			for (map<int, vector<int>>::iterator it = ccs.begin(); it != ccs.end(); ++it) {
				int ccid = (*it).first;
				int num = (*it).second.size();
				cout << "connected component " << ccid << " includes number of faces = " << num << endl;
			}
			cin.ignore();
			cout << "double-check a different way by going through faces for their connected component IDs to ensure the deleted components are not represented and all fall within known components" << endl;
			for (map<int, int>::iterator it = face_ccs.begin(); it != face_ccs.end(); ++it) {
				int tfid = (*it).first;
				int tccid = (*it).second;
				if (ccs.find(tccid) == ccs.end()) {
					cout << "ccid " << tccid << " for face " << tfid << " not present in list of known components" << endl;
					cin.ignore();
				}
				if (std::find(del_ccs.begin(), del_ccs.end(), tccid) != del_ccs.end()) {
					cout << "ccid " << tccid << " for face " << tfid << " IS present in list of components that were deleted" << endl;
					cin.ignore();
				}
			}
		}
	}
}

// try to ensure faces do not project to pixels in any image that are masked-out
void StereoData::CleanFacesAgainstMasks(int cid_mesh) {
	bool debug = true;	

	if (debug) cout << "StereoData::CleanFacesAgainstMasks() for cid_mesh " << cid_mesh << endl;

	// build world space points and map between them and faces
	Matrix<float, 4, Dynamic> Iws(4, 3 * meshes_faces_[cid_mesh].size());
	Iws.row(3).setOnes();
	map<int, Vec3i> fid_map; // map of face ID => Iws indices for each of the three vertices of the face
	int iws_idx = 0;
	for (map<int, Vec3i>::iterator it = meshes_faces_[cid_mesh].begin(); it != meshes_faces_[cid_mesh].end(); ++it) {
		int fid = (*it).first;
		Vec3i vs = (*it).second;
		Vec3i iws_indices;
		for (int i = 0; i < 3; i++) {
			int vid = vs[i];
			Point3f v = meshes_vertices_[cid_mesh][vid];
			Iws(0, iws_idx) = v.x;
			Iws(1, iws_idx) = v.y;
			Iws(2, iws_idx) = v.z;
			iws_indices[i] = iws_idx;
			iws_idx++;
		}
		fid_map[fid] = iws_indices;
	}
	
	// validate mesh against masks
	for (map<int, Matrix<bool, Dynamic, 1>>::iterator it = masks_.begin(); it != masks_.end(); ++it) {
		int cid = (*it).first;

		// reproject world space coordinates to other camera's screen space
		Matrix<float, 3, Dynamic> I_dest = Ps_[cid] * Iws; // note the matrix multiplication property: Ainv * A = A * Ainv
		Matrix<float, 1, Dynamic> H = I_dest.row(2).array().inverse();
		I_dest.row(0) = I_dest.row(0).cwiseProduct(H);
		I_dest.row(1) = I_dest.row(1).cwiseProduct(H);
		
		// validate each face
		for (map<int, Vec3i>::iterator itf = fid_map.begin(); itf != fid_map.end(); ++itf) {
			int fid = (*itf).first;
			Vec3i iws_indices = (*itf).second;
			map<int, Point> vs;
			for (int i = 0; i < 2; i++) {
				Point v;
				v.x = round(Iws(0, iws_indices[i]));
				v.y = round(Iws(1, iws_indices[i]));
				// crop vertices to image bounds; can't reject faces with vertices outside the image since an image could be a close-up cropped shot, so instead validate as best we can for in-image portion of the projection against the inverse mask
				v.x = max(v.x, 0); v.x = min(v.x, widths_[cid]);
				v.y = max(v.y, 0); v.y = min(v.y, heights_[cid]);

				vs[i] = v;
			}

			// find labels of 3 vertices in cid - if labels are the same, then face passes automatically
			unsigned int l1, l2, l3;
			l1 = segs_[cid](vs[0].y, vs[0].x);
			l2 = segs_[cid](vs[1].y, vs[1].x);
			l3 = segs_[cid](vs[2].y, vs[2].x);
			if ((l1 == l2) &&
				(l2 == l3))
				continue;

			// create line iterator for each of 3 edges of face and grab pixels from mask - if any pixel is less than GLOBAL_MIN_MASKSEG_LINEVAL, the face (and its corresponding normal) should be removed from the list for cid_mesh
			// grabs pixels along the line (pt1, pt2)
			// from 8-bit 3-channel image to the buffer
			bool fails = false;
			int vi_last = 2;
			int vi = 0;
			while ((!fails) &&
					(vi < 2)) {
				LineIterator itl(imgMasks_[cid], vs[vi_last], vs[vi], 8);
				for (int i = 0; i < itl.count; i++, ++itl) {
					int val = imgMasks_[cid].at<uchar>(itl.pos());
					if (val < GLOBAL_MIN_MASKSEG_LINEVAL) { // face fails
						fails = true;
						break;
					}
				}

				vi_last = vi;
				vi++;
			}
			if (fails) {
				meshes_faces_[cid_mesh].erase(fid);
			}
		}
	}
}

// don't trust depth data on pixels with segmentation value 0 (on the lines) unless all pixels in neighborhood are within reasonable depth distance of each other (ensuring it represents a shared edge and not an occlusion edge), so mask out untrusted pixels
void StereoData::MaskOutSegmentationOcclusionEdges(int cid) {
	// mask out 0-labeled edges because pushing them to one neighboring label or another often results in mis-labeled pixels that create deformities in the resulting mesh
	int idx;
	for (int c = 0; c < segs_[cid].cols(); c++) {
		for (int r = 0; r < segs_[cid].rows(); r++) {
			if (segs_[cid](r, c) != 0) continue;
			idx = PixIndexFwdCM(Point(c, r), heights_[cid]);

			if (!CheckCreateFaceOcclusion(cid, Point(c, r))) {
				depth_maps_[cid](r, c) = 0.;
				disparity_maps_[cid](idx, 0) = 0.;
				masks_[cid](idx, 0) = false;
				masks_int_[cid](idx, 0) = 0;
				imgMasks_[cid].at<uchar>(r, c) = 0;
			}
		}
	}
}

// don't build faces on pixels unless all pixels in neighborhood are within reasonable depth distance of each other (ensuring it represents a shared edge and not an occlusion edge, since this function is intended to be called on edge pixels in MaskOutSegmentationOcclusionEdges); returns true if it passes this test, false otherwise
bool StereoData::CheckCreateFaceOcclusion(int cid, Point p) {
	int w = widths_[cid];
	int h = heights_[cid];
	float mindepth = depth_maps_[cid](p.y, p.x);
	float maxdepth = depth_maps_[cid](p.y, p.x);
	float depth;
	int n_half = GLOBAL_SMOOTH_KERNEL_SIZE; // test against double the smoothing kernel size (actually double + 1)
	for (int c = p.x - n_half; c < p.x + n_half; c++) {
		for (int r = p.y - n_half; r < p.y + n_half; r++) {
			if ((c<0) ||
				(r>0) ||
				(c >= w) ||
				(r >= h))
				continue;
			depth = depth_maps_[cid](r, c);
			if (depth < mindepth) mindepth = depth;
			if (depth > maxdepth) maxdepth = depth;
		}
	}
	bool create_face = true;
	if (abs(maxdepth - mindepth) >= GLOBAL_MESH_EDGE_DISTANCE_MAX)
		create_face = false;

	return create_face;
}

// builds data structure meshes_texcoords_
void StereoData::BuildTextureCoordinates(int cid_specific) {
	cout << "StereoData::BuildTextureCoordinates()" << endl;

	for (map<int, map<int, Point3f>>::iterator itm = meshes_vertices_.begin(); itm != meshes_vertices_.end(); ++itm) {
		int cid = (*itm).first;
		if ((cid_specific != -1) &&
			(cid != cid_specific)) continue;
		if (!valid_cam_poses_[cid]) continue;

		// build matrix of points
		int num_vertices = (*itm).second.size();
		Matrix<float, 4, Dynamic> Iws(4, num_vertices);
		Iws.row(3).setConstant(1.);
		int i = 0;
		for (map<int, Point3f>::iterator itv = (*itm).second.begin(); itv != (*itm).second.end(); ++itv) {
			int vid = (*itv).first;
			Point3f pWS = (*itv).second;
			Iws(0, i) = pWS.x;
			Iws(1, i) = pWS.y;
			Iws(2, i) = pWS.z;
			i++;
		}

		// project points to screen space and normalize
		Matrix<float, 3, Dynamic> Iss = Ps_[cid] * Iws;
		Matrix<float, 1, Dynamic> H = Iss.row(2).array().inverse();
		Iss.row(0) = Iss.row(0).cwiseProduct(H);
		Iss.row(1) = Iss.row(1).cwiseProduct(H);

		// record texture coordinates for vertices
		meshes_texcoords_[cid].erase(meshes_texcoords_[cid].begin(), meshes_texcoords_[cid].end());
		for (int i = 0; i < num_vertices; i++) {
			Point2f vt;
			vt.x = Iss(0, i) / widths_[cid];
			vt.y = 1. - (Iss(1, i) / heights_[cid]); // y-axis is flipped
			meshes_texcoords_[cid][i] = vt;
		}
	}
}

void StereoData::ViewDepthMaps() {
	for (std::map<int, MatrixXf>::iterator it = depth_maps_.begin(); it != depth_maps_.end(); ++it) {
		int cid = (*it).first;
		DisplayImages::DisplayGrayscaleImage(&(*it).second, (*it).second.rows(), (*it).second.cols(), orientations_[cid]);
	}
}

void StereoData::SavePointCloud(string scene_name, vector<int> exclude_cids) {
	bool debug = false;

	cout << "StereoData::SavePointCloud()" << endl;

	std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\pointcloud.obj";
	ofstream myfile;
	myfile.open(fn);

	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		vector<int>::iterator itex = find(exclude_cids.begin(), exclude_cids.end(), cid);
		if (itex != exclude_cids.end()) continue;

		Matrix<double, 4, Dynamic> Iws_used(4, num_used_pixels_[cid]);
		Matrix<double, 3, Dynamic> Iss_used(3, num_used_pixels_[cid]);
		Iss_used.row(0) = Xuseds_[cid];
		Iss_used.row(1) = Yuseds_[cid];
		Iss_used.row(2).setOnes();
		Matrix<double, Dynamic, Dynamic> depths = depth_maps_[cid].cast<double>();
		depths.resize(depths.rows()*depths.cols(), 1);
		Matrix<double, Dynamic, 1> depths_used_col = EigenMatlab::TruncateByBooleansRows(&depths, &masks_[cid]);
		Matrix<double, 1, Dynamic> depths_used_row = depths_used_col.transpose();
		InverseProjectSStoWS(cid, &Iss_used, &depths_used_row, &Iws_used);
		//Iws_used = WorldToAgisoft_ * Iws_used;

		if (debug) DebugPrintMatrix(&Xuseds_[cid], "Xuseds_[cid]");
		if (debug) DebugPrintMatrix(&Yuseds_[cid], "Yuseds_[cid]");
		if (debug) DebugPrintMatrix(&Iss_used, "Iss_used");
		if (debug) DebugPrintMatrix(&depths_used_row, "depths_used_row");
		if (debug) DebugPrintMatrix(&WorldToAgisoft_, "WorldToAgisoft_");
		if (debug) DebugPrintMatrix(&Iws_used, "Iws_used");
		
		for (int c = 0; c < Iws_used.cols(); c++) {
			if (depths_used_row(c) != 0)
				myfile << "v " << Iws_used(0, c) << " " << Iws_used(1, c) << " " << Iws_used(2, c) << endl;
		}
	}

	myfile.close();
}

void StereoData::WriteAgisoftSparsePointCloud(string scene_name, vector<int> exclude_cids) {
	bool debug = false;

	int num_points = 0;
	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		vector<int>::iterator itex = find(exclude_cids.begin(), exclude_cids.end(), cid);
		if (itex != exclude_cids.end()) continue;

		num_points += num_used_pixels_[cid];
	}

	std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\points0.ply";
	FILE* pFile = fopen(fn.c_str(), "wb"); // write binary mode

	/*
	ply
	format binary_little_endian 1.0
	element vertex 162486
	property float x
	property float y
	property float z
	property uchar red
	property uchar green
	property uchar blue
	property uint frame
	property uint flags
	end_header
	*/
	std::fwrite((void*)"ply\n", sizeof(char), 4, pFile);
	std::fwrite((void*)"format binary_little_endian 1.0\n", sizeof(char), 32, pFile);
	std::fwrite((void*)"element vertex ", sizeof(char), 15, pFile);
	string num_points_str = std::to_string(num_points);
	char *num_points_chars = convert_string_to_chars(num_points_str);
	std::fwrite((void*)num_points_chars, sizeof(char), num_points_str.length(), pFile);
	delete[] num_points_chars;
	std::fwrite((void*)"\n", sizeof(char), 1, pFile);
	std::fwrite((void*)"property float x\n", sizeof(char), 17, pFile);
	std::fwrite((void*)"property float y\n", sizeof(char), 17, pFile);
	std::fwrite((void*)"property float z\n", sizeof(char), 17, pFile);
	std::fwrite((void*)"property uchar red\n", sizeof(char), 19, pFile);
	std::fwrite((void*)"property uchar green\n", sizeof(char), 21, pFile);
	std::fwrite((void*)"property uchar blue\n", sizeof(char), 20, pFile);
	std::fwrite((void*)"property uint frame\n", sizeof(char), 20, pFile);
	std::fwrite((void*)"property uint flags\n", sizeof(char), 20, pFile);
	std::fwrite((void*)"end_header\n", sizeof(char), 11, pFile);

	uchar c = 255;
	unsigned int frame = 0;
	unsigned int flags = 5;

	int point_num = 0;

	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		vector<int>::iterator itex = find(exclude_cids.begin(), exclude_cids.end(), cid);
		if (itex != exclude_cids.end()) continue;

		Matrix<double, 4, Dynamic> Iws_used(4, num_used_pixels_[cid]);
		Matrix<double, 3, Dynamic> Iss_used(3, num_used_pixels_[cid]);
		Iss_used.row(0) = Xuseds_[cid];
		Iss_used.row(1) = Yuseds_[cid];
		Iss_used.row(2).setOnes();
		Matrix<double, Dynamic, Dynamic> depths = depth_maps_[cid].cast<double>();
		depths.resize(depths.rows()*depths.cols(), 1);
		Matrix<double, Dynamic, 1> depths_used_col = EigenMatlab::TruncateByBooleansRows(&depths, &masks_[cid]);
		Matrix<double, 1, Dynamic> depths_used_row = depths_used_col.transpose();
		InverseProjectSStoWS(cid, &Iss_used, &depths_used_row, &Iws_used);
		//Iws_used = WorldToAgisoft_ * Iws_used;

		if (debug) DebugPrintMatrix(&Xuseds_[cid], "Xuseds_[cid]");
		if (debug) DebugPrintMatrix(&Yuseds_[cid], "Yuseds_[cid]");
		if (debug) DebugPrintMatrix(&Iss_used, "Iss_used");
		if (debug) DebugPrintMatrix(&depths_used_row, "depths_used_row");
		if (debug) DebugPrintMatrix(&WorldToAgisoft_, "WorldToAgisoft_");
		if (debug) DebugPrintMatrix(&Iws_used, "Iws_used");

		std::string fn_cam = GLOBAL_FILEPATH_DATA + scene_name + "\\projections" + std::to_string(cid) + ".ply";
		FILE* pFile_cam = fopen(fn_cam.c_str(), "wb"); // write binary mode

		/*
		ply
		format binary_little_endian 1.0
		element vertex 7765
		property float x
		property float y
		property int id
		end_header
		*/
		Matrix<bool, 1, Dynamic> depths_used_row_notzero = depths_used_row.array() != 0;
		int num_points_cam = depths_used_row_notzero.count();
		std::fwrite((void*)"ply\n", sizeof(char), 4, pFile_cam);
		std::fwrite((void*)"format binary_little_endian 1.0\n", sizeof(char), 32, pFile_cam);
		std::fwrite((void*)"element vertex ", sizeof(char), 15, pFile_cam);
		string num_points_cam_str = std::to_string(num_points_cam);
		char *num_points_cam_chars = convert_string_to_chars(num_points_cam_str);
		std::fwrite((void*)num_points_cam_chars, sizeof(char), num_points_cam_str.length(), pFile_cam);
		delete[] num_points_cam_chars;
		std::fwrite((void*)"\n", sizeof(char), 1, pFile_cam);
		std::fwrite((void*)"property float x\n", sizeof(char), 17, pFile_cam);
		std::fwrite((void*)"property float y\n", sizeof(char), 17, pFile_cam);
		std::fwrite((void*)"property int id\n", sizeof(char), 16, pFile_cam);
		std::fwrite((void*)"end_header\n", sizeof(char), 11, pFile_cam);

		int point_num_cam = 0;
		float ws_x, ws_y, ws_z, ss_x, ss_y;

		for (int c = 0; c < Iws_used.cols(); c++) {
			if (depths_used_row(c) != 0) {
				ws_x = static_cast<float>(Iws_used(0, c));
				ws_y = static_cast<float>(Iws_used(1, c));
				ws_z = static_cast<float>(Iws_used(2, c));
				std::fwrite((void*)&ws_x, sizeof(float), 1, pFile);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile);
				std::fwrite((void*)&ws_y, sizeof(float), 1, pFile);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile);
				std::fwrite((void*)&ws_z, sizeof(float), 1, pFile);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile);

				std::fwrite((void*)&c, sizeof(uchar), 1, pFile);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile);
				std::fwrite((void*)&c, sizeof(uchar), 1, pFile);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile);
				std::fwrite((void*)&c, sizeof(uchar), 1, pFile);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile);

				std::fwrite((void*)&frame, sizeof(unsigned int), 1, pFile);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile);
				std::fwrite((void*)&flags, sizeof(unsigned int), 1, pFile);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile);
				ss_x = static_cast<float>(Iss_used(0, c));
				ss_y = static_cast<float>(Iss_used(1, c));
				std::fwrite((void*)&ss_x, sizeof(float), 1, pFile_cam);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile_cam);
				std::fwrite((void*)&ss_y, sizeof(float), 1, pFile_cam);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile_cam);
				std::fwrite((void*)&point_num, sizeof(int), 1, pFile_cam);
				std::fwrite((void*)"\n", sizeof(char), 1, pFile_cam);
				
				point_num++;
				point_num_cam++;
			}
		}

		std::fclose(pFile_cam);
	}

	std::fclose(pFile);
}

// note that vertex indices and vertex texture coordinate indices are the same and each vertex has only one texture coordinate regardless of face
// mesh must be built first
void StereoData::SaveMesh(string scene_name, int cid) {
	bool debug = false;

	if (debug) cout << "SaveMesh::SaveMesh() scene_name " << scene_name << endl;

	// create mtl file
	std::string fn_mtl = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + std::to_string(cid) + ".mtl";
	ofstream myfile_mtl;
	myfile_mtl.open(fn_mtl);
	myfile_mtl << "newmtl Material" << endl;
	myfile_mtl << "Ka  1.0 1.0 1.0" << endl;
	myfile_mtl << "Kd  1.0 1.0 1.0" << endl;
	myfile_mtl << "Ks  0.0 0.0 0.0" << endl;
	myfile_mtl << "d  1.00" << endl;
	myfile_mtl << "Ns  0.0" << endl;
	myfile_mtl << "illum 0" << endl;
	myfile_mtl << "map_Kd " << scene_name + "_cam" + std::to_string(cid) + ".jpg";
	myfile_mtl.close();

	// create obj file
	std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + std::to_string(cid) + ".obj";
	ofstream myfile;
	myfile.open(fn);

	if (debug) cout << "saving mesh for camera " << cid << " to fn " << fn << endl;

	myfile << "mtllib ./" + scene_name + "_cam" << std::to_string(cid) << ".mtl" << endl;
	myfile << "usemtl Material" << endl;

	map<int, int> vertex_idx_map; // map of current vertex indices, which are 0-indexed and not monotonically increasing, to a 1-indexed list of vertex indices for obj file format; camera ID => current index => new index

	// save vertices
	int idx_curr;
	int idx_new = 1;
	for (map<int, Point3f>::iterator it = meshes_vertices_[cid].begin(); it != meshes_vertices_[cid].end(); ++it) {
		myfile << "v " << (*it).second.x << " " << (*it).second.y << " " << (*it).second.z << endl;

		idx_curr = (*it).first;
		vertex_idx_map[idx_curr] = idx_new;
		idx_new++;
	}

	// save texture coordinates
	for (map<int, Point2f>::iterator it = meshes_texcoords_[cid].begin(); it != meshes_texcoords_[cid].end(); ++it) {
		myfile << "vt " << (*it).second.x << " " << (*it).second.y << endl;
	}

	// save vertex normals - set all to be along the camera space ray through the corresponding screen space pixel
	for (map<int, Point3f>::iterator it = meshes_vertex_normals_[cid].begin(); it != meshes_vertex_normals_[cid].end(); ++it) {
		myfile << "vn " << (*it).second.x << " " << (*it).second.y << " " << (*it).second.z << endl;
	}

	// save faces
	int v1, v2, v3;
	int v1_local, v2_local, v3_local;
	for (map<int, Vec3i>::iterator it = meshes_faces_[cid].begin(); it != meshes_faces_[cid].end(); ++it) {
		v1_local = (*it).second[0];
		v2_local = (*it).second[1];
		v3_local = (*it).second[2];

		v1 = vertex_idx_map[v1_local];
		v2 = vertex_idx_map[v2_local];
		v3 = vertex_idx_map[v3_local];
		//myfile << "f " << v1 << " " << v2 << " " << v3 << endl; // note that vertex indices and vertex texture coordinate indices are the same and each vertex has only one texture coordinate regardless of face
		myfile << "f " << v1 << "/" << v1 << "/" << v1 << " " << v2 << "/" << v2 << "/" << v2 << " " << v3 << "/" << v3 << "/" << v3 << endl; // note that vertex indices, vertex texture coordinate indices, vertex normals are all the same and each vertex has only one texture coordinate regardless of face
	}

	myfile.close();
}

// operates on the saved mesh file, so must save the mesh file first
void StereoData::DecimateMeshes(string scene_name, int cid_specific) {
	bool debug = true;

	cout << "StereoData::DecimateMeshes()" << endl;
	
	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		if (!valid_cam_poses_[cid]) continue;
		if ((cid_specific != -1) &&
			(cid != cid_specific))
			continue;

		std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + std::to_string(cid) + ".obj";
		if (debug) cout << "GLOBAL_TARGET_MESH_FACES " << GLOBAL_TARGET_MESH_FACES << ", meshes_faces_[cid].size() " << meshes_faces_[cid].size() << endl;
		double decimation_ratio = static_cast<double>(GLOBAL_TARGET_MESH_FACES) / static_cast<double>(meshes_faces_[cid].size());

		if (debug) cout << "decimation_ratio " << decimation_ratio << endl;

		if (decimation_ratio >= 1.) continue;

		string command;

		if (debug)
			command = "cd " + GLOBAL_FILEPATH_BLENDER_EXECUTABLE + " && blender --background --python \"" + GLOBAL_FILEPATH_SOURCE + "decimate.py\" -- \"" + fn + "\" \"" + fn + "\" " + to_string(decimation_ratio); // output to screen
		else
			command = "cd " + GLOBAL_FILEPATH_BLENDER_EXECUTABLE + " && blender --background --python \"" + GLOBAL_FILEPATH_SOURCE + "decimate.py\" > NUL 2>&1 -- \"" + fn + "\" \"" + fn + "\" " + to_string(decimation_ratio); // suppress output
		
		if (debug) cout << "command" << endl;

		bool run_successful = false;
		int max_attempts = 3;
		int attempt = 0;
		while ((!run_successful) &&
			(attempt < max_attempts)) {
			cout << "decimation attempt " << attempt << endl;
			system(command.c_str());

			wstring fn_ws;
			StringToWString(fn_ws, fn);
			__int64 fs = FileSize(fn_ws);
			if (fs < GLOBAL_TARGET_MAX_FILE_SIZE) run_successful = true;
			else {
				std::chrono::milliseconds dura(10000);
				std::this_thread::sleep_for(dura);
			}
			attempt++;
		}
		if (!run_successful)
			cout << "decimation was unsuccessful" << endl;
	}
}

// note that vertex indices and vertex texture coordinate indices are the same and each vertex has only one texture coordinate regardless of face
// like SaveMesh(), but saves each camera's mesh separately, and also saves associated material file and adds reference to it in the obj file
void StereoData::SaveMeshes(string scene_name, int cid_specific) {
	bool debug = true;

	if (debug) cout << "StereoData::SaveMeshes() scene_name " << scene_name << endl;

	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		if (!valid_cam_poses_[cid]) continue;
		if ((cid_specific != -1) &&
			(cid != cid_specific))
			continue;

		BuildMeshes(cid);
		SaveMesh(scene_name, cid); // must save before reducing since decimate operates on the saved file
		DecimateMeshes(scene_name, cid);
		LoadMeshes(scene_name, cid);
		BuildTextureCoordinates(cid);
		SaveMesh(scene_name, cid); // save again now that it's been decimated
	}
}

// note: currently, this doesn't build meshes_normals_ data structure properly, though does wipe its old data
void StereoData::LoadMeshes(string scene_name, int cid_specific) {
	bool debug = false;

	cout << "Scene::LoadMeshes()" << endl;

	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		if (!valid_cam_poses_[cid]) continue;
		if ((cid_specific != -1) &&
			(cid != cid_specific))
			continue;

		cout << "loading mesh for camera " << cid << endl;

		meshes_vertices_[cid].erase(meshes_vertices_[cid].begin(), meshes_vertices_[cid].end());
		meshes_vertex_normals_[cid].erase(meshes_vertex_normals_[cid].begin(), meshes_vertex_normals_[cid].end());
		meshes_faces_[cid].erase(meshes_faces_[cid].begin(), meshes_faces_[cid].end());
		meshes_vertex_faces_[cid].erase(meshes_vertex_faces_[cid].begin(), meshes_vertex_faces_[cid].end());

		std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + std::to_string(cid) + ".obj";
		cout << "checking for file " << fn << endl;
		ifstream myfile;
		myfile.open(fn);
		if (!myfile) continue;

		cout << "found file" << endl;

		int vid = 0;
		int vnid = 0;
		int fid = 0;

		std::string line;
		while (std::getline(myfile, line))
		{
			std::istringstream iss(line);

			string code;
			iss >> code;
			if (code == "v") {
				float x, y, z;
				iss >> x >> y >> z;
				Point3f p(x, y, z);
				meshes_vertices_[cid][vid] = p;
				vector<int> emptylist;
				meshes_vertex_faces_[cid][vid] = emptylist;
				vid++;
			}
			else if (code == "vn") {
				float x, y, z;
				iss >> x >> y >> z;
				Point3f p(x, y, z);
				meshes_vertex_normals_[cid][vnid] = p;
				vnid++;
			}
			else if (code == "f") { // first index is a vertex index, second is a texture coordinate (ignored here), third is a vertex normal
				string v1, v2, v3;
				iss >> v1 >> v2 >> v3;
				int fvid1 = ParseObjFaceVertex(v1) - 1; // make 0-indexed;
				int fvid2 = ParseObjFaceVertex(v2) - 1; // make 0-indexed;
				int fvid3 = ParseObjFaceVertex(v3) - 1; // make 0-indexed;

				Vec3i verts;
				verts[0] = fvid1;
				verts[1] = fvid2;
				verts[2] = fvid3;

				meshes_faces_[cid][fid] = verts;

				meshes_vertex_faces_[cid][fvid1].push_back(fid);
				meshes_vertex_faces_[cid][fvid2].push_back(fid);
				meshes_vertex_faces_[cid][fvid3].push_back(fid);

				fid++;
			}
		}

		myfile.close();
	}
}

int StereoData::ParseObjFaceVertex(string fv) {
	int vid;
	std::size_t slashpos = fv.find("/");
	if (slashpos != std::string::npos) {
		string sub;
		sub = fv.substr(0, slashpos);
		vid = atoi(sub.c_str());
	}
	else
		vid = atoi(fv.c_str());

	return vid;
}

// creates a disparity proposal (updating Dproposal) that uses disparity values from camera cid_ref except where cid projects to cid_ref, in which case it replaces the disparities with disparities yielded from reprojecting from cid
// reproj_known is updated to hold values reflecting whether reprojected cid_ref pixel disparities are considered known or unknown for cid
// requires stereo optimization has been run and all "unknown" pixels have non-zero depths except for the pixels that may have been masked-out because there are no valid values according to other camera masks...this can happen if data was loaded from file instead of valid ranges computed along the way
void StereoData::BuildComparisonDisparityProposal(int cid_ref, int cid, Matrix<double, Dynamic, 1> *Dproposal, Matrix<bool, Dynamic, 1> *reproj_known) {
	assert(Dproposal->rows() == num_pixels_[cid_ref]);
	assert(num_pixels_[cid_ref] == num_pixels_[cid]);
	assert(reproj_known->rows() == num_pixels_[cid_ref]);

	bool debug = false;

	cout << "StereoData::BuildComparisonDisparityProposal(" << cid_ref << ", " << cid << ")" << endl;

	(*Dproposal) = disparity_maps_[cid_ref];

	// take used pixels in cid and find their coordinates in world space (Iws_used), the camera space of cid_ref (Icsref_used), and the screen space of cid_ref (Issref_used); Icsref_used yields the disparities that cid believes are valid for the pixels in Issref_used; Issref_used must be checked against the unknown mask of cid_ref - for those which are unknown, the current cid_ref disparity in Dproposal should be replaced by the associated cid disparity from the inverse of Icsref_used's z coordinate (as long as it's not 0)
	Matrix<double, 4, Dynamic> Iws_used(4, num_used_pixels_[cid]);
	Matrix<double, 3, Dynamic> Iss_used(3, num_used_pixels_[cid]);
	Iss_used.row(0) = Xuseds_[cid].transpose();
	Iss_used.row(1) = Yuseds_[cid].transpose();
	Iss_used.row(2).setOnes();
	Matrix<double, Dynamic, Dynamic> depths = depth_maps_[cid].cast<double>();
	depths.resize(depths.rows()*depths.cols(), 1);
	Matrix<double, Dynamic, 1> depths_used_col = EigenMatlab::TruncateByBooleansRows(&depths, &masks_[cid]);
	Matrix<double, 1, Dynamic> depths_used_row = depths_used_col.transpose();
	InverseProjectSStoWS(cid, &Iss_used, &depths_used_row, &Iws_used);
	Matrix<double, 4, Dynamic> Icsref_used = RTs_[cid_ref].cast<double>() * Iws_used;
	Matrix<double, 3, 4> K_ext;
	K_ext.block(0, 0, 3, 3) << Ks_[cid_ref].cast<double>();
	K_ext.col(3) << 0., 0., 0.;
	Matrix<double, 3, Dynamic> Issref_used = K_ext * Icsref_used;
	// normalize SS coords
	Matrix<double, 1, Dynamic> Hcoords = Issref_used.row(2).array().inverse();
	Issref_used.row(0) = Issref_used.row(0).cwiseProduct(Hcoords);
	Issref_used.row(1) = Issref_used.row(1).cwiseProduct(Hcoords);
	Icsref_used.row(2) = Icsref_used.row(2).cwiseQuotient(Icsref_used.row(3));

	if (debug) { // check against standard SS to SS projection function
		Matrix<double, 3, 4> pcid2ref = Pss1Toss2(cid, cid_ref).cast<double>();
		Matrix<double, Dynamic, 4> Iss_used2(num_used_pixels_[cid], 4);
		Iss_used2.col(0) = Xuseds_[cid];
		Iss_used2.col(1) = Yuseds_[cid];
		Iss_used2.col(2).setConstant(1.);
		Matrix<double, Dynamic, 1> disparities_used = DepthMap::ConvertDepthMapToDisparityMap(&depths_used_row);
		Iss_used2.col(3) = disparities_used;
		Matrix<double, Dynamic, 3> I2_ss = Iss_used2 * pcid2ref.transpose();
		Matrix<double, Dynamic, 1> Hcoords2 = I2_ss.col(2).array().inverse();
		I2_ss.col(0) = I2_ss.col(0).cwiseProduct(Hcoords2);
		I2_ss.col(1) = I2_ss.col(1).cwiseProduct(Hcoords2);
		I2_ss.col(2) = I2_ss.col(2).cwiseProduct(Hcoords2);
		Issref_used.row(2) = Issref_used.row(2).cwiseProduct(Hcoords);
		DebugPrintMatrix(&Issref_used, "Issref_used");
		DebugPrintMatrix(&I2_ss, "I2_ss");
	}

	Matrix<bool, Dynamic, 1> known_depths_used = EigenMatlab::TruncateByBooleansRows(&known_depths_[cid], &masks_[cid]); // could have some trues that correspond to points without actual known depths (zero depth values) because mask was expanded but when reloaded was not, so are some masked-in pixels with zeros right now
	reproj_known->setConstant(false);


	if (debug) { // test calculating disparities a different way
		Point3d cam_pos;
		cam_pos.x = RTinvs_[cid_ref](0, 3);
		cam_pos.y = RTinvs_[cid_ref](1, 3);
		cam_pos.z = RTinvs_[cid_ref](2, 3);
		Matrix<double, 1, Dynamic> disps(1, Iws_used.cols());
		double world_to_agisoft_scale = 1. / static_cast<double>(agisoft_to_world_scales_[cid_ref]);
		for (int c = 0; c < Iws_used.cols(); c++) {
			Point3d pt_pos;
			pt_pos.x = Iws_used(0, c);
			pt_pos.y = Iws_used(1, c);
			pt_pos.z = Iws_used(2, c);
			double dist = vecdist(pt_pos, cam_pos);
			double disp;
			if (dist == 0) disp = 0;
			else disp = 1. / dist;
			disps(0, c) = world_to_agisoft_scale * disp;
		}
		DebugPrintMatrix(&disps, "disps");
	}

	Matrix<bool, Dynamic, 1> changed(Dproposal->rows(), 1);
	changed.setConstant(false);
	int x, y, k;
	int h = heights_[cid_ref];
	int w = widths_[cid_ref];
	double d;
	bool known;
	for (int i = 0; i < num_used_pixels_[cid]; i++) {
		x = static_cast<int>(Issref_used(0, i));
		y = static_cast<int>(Issref_used(1, i));
		k = x*h + y;

		if ((x >= 0) &&
			(y >= 0) &&
			(x < w) &&
			(y < h)) {
			d = 1 / Icsref_used(2, i);
			known = known_depths_used(i, 0);
			if (d > 0) {
				if (masks_unknowns_[cid_ref](k, 0)) {
					if ((!changed(k, 0)) ||
						(d >(*Dproposal)(k, 0))) { // use the maximum disparity (minimum depth) that projects to each pixel coordinate given by Issref_used, since more than one may project to the same pixel
						(*Dproposal)(k, 0) = d;
						(*reproj_known)(k, 0) = known;
						changed(k, 0) = true;
					}
				}
				if ((masks_unknowns_[cid_ref](k + h, 0)) &&
					(x < (w - 1))) {
					if ((!changed(k + h, 0)) ||
						(d >(*Dproposal)(k + h, 0))) { // use the maximum disparity (minimum depth) that projects to each pixel coordinate given by Issref_used, since more than one may project to the same pixel
						(*Dproposal)(k + h, 0) = d;
						(*reproj_known)(k + h, 0) = known;
						changed(k + h, 0) = true;
					}
				}
				if ((masks_unknowns_[cid_ref](k + 1, 0)) &&
					(y < (h - 1))) {
					if ((!changed(k + 1, 0)) ||
						(d >(*Dproposal)(k + 1, 0))) { // use the maximum disparity (minimum depth) that projects to each pixel coordinate given by Issref_used, since more than one may project to the same pixel
						(*Dproposal)(k + 1, 0) = d;
						(*reproj_known)(k + 1, 0) = known;
						changed(k + 1, 0) = true;
					}
				}
				if ((masks_unknowns_[cid_ref](k + h +1, 0)) &&
					(x < (w - 1)) &&
					(y < (h - 1))) {
					if ((!changed(k + h + 1, 0)) ||
						(d >(*Dproposal)(k + h + 1, 0))) { // use the maximum disparity (minimum depth) that projects to each pixel coordinate given by Issref_used, since more than one may project to the same pixel
						(*Dproposal)(k + h + 1, 0) = d;
						(*reproj_known)(k + h + 1, 0) = known;
						changed(k + h + 1, 0) = true;
					}
				}
			}
		}
	}

	if (debug) DisplayImages::DisplayDisparityImage(Dproposal, heights_[cid_ref], widths_[cid_ref], orientations_[cid_ref]);
}

// syncs depth and disparity maps, and updates pixel data so that knowns are those sync'd; others may be non-zero and still unknown
// order arg cameras in cid_order so that can propogate in order of closest to farthest, but do not cull camera angles more than 90 degrees away from each other because they may share views on faces with angles less than that from each other
void StereoData::SyncDepthMaps(Scene *scene, vector<int> cid_order) {
	std::map<int, Matrix<bool, Dynamic, 1>> change_maps;
	for (vector<int>::iterator it = cid_order.begin(); it != cid_order.end(); ++it) {
		int cid = (*it);
		if ((!scene->cameras_[cid]->enabled_) ||
			(!scene->cameras_[cid]->posed_) ||
			(!scene->cameras_[cid]->has_depth_map_)) continue; // disqualify cameras for which we are missing important information
		
		Matrix<bool, Dynamic, 1> change_map(depth_maps_[cid].rows()*depth_maps_[cid].cols(), 1);
		change_map.setConstant(false);
		change_maps[cid] = change_map;
	}

	for (vector<int>::iterator it1 = cid_order.begin(); it1 != cid_order.end(); ++it1) {
		int cid1 = (*it1);
		if (!valid_cam_poses_[cid1]) continue;
		if ((!scene->cameras_[cid1]->enabled_) ||
			(!scene->cameras_[cid1]->posed_) ||
			(!scene->cameras_[cid1]->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

		for (vector<int>::iterator it2 = cid_order.begin(); it2 != cid_order.end(); ++it2) {
			int cid2 = (*it2);
			if (cid2 <= cid1) continue;
			if (!valid_cam_poses_[cid2]) continue;
			if ((!scene->cameras_[cid2]->enabled_) ||
				(!scene->cameras_[cid2]->posed_) ||
				(!scene->cameras_[cid2]->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

			scene->cameras_[cid1]->BuildMesh();
			scene->CleanFacesAgainstMasks(cid1);
			scene->cameras_[cid1]->ReprojectMeshDepths(scene->cameras_[cid2]->view_dir_, &scene->cameras_[cid2]->P_, &scene->cameras_[cid2]->RT_, &depth_maps_[cid2], &change_maps[cid2]);
			UpdateDisparityMapFromDepthMap(cid2);

			scene->cameras_[cid2]->BuildMesh();
			scene->CleanFacesAgainstMasks(cid2);
			scene->cameras_[cid2]->ReprojectMeshDepths(scene->cameras_[cid1]->view_dir_, &scene->cameras_[cid1]->P_, &scene->cameras_[cid1]->RT_, &depth_maps_[cid1], &change_maps[cid1]);
			UpdateDisparityMapFromDepthMap(cid1);
		}
	}

	for (vector<int>::iterator it = cid_order.begin(); it != cid_order.end(); ++it) {
		int cid = (*it);
		if ((!scene->cameras_[cid]->enabled_) ||
			(!scene->cameras_[cid]->posed_) ||
			(!scene->cameras_[cid]->has_depth_map_)) continue; // disqualify cameras for which we are missing important information
		
		SpecifyPixelData(cid, &change_maps[cid]);
	}
}

void StereoData::UpdateDisparityMapFromDepthMap(int cid) {
	MatrixXf disparity_mapf = DepthMap::ConvertDepthMapToDisparityMap(&depth_maps_[cid]);
	disparity_mapf.resize(disparity_mapf.rows()*disparity_mapf.cols(), 1);
	Matrix<double, Dynamic, 1> disparity_mapd = disparity_mapf.cast<double>();
	disparity_maps_[cid] = disparity_mapd;
}

void StereoData::UpdateDepthMapFromDisparityMap(int cid) {
	MatrixXd depth_mapd = DepthMap::ConvertDisparityMapToDepthMap(&disparity_maps_[cid]);
	depth_mapd.resize(depth_maps_[cid].rows(), depth_maps_[cid].cols());
	Matrix<float, Dynamic, Dynamic> depth_mapf = depth_mapd.cast<float>();
	depth_maps_[cid] = depth_mapf;
}

void StereoData::SmoothDisparityMaps() {
	cout << "StereoData::SmoothDisparityMaps()" << endl;

	for (std::map<int, Mat>::iterator it = imgsT_.begin(); it != imgsT_.end(); ++it) {
		int cid = (*it).first;
		if (!valid_cam_poses_[cid]) continue;
		Matrix<double, Dynamic, 1> disps = disparity_maps_[cid];
		SmoothDisparityMap(cid, &disps);
		disparity_maps_[cid] = disps;
		UpdateDepthMapFromDisparityMap(cid);
	}
}

void StereoData::SmoothDisparityMap(int cid, Matrix<double, Dynamic, 1> *D, int kernel_size, int smooth_iters) {
	bool debug = false;

	cout << "StereoData::SmoothDisparityMap() cid " << cid << endl;

	// Dnew = (D - vals.d_min) / vals.d_step;  // changes Dnew from disparity values to a number in the range [0,1] that represents a disparity from d_min (0) to d_max (1), which is important to do before interpolation, but then must un-normalize again after interpolation
	//(*D) = D->array() - min_disps_[cid];
	//double factor = 1. / (max_disps_[cid] - min_disps_[cid]);
	//(*D) *= factor;

	// apply gaussian smoothing with masking according to segment in sd_->segs_
	int kernel_side = (kernel_size - 1) / 2;

	// create the kernel
	Matrix<double, Dynamic, Dynamic> kernel(kernel_size, kernel_size);
	double sigma = 1;// ((static_cast<double>(kernel_size) / 2.) - 1.) * 0.3 + 0.8;
	//double A = 1. / (2 * CV_PI*pow(sigma, 2)); // normalization (regularization) constant
	double *pK = kernel.data();
	for (int c = 0; c < kernel_size; ++c)
	{
		double x = c - kernel_side; // in 2D Gaussian eqn, assume center coord is (0,0)
		for (int r = 0; r < kernel_size; ++r)
		{
			double y = r - kernel_side; // in 2D Gaussian eqn, assume center coord is (0,0)
			*pK++ = exp(-1 * (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))); // Gaussian formula in x,y here
		}
	}
	double A = 1. / kernel.sum();
	kernel *= A;
	if (debug) {
		DebugPrintMatrix(&kernel, "kernel");
		DisplayImages::DisplayGrayscaleImage(&kernel, kernel_size, kernel_size);
		DebugPrintMatrix(D, "D before");
	}

	Matrix<double, Dynamic, Dynamic> D_updated = (*D);
	D_updated.resize(heights_[cid], widths_[cid]);
	if (debug) DebugPrintMatrix(&D_updated, "D_updated");

	for (int iter = 0; iter < smooth_iters; iter++) {
		Matrix<double, Dynamic, Dynamic> D_iter = D_updated;

		// apply the kernel; for each pixel, find its label and use its value in place of the values of neighboring pixels that have a different label so we don't smooth across different segments
		unsigned int label_center, label_curr;
		double val_center, val_curr, val_kernel, val_new;
		double sum_factors; // handles cases where full kernel cannot be applied - used as sum of kernel factors actually applied to normalize afterward
		int idx, idx2;
		for (int c = kernel_side; c < (widths_[cid] - kernel_side); c++) {
			for (int r = kernel_side; r < (heights_[cid] - kernel_side); r++) {
				idx = PixIndexFwdCM(Point(c, r), heights_[cid]);
				//if (debug) assert((idx >= 0) && (idx < num_pixels_[cid]));
				if (!masks_[cid](idx, 0)) continue;
				if (!masks_unknowns_[cid](idx, 0)) continue; // static knowns
				label_center = segs_[cid](r, c);
				val_center = D_updated(r, c);
				pK = kernel.data();
				val_new = 0;
				sum_factors = 0;
				for (int x = (c - kernel_side); x <= (c + kernel_side); x++) {
					for (int y = (r - kernel_side); y <= (r + kernel_side); y++) {
						idx2 = PixIndexFwdCM(Point(x, y), heights_[cid]);
						//if (debug) assert((idx2 >= 0) && (idx2 < num_pixels_[cid]));
						label_curr = segs_[cid](y, x);
						if ((label_curr != label_center) ||
							(!masks_[cid](idx2, 0)) ||
							(masks_int_[cid](idx2, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL)) // add the masks_int_ check so that border values do not influence smoothing since may behave strangely due to EW_ values, resulting in pulling of nearby depths toward adjacent segments; check for masked-out or on a line segment
							continue;
						val_curr = D_updated(y, x);
						val_kernel = *pK++;
						val_new += val_kernel * val_curr;
						sum_factors += val_kernel;
					}
				}
				val_new /= sum_factors; // sum_factors will be 1 if entire kernel has been applied, and will normalize val_new if not

				D_iter(r, c) = val_new;
			}
		}

		//if (debug) DebugPrintMatrix(&D_iter, "D_iter");
		D_updated = D_iter;
		

		//cout << "iteration " << iter << endl;
		//if (debug) DisplayImages::DisplayGrayscaleImage(&D_updated, heights_[cid], widths_[cid], orientations_[cid]);
	}

	D_updated.resize(heights_[cid] * widths_[cid], 1);
	(*D) = D_updated;
	
	// un-normalize disparity map
	//(*D) *= (max_disps_[cid] - min_disps_[cid]);
	//(*D) = D->array() + min_disps_[cid];

	//SnapDisparitiesToValidRanges(cid, D); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable
	if (debug) DebugPrintMatrix(D, "D after");
	if (debug) DisplayImages::DisplayGrayscaleImage(D, heights_[cid], widths_[cid], orientations_[cid]);
}

// fill pixels in segment with label seg_label that have zeros according to disocclusion fill algorithm from Zinger 2010
// D is height x width
bool StereoData::FillSegmentDisparityZeros(int cid, int seg_label, Matrix<double, Dynamic, Dynamic> *D) {
	bool debug = false;

	if (debug) cout << "StereoData::FillSegmentDisparityZeros() cid " << cid << ", seg_label " << seg_label << endl;

	if (debug) DisplayImages::DisplayGrayscaleImage(D, heights_[cid], widths_[cid], orientations_[cid]);

	Matrix<double, Dynamic, Dynamic> D_updated = (*D);
	
	bool made_change = false;
	unsigned int label_center, label_curr;
	int idx, x, y;
	bool still_looking, found_one;
	double numerator, denominator, t, distsqd;
	for (int c = 0; c < widths_[cid]; c++) {
		for (int r = 0; r < heights_[cid]; r++) {
			idx = PixIndexFwdCM(Point(c, r), heights_[cid]);
			//if (debug) assert((idx >= 0) && (idx < num_pixels_[cid]));
			label_center = segs_[cid](r, c);
			if (label_center != seg_label) continue; // only concern ourselves with the segment of interest
			if (!masks_[cid](idx, 0)) continue;
			if (!masks_unknowns_[cid](idx, 0)) continue; // static knowns
			if ((*D)(r, c) != 0) continue; // only update pixels with 0 disparity value

			// search in 8 canonical directions for non-zero pixels from this segment
			numerator = 0.;
			denominator = 0.;
			found_one = false;
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					if ((i == 0) && (j == 0)) continue;
					x = c;
					y = r;
					still_looking = true;
					while (still_looking) {
						x += i;
						y += j;
						if ((x < 0) || (x >= widths_[cid]) ||
							(y < 0) || (y >= heights_[cid])) {
							still_looking = false;
							continue;
						}
						label_curr = segs_[cid](y, x);
						if (label_curr != seg_label) {
							still_looking = false;
							continue;
						}
						t = (*D)(y, x);
						if (t == 0.) continue;
						distsqd = pow(static_cast<double>(x - c), 2) + pow(static_cast<double>(y - r), 2);
						numerator += t / distsqd;
						denominator += 1. / distsqd;
						still_looking = false;
						found_one = true;
					}
				}
			}
			if (found_one) {
				D_updated(r, c) = numerator / denominator;
				//cout << "updated D_updated(" << r << ", " << c << ") to " << numerator / denominator << endl;
				//cin.ignore();
				made_change = true;
			}
		}
	}

	(*D) = D_updated;

	if (debug) DisplayImages::DisplayGrayscaleImage(D, heights_[cid], widths_[cid], orientations_[cid]);

	return made_change;
}

// apply gaussian smoothing on the given segment seg_label that works as follows: only pixels in the same segment can have an effect on value; also, only pixels with true value in trustMask can have an effect on value; only pixels with a false value in trustMask can be altered
void StereoData::SmoothDisparityMapSegmentFromTrusted(int cid, int seg_label, Matrix<double, Dynamic, 1> *D, Matrix<bool, Dynamic, Dynamic> *trustMask, int kernel_size) {
	bool debug = false;

	if (debug) cout << "StereoData::SmoothDisparityMapSegmentFromTrusted() cid " << cid << endl;

	// Dnew = (D - vals.d_min) / vals.d_step;  // changes Dnew from disparity values to a number in the range [0,1] that represents a disparity from d_min (0) to d_max (1), which is important to do before interpolation, but then must un-normalize again after interpolation
	//(*D) = D->array() - min_disps_[cid];
	//double factor = 1. / (max_disps_[cid] - min_disps_[cid]);
	//(*D) *= factor;

	// apply gaussian smoothing with masking according to segment in sd_->segs_
	int kernel_side = (kernel_size - 1) / 2;

	// create the kernel
	Matrix<double, Dynamic, Dynamic> kernel(kernel_size, kernel_size);
	double sigma = 1;// ((static_cast<double>(kernel_size) / 2.) - 1.) * 0.3 + 0.8;
	//double A = 1. / (2 * CV_PI*pow(sigma, 2)); // normalization (regularization) constant
	double *pK = kernel.data();
	for (int c = 0; c < kernel_size; ++c)
	{
		double x = c - kernel_side; // in 2D Gaussian eqn, assume center coord is (0,0)
		for (int r = 0; r < kernel_size; ++r)
		{
			double y = r - kernel_side; // in 2D Gaussian eqn, assume center coord is (0,0)
			*pK++ = exp(-1 * (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))); // Gaussian formula in x,y here
		}
	}
	double A = 1. / kernel.sum();
	kernel *= A;
	if (debug) {
		DebugPrintMatrix(&kernel, "kernel");
		DisplayImages::DisplayGrayscaleImage(&kernel, kernel_size, kernel_size);
		DebugPrintMatrix(D, "D before");
	}

	Matrix<double, Dynamic, Dynamic> D_updated = (*D);
	D_updated.resize(heights_[cid], widths_[cid]);
	if (debug) DebugPrintMatrix(&D_updated, "D_updated");

	// apply the kernel; for each pixel, find its label and use its value in place of the values of neighboring pixels that have a different label so we don't smooth across different segments
	unsigned int label_center, label_curr;
	double val_center, val_curr, val_kernel, val_new;
	double sum_factors; // handles cases where full kernel cannot be applied - used as sum of kernel factors actually applied to normalize afterward
	int idx, idx2, num_influencers;
	for (int c = 0; c < widths_[cid]; c++) {
		for (int r = 0; r < heights_[cid]; r++) {
			idx = PixIndexFwdCM(Point(c, r), heights_[cid]);
			//if (debug) assert((idx >= 0) && (idx < num_pixels_[cid]));
			label_center = segs_[cid](r, c);
			if (label_center != seg_label) continue; // only concern ourselves with the segment of interest
			if (!masks_[cid](idx, 0)) continue;
			if (!masks_unknowns_[cid](idx, 0)) continue; // static knowns
			if ((*trustMask)(r, c)) continue; // static trusted values
			val_center = D_updated(r, c);
			pK = kernel.data();
			val_new = 0;
			sum_factors = 0;
			num_influencers = 0;
			for (int x = (c - kernel_side); x <= (c + kernel_side); x++) {
				for (int y = (r - kernel_side); y <= (r + kernel_side); y++) {
					if ((x < 0) || (x >= widths_[cid]) ||
						(y < 0) || (y >= heights_[cid]))
						continue;
					idx2 = PixIndexFwdCM(Point(x, y), heights_[cid]);
					//if (debug) assert((idx2 >= 0) && (idx2 < num_pixels_[cid]));
					label_curr = segs_[cid](y, x);
					if ((label_curr != label_center) ||
						(!masks_[cid](idx2, 0)) ||
						(!(*trustMask)(y, x)) || // must be a trusted pixel
						(masks_int_[cid](idx2, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL)) // add the masks_int_ check so that border values do not influence smoothing since may behave strangely due to EW_ values, resulting in pulling of nearby depths toward adjacent segments; check for masked-out or on a line segment
						continue;
					val_curr = D_updated(y, x);
					val_kernel = *pK++;
					val_new += val_kernel * val_curr;
					sum_factors += val_kernel;
					num_influencers++;
				}
			}

			if (num_influencers > 0) {
				val_new /= sum_factors; // sum_factors will be 1 if entire kernel has been applied, and will normalize val_new if not
				D_updated(r, c) = val_new;
			}
		}
	}

	D_updated.resize(heights_[cid] * widths_[cid], 1);
	(*D) = D_updated;

	// un-normalize disparity map
	//(*D) *= (max_disps_[cid] - min_disps_[cid]);
	//(*D) = D->array() + min_disps_[cid];

	//SnapDisparitiesToValidRanges(cid, D); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable
	if (debug) DebugPrintMatrix(D, "D after");
	if (debug) DisplayImages::DisplayGrayscaleImage(D, heights_[cid], widths_[cid], orientations_[cid]);
}

// kernel_size must be odd and >=3
void StereoData::SmoothDisparityMap_Avg(int cid, Matrix<double, Dynamic, 1> *D, int kernel_size, int smooth_iters) {
	bool debug = false;

	cout << "StereoData::SmoothDisparityMap() cid " << cid << endl;

	// Dnew = (D - vals.d_min) / vals.d_step;  // changes Dnew from disparity values to a number in the range [0,1] that represents a disparity from d_min (0) to d_max (1), which is important to do before interpolation, but then must un-normalize again after interpolation
	//(*D) = D->array() - min_disps_[cid];
	//double factor = 1. / (max_disps_[cid] - min_disps_[cid]);
	//(*D) *= factor;

	// apply gaussian smoothing with masking according to segment in sd_->segs_
	int kernel_side = (kernel_size - 1) / 2;

	// create the kernel
	Matrix<double, Dynamic, Dynamic> kernel(kernel_size, kernel_size);
	double num_neighbors = static_cast<double>(kernel_size * kernel_size) - 1; // excludes current pixel
	double *pK = kernel.data();
	for (int c = 0; c < kernel_size; ++c)
	{
		double x = c - kernel_side; // in 2D Gaussian eqn, assume center coord is (0,0)
		for (int r = 0; r < kernel_size; ++r)
		{
			double y = r - kernel_side; // in 2D Gaussian eqn, assume center coord is (0,0)
			*pK++ = 1. / num_neighbors;
		}
	}
	if (debug) {
		DebugPrintMatrix(&kernel, "kernel");
		DisplayImages::DisplayGrayscaleImage(&kernel, kernel_size, kernel_size);
		DebugPrintMatrix(D, "D before");
	}

	Matrix<double, Dynamic, Dynamic> D_updated = (*D);
	D_updated.resize(heights_[cid], widths_[cid]);

	for (int iter = 0; iter < smooth_iters; iter++) {
		Matrix<double, Dynamic, Dynamic> D_iter = D_updated;

		// apply the kernel; for each pixel, find its label and use its value in place of the values of neighboring pixels that have a different label so we don't smooth across different segments
		unsigned int label_center, label_curr;
		double val_center, val_curr, val_kernel, val_new;
		int idx, idx2;
		for (int c = kernel_side; c < widths_[cid] - kernel_side; c++) {
			for (int r = kernel_side; r < heights_[cid] - kernel_side; r++) {
				idx = PixIndexFwdCM(Point(c, r), heights_[cid]);
				if (!masks_[cid](idx, 0)) continue;
				if (!masks_unknowns_[cid](idx, 0)) continue; // static knowns
				label_center = segs_[cid](r, c);
				val_center = D_updated(r, c);
				pK = kernel.data();
				val_new = 0;
				for (int x = (c - kernel_side); x <= (c + kernel_side); x++) {
					for (int y = (r - kernel_side); y <= (r + kernel_side); y++) {
						idx2 = PixIndexFwdCM(Point(x, y), heights_[cid]);
						label_curr = segs_[cid](y, x);
						if ((label_curr != label_center) ||
							(!masks_[cid](idx2, 0)) ||
							(masks_int_[cid](idx2, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL)) // add the masks_int_ check so that border values do not influence smoothing since may behave strangely due to EW_ values, resulting in pulling of nearby depths toward adjacent segments; masked-out or on a line segment
							val_curr = val_center;
						else
							val_curr = D_updated(y, x);
						val_kernel = *pK++;
						val_new += val_kernel * val_curr;
					}
				}

				D_iter(r, c) = val_new;
			}
		}

		D_updated = D_iter;

		//cout << "iteration " << iter << endl;
		//if (debug) DisplayImages::DisplayGrayscaleImage(&D_updated, heights_[cid], widths_[cid], orientations_[cid]);
	}

	D_updated.resize(heights_[cid] * widths_[cid], 1);
	(*D) = D_updated;

	// un-normalize disparity map
	//(*D) *= (max_disps_[cid] - min_disps_[cid]);
	//(*D) = D->array() + min_disps_[cid];

	//SnapDisparitiesToValidRanges(cid, D); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable
	if (debug) DebugPrintMatrix(D, "D after");
	if (debug) DisplayImages::DisplayGrayscaleImage(D, heights_[cid], widths_[cid], orientations_[cid]);
}

// zero-depth, masked-in values are assigned the mean of their non-zero neighbors within a kernel_size pixels kernel, until all masked-in zeros have a value
void StereoData::SmoothDisparityMap_MeanNonZero(int cid, Matrix<double, Dynamic, 1> *D, int kernel_size) {
	bool debug = false;

	cout << "StereoData::SmoothDisparityMap_MeanNonZero() cid " << cid << endl;

	// Dnew = (D - vals.d_min) / vals.d_step;  // changes Dnew from disparity values to a number in the range [0,1] that represents a disparity from d_min (0) to d_max (1), which is important to do before interpolation, but then must un-normalize again after interpolation
	//(*D) = D->array() - min_disps_[cid];
	//double factor = 1. / (max_disps_[cid] - min_disps_[cid]);
	//(*D) *= factor;

	// apply gaussian smoothing with masking according to segment in sd_->segs_
	int kernel_side = (kernel_size - 1) / 2;

	Matrix<double, Dynamic, Dynamic> D_updated = (*D);
	D_updated.resize(heights_[cid], widths_[cid]);

	Matrix<bool, Dynamic, Dynamic> mask = masks_[cid];
	mask.resize(heights_[cid], widths_[cid]);
	Matrix<bool, Dynamic, Dynamic> zerotest = D_updated.array() == 0;
	zerotest = (mask.array() == false).select(Matrix<bool, Dynamic, Dynamic>::Constant(zerotest.rows(), zerotest.cols(), false), zerotest);
	int num_zeros = zerotest.count();

	/*
	bool *pM = mask.data();
	bool *pZ = zerotest.data();
	for (int c = 0; c < widths_[cid]; c++) {
		for (int r = 0; r < heights_[cid]; r++) {
			if (*pM++) 
		}
	}
	*/

	int num_zeros_last = num_zeros; // use this to prevent infinite loops that can occur when the algorithm cannot change a masked-in zero depth; can occur when an entire segment is filled with zeros, or at least one contiguous portion if, for some reason, the segment is incorrectly incontiguous
	while (num_zeros > 0) {
		Matrix<double, Dynamic, Dynamic> D_iter = D_updated;

		// apply the kernel; for each pixel, find its label and use its value in place of the values of neighboring pixels that have a different label so we don't smooth across different segments
		unsigned int label_center, label_curr;
		double val_center, val_curr;
		int  idx2;
		for (int c = kernel_side; c < widths_[cid] - kernel_side; c++) {
			for (int r = kernel_side; r < heights_[cid] - kernel_side; r++) {
				if (!mask(r,c)) continue;
				label_center = segs_[cid](r, c);
				val_center = D_updated(r, c);
				if (val_center != 0) continue;
				int num_nonzero_neighbors = 0;
				double sum = 0;
				for (int x = (c - kernel_side); x <= (c + kernel_side); x++) {
					for (int y = (r - kernel_side); y <= (r + kernel_side); y++) {
						idx2 = PixIndexFwdCM(Point(x, y), heights_[cid]);
						label_curr = segs_[cid](y, x);
						val_curr = D_updated(y, x);
						if ((label_curr == label_center) &&
							(mask(y,x)) &&
							(masks_int_[cid](idx2, 0) > GLOBAL_MAX_MASKSEG_LINEVAL) && // masked-in and not on a line segment
							(val_curr != 0)) {
							sum += val_curr;
							num_nonzero_neighbors++;
						}
					}
				}

				if (num_nonzero_neighbors > 0)
					D_iter(r, c) = sum / num_nonzero_neighbors;
			}
		}

		D_updated = D_iter;
		
		zerotest = D_updated.array() == 0;
		zerotest = (mask.array() == false).select(Matrix<bool, Dynamic, Dynamic>::Constant(zerotest.rows(), zerotest.cols(), false), zerotest);
		num_zeros = zerotest.count();

		if (debug) cout << "iteration with remaining num_zeros " << num_zeros << endl;
		if (debug) DisplayImages::DisplayGrayscaleImage(&D_updated, heights_[cid], widths_[cid], orientations_[cid]);

		if (num_zeros == num_zeros_last) break; // use this to prevent infinite loops that can occur when the algorithm cannot change a masked-in zero depth; can occur when an entire segment is filled with zeros, or at least one contiguous portion if, for some reason, the segment is incorrectly incontiguous
		num_zeros_last = num_zeros;
	}

	D_updated.resize(heights_[cid] * widths_[cid], 1);
	(*D) = D_updated;

	// un-normalize disparity map
	//(*D) *= (max_disps_[cid] - min_disps_[cid]);
	//(*D) = D->array() + min_disps_[cid];

	//SnapDisparitiesToValidRanges(cid, D); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable
	if (debug) DebugPrintMatrix(D, "D after");
	if (debug) DisplayImages::DisplayGrayscaleImage(D, heights_[cid], widths_[cid], orientations_[cid]);
}


/*
// invalid disparities in other images should be replaced with a disparity from img_ref. If no pixel from img_ref projects there, use the closest from other images. If nothing from another image either, average pixels around it. Don’t update img2 invalid disparities from img_ref until img_ref is Boolean tested against all other images so the entire crowd weighs in. fs
// requires that reproj_ss_pts have already been normalized by their homogeneous values
// updates disparity_maps_[cid]
void StereoData::ReplaceInvalids(int cid_ref, int cid, Matrix<bool, Dynamic, 1> *invalids, Matrix<double, Dynamic, 3> *reproj_ss_pts) {
	assert(invalids->rows() == num_unknown_pixels_[cid_ref]);
	assert(reproj_ss_pts->rows() == num_unknown_pixels_[cid_ref]);

	if (invalids->count() == 0) return;

	Matrix<double, Dynamic, 4> WC(num_unknown_pixels_[cid_ref], 4);
	WC.col(0) = reproj_ss_pts->col(0);
	WC.col(1) = reproj_ss_pts->col(1);
	WC.col(2).setOnes();

	int h = imgsT_[cid].rows;
	int w = imgsT_[cid].cols;
	Matrix<double, Dynamic, 1> X = reproj_ss_pts->col(0);
	Matrix<double, Dynamic, 1> Y = reproj_ss_pts->col(1);
	Matrix<double, Dynamic, 1> disps_unk(num_unknown_pixels_[cid_ref], 1);
	Interpolation::Interpolate(w, h, &disparity_maps_[cid], &X, &Y, invalids, 0, &disps_unk);
	WC.col(3) = disps_unk;

	bool *pI = invalids->data();
	double *pX = reproj_ss_pts->col(0).data();
	double *pY = reproj_ss_pts->col(1).data();
	int x, y, idx;

	Matrix<double, Dynamic, 3> T(num_unknown_pixels_[cid_ref], 3); // reprojected SS coords
	Matrix<double, Dynamic, 1> H; // to hold homogeneous coordinates after reprojection
	Matrix<double, Dynamic, 1> disps_inv(num_unknown_pixels_[cid_ref], 1);
	disps_inv.setZero();
	Matrix<double, Dynamic, 1> disparities_inv_full(imgsT_[cid].cols * imgsT_[cid].rows, 1);

	// calculate the coordinates in the input image
	Matrix<float, 3, 4> Pout2in = Pss1Toss2(cid, cid_ref);
	T = WC * Pout2in.transpose().cast<double>();
	H = T.col(2).array().inverse(); // determine homogeneous coordinates to divide by
	T.col(0) = T.col(0).cwiseProduct(H); // divide by homogeneous coordinates
	T.col(1) = T.col(1).cwiseProduct(H); // divide by homogeneous coordinates

	X = T.col(0);
	Y = T.col(1);
	Interpolation::Interpolate<double>(imgsT_[cid].cols, imgsT_[cid].rows, &disparity_maps_[cid_ref], &X, &Y, &masks_dilated_[cid_ref], 0, &disps_unk);
	EigenMatlab::AssignByBooleans(&disps_inv, invalids, &disps_unk);
	disparities_inv_full.setZero();
	EigenMatlab::AssignByTruncatedBooleans(&disparities_inv_full, &masks_unknowns_[cid], &disps_inv); // ExpandUnknownToFullSize with cid instead of cid_out_
	disparity_maps_[cid] = (disparities_inv_full.array() != 0).select(disparities_inv_full, disparity_maps_[cid]); // sets disparities coefficients to disparities_reproj for positions where disparities_reproj coefficients != 0


	for (std::map<int, Eigen::Matrix<float, Dynamic, 3>>::iterator it = As_.begin(); it != As_.end(); ++it) {
		int cid2 = (*it).first;
		if ((cid2 == cid_ref) ||
			(cid2 == cid))
			continue;
		if (!valid_cam_poses_[cid2]) continue; // cams with inaccurate poses are not included in mask-checking
		if (invalids->count() == 0) break;

		// calculate the coordinates in the input image
		Matrix<float, 3, 4> Pout2in = Pss1Toss2(cid, cid_ref);
		T = WC * Pout2in.transpose().cast<double>();
		H = T.col(2).array().inverse(); // determine homogeneous coordinates to divide by
		T.col(0) = T.col(0).cwiseProduct(H); // divide by homogeneous coordinates
		T.col(1) = T.col(1).cwiseProduct(H); // divide by homogeneous coordinates

		X = T.col(0);
		Y = T.col(1);
		Interpolation::Interpolate<double>(imgsT_[cid].cols, imgsT_[cid].rows, &disparity_maps_[cid2], &X, &Y, &masks_dilated_[cid2], 0, &disps_unk);
		EigenMatlab::AssignByBooleans(&disps_inv, invalids, &disps_unk);
		disparities_inv_full.setZero();
		EigenMatlab::AssignByTruncatedBooleans(&disparities_inv_full, &masks_unknowns_[cid], &disps_inv); // ExpandUnknownToFullSize with cid instead of cid_out_
		disparity_maps_[cid] = (disparities_inv_full.array() != 0).select(disparities_inv_full, disparity_maps_[cid]); // sets disparities coefficients to disparities_reproj for positions where disparities_reproj coefficients != 0
	}
}
*/