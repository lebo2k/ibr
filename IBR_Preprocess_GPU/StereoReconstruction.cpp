#include "StereoReconstruction.h"

StereoReconstruction::StereoReconstruction(std::string scene_name) {
	scene_name_ = scene_name;
	sd_ = new StereoData();
	seg_ = new Segmentation();
	op_ = new Optimization();
}

StereoReconstruction::~StereoReconstruction() {
	delete sd_;
	delete seg_;
	delete op_;
}

// initialize using Middlebury's cones dataset
void StereoReconstruction::Init_MiddleburyCones(int cid_ref) {
	sd_->Init_MiddleburyCones(cid_ref);
}

// bv_min and bv_max are the minimum and maximum bounding volume coordinates for the point cloud in the scene captured by all input images in imgsT ... used to determine minimum and maximum depth bound in the camera space for the output camera
void StereoReconstruction::Init(std::map<int, Mat> imgsT, std::map<int, Mat> imgMasks, std::map<int, Mat> imgMasks_valid, std::map<int, MatrixXf> depth_maps, std::map<int, Matrix3d> Ks, std::map<int, Matrix3d> Kinvs, std::map<int, Matrix4d> RTs, std::map<int, Matrix4d> RTinvs, std::map<int, Matrix<double, 3, 4>> Ps, std::map<int, Matrix<double, 4, 3>> Pinvs, std::map<int, float> min_depths, std::map<int, float> max_depths, std::map<int, std::map<unsigned int, Matrix<unsigned int, Dynamic, 1>>> unknown_segs, std::map<int, float> agisoft_to_world_scales, Matrix4d AgisoftToWorld, Matrix4d WorldToAgisoft, std::vector<int> exclude_cam_ids, int max_num_cams, Scene *scene) {

	sd_->Init(imgsT, imgMasks, imgMasks_valid, depth_maps, Ks, Kinvs, RTs, RTinvs, Ps, Pinvs, min_depths, max_depths, unknown_segs, agisoft_to_world_scales, AgisoftToWorld, WorldToAgisoft, exclude_cam_ids, max_num_cams, scene);
}

// syncs disparity data across cameras as best as possible
void StereoReconstruction::SyncDisparityData(std::vector<int> cids, int max_num_cams, Scene *scene) {
	bool debug = false;
	bool debug_save_by_camera = true;

	cout << "StereoReconstruction::SyncDisparityData()" << endl;

	// sync them; order arg cameras in cid_order so that can propogate in order of closest to farthest, but do not cull camera angles more than 90 degrees away from each other because they may share views on faces with angles less than that from each other
	int cid_out = *cids.begin();
	Matrix4d RTinv_out = sd_->RTinvs_[cid_out].cast<double>();
	Point3d view_pos = Camera::GetCameraPositionWS(&RTinv_out);
	Point3d view_dir = Camera::GetCameraViewDirectionWS(&RTinv_out);
	std::vector<int> exclude_cam_ids;
	std::vector<int> ordered_cids = scene->GetClosestCams(view_pos, view_dir, exclude_cam_ids, cids.size(), false);
	sd_->SyncDepthMaps(scene, ordered_cids);

	// smooth the results
	for (std::map<int, Camera*>::iterator it1 = scene->cameras_.begin(); it1 != scene->cameras_.end(); ++it1) {
		if ((!(*it1).second->enabled_) ||
			(!(*it1).second->posed_) ||
			(!(*it1).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

		int cid_out = (*it1).first;
		if (!sd_->valid_cam_poses_[cid_out]) continue;

		if (sd_->num_unknown_pixels_[cid_out] > 0) {
			//SmoothDisparityData(cid_out);

			Matrix<double, Dynamic, 1> disps = sd_->disparity_maps_[cid_out];
			sd_->SmoothDisparityMap(cid_out, &disps);
			sd_->disparity_maps_[cid_out] = disps;
			sd_->UpdateDepthMapFromDisparityMap(cid_out);
		}

		if (debug_save_by_camera) {
			scene->cameras_[cid_out]->dm_->depth_map_ = sd_->depth_maps_[cid_out];
			scene->UpdateDisparities();
			std::string mat_dm_name = scene_name_ + "_Dsynced_out_" + to_string(cid_out);
			SaveEigenMatrix(mat_dm_name, TYPE_FLOAT, sd_->depth_maps_[cid_out]);
			DisplayImages::SaveGrayscaleImage(&sd_->depth_maps_[cid_out], sd_->heights_[cid_out_], sd_->widths_[cid_out_], mat_dm_name);
			scene->cameras_[cid_out]->SaveRenderingDataRLE(scene->name_, scene->min_disps_[cid_out], scene->max_disps_[cid_out], scene->disp_steps_[cid_out]);
		}
	}
}

// note: crashes during the sd_->BuildValidDisparityRanges call when called after computing disparity maps (echoes to screen the first cout from BuildValidDisparityRanges before crashing, and never echoes the running time that should show up when the procedure completes); if disparity maps are precomputed and loaded in, does not crash there; not sure what the reason for the crash is yet
void StereoReconstruction::SmoothDisparityData(int cid_out) {
	bool debug = false;

	cout << "StereoReconstruction::SmoothDisparityData()" << endl;

	cid_out_ = cid_out;

	sd_->ClearStatistics(cid_out_); // perform before loading full map so get unknown pixels from partial map to explore
	sd_->BuildValidDisparityRanges(cid_out_); // even if had computed in the past, must recompute now to collect data for new set of unknowns
	op_->Init(sd_, cid_out_);

	Matrix<double, Dynamic, 1> Dcurr = sd_->disparity_maps_[cid_out_];
	Matrix<double, Dynamic, 1> Dproposal = Dcurr;

	Matrix<double, Dynamic, 1> energy(sd_->max_iters_, 1);
	energy.setZero();
	sd_->map_.setZero();
	sd_->count_updated_vals_.setZero();
	sd_->count_unlabelled_vals_.setZero();
	sd_->count_unlabelled_regions_.setZero();
	sd_->count_unlabelled_after_QPBOP_.setZero();
	sd_->timings_data_term_eval_.setZero();
	sd_->timings_smoothness_term_eval_.setZero();
	sd_->timings_qpbo_fuse_time_.setZero();
	sd_->timings_iteration_time_.setZero();

	int iter = sd_->average_over_; // starting value for iter (not +1 because 0-indexed instead of 1-indexed like Matlab)

	// initialize energies of phantom iterations to the maximum value. Phantom iterations are front-padded results to ensure can always average over the correct number of iterations; making them the maximum value ensures convergence is not achieved until enough iterations have been performed that these fall off the grid, so to speak. Another solution would be to simply run the loop for a minimum number of iterations before checking for convergence.
	for (int r = 0; r < sd_->average_over_; r++) {
		energy(r, 0) = DBL_MAX; // for float, would be FLT_MAX
	}
	energy(iter, 0) = DBL_MAX / 1e20; // ensures initial while check for loop entry does not meet convergence test by setting current iteration's energy significantly lower than earlier energies, so stabilization has not yet been achieved.

	// for the loop condition: note that energy is guaranteed not to increase, so convergence is checked by determining ratio of current energy to energy average_over iterations ago, then comparing to a convergence threshold that is dependent on the number of iterations over which we are averaging convergence, yielding a check that is truly an "averaging" of convergence.  More generally, if energy is falling, loop will continue.  Once has stabilized sufficiently, or once we exceed the maximum number of iterations, the loop will terminate.
	while ((1 - (energy(iter, 0) / energy(iter - sd_->average_over_, 0)) > sd_->converge_) &&
		(iter < sd_->max_iters_)) {
		iter++;

		Dproposal = Dcurr; // copy over vals so get known disparities; important for each iteration because all used values are included in optimization calculation in order to include smoothness terms, even if no pairwise terms are included for known values, then results for unknown pixels only are collected, but need optimization to not think it's choosing an "illegal" known value swap that results in a high degree of smoothness, throwing off unknown pixel fuse results

		DisparityProposalSelectionUpdate_Smooth(iter - sd_->average_over_, &Dcurr, &Dproposal);

		// run optimization of current proposal versus new proposal
		Matrix<bool, Dynamic, 1> Dswap(sd_->num_pixels_[cid_out_], 1); // matrix of booleans for each pixel where, if true, the value from Dnew should replace the current value in Dcurr
		map<int, map<int, int>> label_mappings; // empty here
		op_->FuseProposals(&Dcurr, &Dproposal, &Dswap, iter, &energy, label_mappings);
		Dswap = (Dproposal.array() == 0).select(Matrix<bool, Dynamic, 1>::Constant(sd_->num_pixels_[cid_out_], false), Dswap);// sets Dswap coefficients to false for positions where Dproposal coefficents are 0 (the oobv used in sd_->BuildComparisonDisparityProposal()); do this before swapping Dnew into D
		Dswap = (sd_->known_depths_[cid_out_].array()).select(Matrix<bool, Dynamic, 1>::Constant(sd_->num_pixels_[cid_out_], false), Dswap);// sets Dswap coefficients to false for positions where known_depths_ coefficents are true; do this before swapping Dnew into D
		Dcurr = (Dswap.array()).select(Dproposal, Dcurr); // sets D coefficients to Dnew coefficient values for positions where Dswap coefficients are true
	}

	//sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable

	// update disparity and depth maps
	sd_->disparity_maps_[cid_out_] = Dcurr;
	sd_->UpdateDepthMapFromDisparityMap(cid_out_);
}

// reconstruct depth maps for cameras with IDs in cids
// orders reconstruction so that cameras are followed by those closest to them
// arg max_num_cams gives the maximum number of other cameras to use during reconstruction for any one camera's depth values
// if save_as_computed is true, saves mesh and masked image for each camera as soon as its depths have been computed
// if all_max, assumes all pixels in all segments should be assigned valid maximum disparities without optimization
void StereoReconstruction::ReconstructAll(std::vector<int> cids, Scene *scene, bool save_as_computed, bool all_max) {
	bool debug = false;
	
	for (std::vector<int>::iterator it = cids.begin(); it != cids.end(); ++it) { // reconstruct each camera as output depth map in turn
		int cid_out = (*it);

		//if (cid_out < 15) continue; // remove this line ***********************************************************************************************
		//if (cid_out < 41) continue; // remove this line ***********************************************************************************************

		if (GLOBAL_LOAD_COMPUTED_DISPARITY_MAPS) {
			//std::string mat_dm_name = scene_name_ + "_Dsynced_out_" + to_string(cid_out);
			std::string mat_dm_name = scene_name_ + "_Dresult_out_" + to_string(cid_out);
			LoadEigenMatrix(mat_dm_name, TYPE_FLOAT, sd_->depth_maps_[cid_out]);
			sd_->UpdateDisparityMapFromDepthMap(cid_out);
			//std::string mat_mask_name = scene_name_ + "_mask_out_" + to_string(cid_out); // would be useful to save this, but need to recompute masks anyway because need to recompute BuildValidDisparityRanges() anyway to get unknowns correct, masks correct, and valid ranges correct
			//LoadEigenMatrix(mat_mask_name, TYPE_FLOAT, sd_->masks_[cid_out]);
			//sd_->DilateMask(cid_out);
			//sd_->InitPixelData(cid_out); // don't do this - don't want to recompute unknowns now that there are none - use the same unknowns as before
			if (debug) DisplayImages::DisplayGrayscaleImage(&sd_->depth_maps_[cid_out], sd_->heights_[cid_out], sd_->widths_[cid_out], sd_->orientations_[cid_out_]);
		}
		else {
			Stereo(cid_out, all_max);
			//std::string mat_dm_name = scene_name_ + "\\" + scene_name_ + "_Dresult_out_" + to_string(cid_out);
			//SaveEigenMatrix(mat_dm_name, TYPE_FLOAT, sd_->depth_maps_[cid_out]);
			//DisplayImages::SaveGrayscaleImage(&sd_->depth_maps_[cid_out], sd_->heights_[cid_out_], sd_->widths_[cid_out_], mat_dm_name);
			//SaveEigenMatrix(mat_mask_name, TYPE_FLOAT, sd_->masks_[cid_out]);
		}

		if ((debug) &&
			(!GLOBAL_LOAD_COMPUTED_DISPARITY_MAPS)) {

			Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out], 1);
			bool passes = sd_->TestDisparitiesAgainstMasks(cid_out, &sd_->disparity_maps_[cid_out], &pass);
			if (!passes) {
				cout << "StereoReconstruction::DisparityProposalSelectionUpdate_SmoothStar() TestDisparitiesAgainstMasks failed" << endl;
				DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out], sd_->widths_[cid_out], sd_->orientations_[cid_out_]);
			}
			else cout << "StereoReconstruction::DisparityProposalSelectionUpdate_SmoothStar() TestDisparitiesAgainstMasks passed" << endl;
			DisplayImages::DisplayGrayscaleImage(&sd_->depth_maps_[cid_out], sd_->heights_[cid_out], sd_->widths_[cid_out], sd_->orientations_[cid_out_]);
		}

		if ((save_as_computed) &&
			(!GLOBAL_LOAD_COMPUTED_DISPARITY_MAPS)) {
			string mesh_name = "mesh";
			sd_->SaveMeshes(scene_name_, cid_out);
			scene->cameras_[cid_out]->dm_->depth_map_ = sd_->depth_maps_[cid_out];
			scene->UpdateDisparities();
			scene->cameras_[cid_out]->SaveMaskedImage(scene->name_); // ensure depth map is up-to-date before saving this out
		}
	}
}

void StereoReconstruction::Stereo_optim_singlecompare(Matrix<double, Dynamic, 1> *Dcurr, Matrix<double, Dynamic, 1> *Dproposal, Matrix<double, Dynamic, 1> *Dnew) {
	cout << "StereoReconstruction::Stereo_optim_singlecompare()" << endl;

	Matrix<double, Dynamic, 1> energy(sd_->max_iters_, 1);
	energy.setZero();
	sd_->map_.setZero();
	sd_->count_updated_vals_.setZero();
	sd_->count_unlabelled_vals_.setZero();
	sd_->count_unlabelled_regions_.setZero();
	sd_->count_unlabelled_after_QPBOP_.setZero();
	sd_->timings_data_term_eval_.setZero();
	sd_->timings_smoothness_term_eval_.setZero();
	sd_->timings_qpbo_fuse_time_.setZero();
	sd_->timings_iteration_time_.setZero();
	int iter = 0;

	// initialize energies of phantom iterations to the maximum value. Phantom iterations are front-padded results to ensure can always average over the correct number of iterations; making them the maximum value ensures convergence is not achieved until enough iterations have been performed that these fall off the grid, so to speak. Another solution would be to simply run the loop for a minimum number of iterations before checking for convergence.
	for (int r = 0; r < sd_->average_over_; r++) {
		energy(r, 0) = DBL_MAX; // for float, would be FLT_MAX
	}

	// run optimization of current proposal versus new proposal
	Matrix<bool, Dynamic, 1> Dswap(sd_->num_pixels_[cid_out_], 1); // matrix of booleans for each pixel where, if true, the value from Dnew should replace the current value in Dcurr
	map<int, map<int, int>> label_mappings; // empty here
	op_->FuseProposals(Dcurr, Dproposal, &Dswap, iter, &energy, label_mappings);
	(*Dnew) = (*Dcurr);
	(*Dnew) = (Dswap.array()).select((*Dproposal), (*Dnew)); // sets Dnew coefficients to Dproposal coefficient values for positions where Dswap coefficients are true
	//sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew);
}

void StereoReconstruction::ImproveKnowns() {
	bool debug = true;

	// save unknown mask for reinstating pixel data afterward
	Matrix<bool, Dynamic, 1> mask_unknowns_tmp = sd_->masks_unknowns_[cid_out_];

	// set pixel data to all masked-in are unknowns
	sd_->SpecifyPixelData(cid_out_, &sd_->masks_[cid_out_]);

	// optimize known values by running QPBO on current knowns against validmax proposal
	sd_->ClearStatistics(cid_out_);
	//sd_->BuildValidDisparityRanges(cid_out_);

	
	if (debug) {
		std::string mat_name = scene_name_ + "_dispvalid_all_" + to_string(cid_out_);
		//SaveEigenMatrix(mat_name, TYPE_DOUBLE, sd_->unknown_disps_valid_[cid_out_]);
		LoadEigenMatrix(mat_name, TYPE_DOUBLE, sd_->unknown_disps_valid_[cid_out_]);
	}
	else sd_->BuildValidDisparityRanges(cid_out_);
	

	op_->Init(sd_, cid_out_);
	Matrix<double, Dynamic, 1> Dnew(sd_->num_pixels_[cid_out_], 1);
	Matrix<double, Dynamic, 1> D_validmax(sd_->num_pixels_[cid_out_], 1);
	InitValidMaximumProposal(&D_validmax);
	DisplayImages::DisplayGrayscaleImage(&D_validmax, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	Matrix<double, Dynamic, 1> Dcurr = sd_->disparity_maps_[cid_out_];
	Stereo_optim_singlecompare(&Dcurr, &D_validmax, &Dnew);

	// restore pixel data
	sd_->SpecifyPixelData(cid_out_, &mask_unknowns_tmp);
	//sd_->BuildValidDisparityRanges(cid_out_);

	
	if (debug) {
		std::string mat_name = scene_name_ + "_dispvalid_unk_" + to_string(cid_out_);
		//SaveEigenMatrix(mat_name, TYPE_DOUBLE, sd_->unknown_disps_valid_[cid_out_]);
		LoadEigenMatrix(mat_name, TYPE_DOUBLE, sd_->unknown_disps_valid_[cid_out_]);
	}
	else sd_->BuildValidDisparityRanges(cid_out_);
	

	// update known disparities in disparity and depth maps
	if (debug) {
		DisplayImages::DisplayGrayscaleImage(&Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		DisplayImages::DisplayGrayscaleImage(&sd_->disparity_maps_[cid_out_], sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		DisplayImages::DisplayGrayscaleImage(&sd_->known_depths_[cid_out_], sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
	EigenMatlab::AssignByBooleans(&sd_->disparity_maps_[cid_out_], &sd_->known_depths_[cid_out_], &Dnew);
	sd_->UpdateDepthMapFromDisparityMap(cid_out_);

	if (debug) {
		cout << "StereoReconstruction::ImproveKnowns() updated disparity map for cid_out_" << endl;
		DisplayImages::DisplayGrayscaleImage(&sd_->disparity_maps_[cid_out_], sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// updates depth map for camera cid_out in sd_->depth_maps_
// images is map of camera ID => BGR image
// height and width are pixel sizes of output display
// RTins is map of camera ID => camera extrinsics matrix
// analysis to utilize only those cameras whose IDs are given in sd_->use_cids_; list must include output camera ID
// if all_max, assumes all pixels in all segments should be assigned valid maximum disparities without optimization
void StereoReconstruction::Stereo(int cid_out, bool all_max) {
	bool save_segplns = false;
	bool load_segplns = false;

	cid_out_ = cid_out;

	//ImproveKnowns();

	sd_->ClearStatistics(cid_out_);
	sd_->BuildValidDisparityRanges(cid_out_);

	// Initialize valid maximums proposal, which does not require stereo optimization (use valid disp range valid minimums, where one exists, as additional disparity proposal for smoothstar phase with valid randoms where no valid range is available); leverages mask datta
	Matrix<double, Dynamic, 1> D_validmax(sd_->num_pixels_[cid_out_], 1);
	InitValidMaximumProposal(&D_validmax);

	if (all_max) {
		// update sd_->disparity_maps_[cid_out_] and sd_->depth_maps_[cid_out_] with result, and mask out segmentation occlusion edges
		sd_->disparity_maps_[cid_out_] = D_validmax;
		sd_->UpdateDepthMapFromDisparityMap(cid_out_);
		// update stereo reconstruction flag for the camera
		sd_->stereo_computed_[cid_out_] = true;
		return;
	}

	sd_->BuildCrowdDisparityProposal(cid_out_);
	op_->Init(sd_, cid_out);

	// set up visualizations of intermediate optimization data, if desired
	if (GLOBAL_OPTIMIZATION_DEBUG_VIEW_PLOTS) op_->InitOutputFigures();

	/*
	// for testing
	Matrix<double, Dynamic, 1> D_sameuni(sd_->num_pixels_[cid_out_], 1);
	Stereo_optim_sameuni(&D_sameuni);
	DisplayImages::DisplayGrayscaleImage(&D_sameuni, sd_->heights_[cid_out_], sd_->widths_[cid_out_]);
	*/


	Matrix<double, Dynamic, 1> D_segpln(sd_->num_pixels_[cid_out_], 1);
	/*
	// Initialize planar segmentation proposal
	if (load_segplns) {
	std::string mat_name = scene_name_ + "_D_segplns_" + to_string(cid_out_);
	sd_->D_segpln_.resize(sd_->num_unknown_pixels_[cid_out_], 1);
	LoadEigenMatrix(mat_name, TYPE_FLOAT, sd_->D_segpln_);
	}
	else {
	seg_->Init(sd_);
	seg_->SegmentPlanar(cid_out_);
	if (save_segplns) {
	std::string mat_name = scene_name_ + "_D_segplns_" + to_string(cid_out_);
	SaveEigenMatrix(mat_name, TYPE_FLOAT, sd_->D_segpln_);
	}
	}
	InitSegPlnProposal(&D_segpln);
	*/

	// Initialize crowd-sourced proposal, which does not require stereo optimization; leverages known pixels in other camera views (are not known in this camera view because there could be occlusions)
	Matrix<double, Dynamic, 1> D_crowdsourced(sd_->num_pixels_[cid_out_], 1);
	InitCrowdSourcedProposal(&D_crowdsourced);

	// find segment label mappings from cid_out_ to each of the input images where possible
	map<int, map<int, int>> label_mappings; // for cid_out_, map of cid_in => label_out => label_in
	for (vector<int>::iterator it = sd_->use_cids_[cid_out_].begin(); it != sd_->use_cids_[cid_out_].end(); ++it) {
		int cid_in = (*it);
		if (cid_in == cid_out_) continue;
		map<int, int> mapping = sd_->MapSegmentationLabelsAcrossImages(cid_out_, cid_in, &D_crowdsourced);
		label_mappings[cid_in] = mapping;
	}

	/*
	Matrix<double, Dynamic, 1> D_percuni(num_pixels, 1);
	if (debug_load_from_file_percuni) {
	std::string mat_name = scene_name_ + "_D_percuni_" + to_string(cid_out_);
	LoadEigenMatrix(mat_name, TYPE_DOUBLE, D_percuni);
	}
	else {
	Stereo_optim(PERC_UNI, &D_percuni);
	if (debug_save_to_file_percuni) {
	std::string mat_name = scene_name_ + "_D_percuni_" + to_string(cid_out_);
	SaveEigenMatrix(mat_name, TYPE_DOUBLE, D_percuni);
	}
	}
	*/

	// Smooth*
	Matrix<double, Dynamic, 1> Dproposal(sd_->num_pixels_[cid_out_], 1);
	//Stereo_optim_old(SMOOTH_STAR, &Dproposal, &D_crowdsourced, &D_validmax, &D_segpln);
	Stereo_optim(&Dproposal, &D_crowdsourced, &D_validmax, label_mappings);

	// clean, smooth, and snap results
	//sd_->SmoothDisparityMap(cid_out_, &Dproposal, GLOBAL_SMOOTH_KERNEL_SIZE, GLOBAL_SMOOTH_ITERS); // reinstate? ************************************************
	//sd_->SnapDisparitiesToValidRanges(cid_out_, &Dproposal); // reinstate? ************************************************
	CleanProposalSegments(&Dproposal);
	sd_->SmoothDisparityMap(cid_out_, &Dproposal, GLOBAL_SMOOTH_KERNEL_SIZE, GLOBAL_SMOOTH_ITERS); // comment out? ************************************************
	sd_->SnapDisparitiesToValidRanges(cid_out_, &Dproposal);

	// clean up windows of visualization of intermediate optimization data, if created
	if (GLOBAL_OPTIMIZATION_DEBUG_VIEW_PLOTS) op_->CloseOutputFigures();

	// update sd_->disparity_maps_[cid_out_] and sd_->depth_maps_[cid_out_] with result, and mask out segmentation occlusion edges
	sd_->disparity_maps_[cid_out_] = Dproposal;
	sd_->UpdateDepthMapFromDisparityMap(cid_out_);
	//sd_->MaskOutSegmentationOcclusionEdges(cid_out_); // don't trust depth data on pixels with segmentation value 0 (on the lines) unless all pixels in neighborhood are within reasonable depth distance of each other (ensuring it represents a shared edge and not an occlusion edge), so mask out untrusted pixels

	// update stereo reconstruction flag for the camera
	sd_->stereo_computed_[cid_out_] = true;
}

// temporary
void StereoReconstruction::InitRandomProposal(Matrix<double, Dynamic, 1> *D) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	double *pD = D->data();
	bool *pM = sd_->masks_[cid_out_].data();
	double disp_val;
	int unk_pix_idx;
	int idx = 0;
	for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
		for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
			if (!*pM++) { // masked out
				*pD++ = 0.;
				idx++;
				continue;
			}
			unk_pix_idx = sd_->unknown_maps_fwd_[cid_out_](idx, 0);

			if (unk_pix_idx == -1) { // will occur with known depths when don't trust the data and are treating the pixel depth as unknown despite having a value for it
				disp_val = sd_->min_disps_[cid_out_] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_])); // generates a random float in the range[sd_->min_disp_, sd_->max_disp_]; this approach may assign disparity values that do not conform to the masks of one or more other input images
			}
			else {
				float disp_perc = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // generates a random double in the range[0, 1]
				disp_perc = 1 - disp_perc * GLOBAL_RANDDISP_MAXPERCDEPTH;
				disp_val = static_cast<double>(sd_->GenerateValidDisparityAtPerc(cid_out_, unk_pix_idx, disp_perc)); // if possible, generates disparity values that safely conform to the masks of all other input images; if not possible because no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
			}
			*pD++ = disp_val;
			idx++;
		}
	}

	// truncate disparities to allowed range
	(*D) = D->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	(*D) = D->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, D, &pass);
		if (!passes) {
			cout << "StereoReconstruction::InitRandomProposal() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::InitRandomProposal() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// pixels in imgMasks_color_[cid] that are (255, 0, 0) signify that we should assign the max valid disparity to the pixel and treat it as known; we do this by updating both proposal this way rather than changing the knowns list for the camera because the valid ranges are not available earlier when the knowns lists is set (and valid ranges in their current form are determined for unknown pixels anyway)
void StereoReconstruction::UpdateProposalsFromColorMask(Matrix<double, Dynamic, 1> *Dcurr, Matrix<double, Dynamic, 1> *Dnew) {
	bool debug = false;

	if (debug) {
		cout << "before UpdateProposalsFromColorMask()" << endl;
		cout << "D" << endl;
		DisplayImages::DisplayGrayscaleImage(Dcurr, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		cout << "Dnew" << endl;
		DisplayImages::DisplayGrayscaleImage(Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}

	if (debug) display_mat(&sd_->imgMasks_color_[cid_out_], "imgMasks_color_", sd_->orientations_[cid_out_]);

	Vec3b *pMc;
	Vec3b color;
	int idx_full, unk_pix_idx;
	double disp_val;
	for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
		pMc = sd_->imgMasks_color_[cid_out_].ptr<Vec3b>(r);
		for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
			color = pMc[c];
			if ((color[0] > GLOBAL_MAX_MASKSEG_LINEVAL) &&
				(color[1] < GLOBAL_MIN_MASKSEG_LINEVAL) &&
				(color[2] < GLOBAL_MIN_MASKSEG_LINEVAL)) {
				idx_full = PixIndexFwdCM(Point(c, r), sd_->heights_[cid_out_]);
				unk_pix_idx = sd_->unknown_maps_fwd_[cid_out_](idx_full, 0);
				disp_val = static_cast<double>(sd_->GetMaxValidDisparity(cid_out_, unk_pix_idx));
				(*Dcurr)(idx_full, 0) = disp_val;
				(*Dnew)(idx_full, 0) = disp_val;
			}
		}
	}

	if (debug) {
		cout << "after UpdateProposalsFromColorMask()" << endl;
		cout << "D" << endl;
		DisplayImages::DisplayGrayscaleImage(Dcurr, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		cout << "Dnew" << endl;
		DisplayImages::DisplayGrayscaleImage(Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// temporary
void StereoReconstruction::DisparityProposalSelectionUpdate_SameUni(Eigen::Matrix<double, Dynamic, 1> *Dnew) {
	assert(Dnew->rows() == sd_->num_pixels_[cid_out_]);

	// random fronto-parallel - a random number is chosen and assigned to all pixels, so they all get the same value
	bool debug = false;
	if (debug) cout << endl << "StereoReconstruction::DisparityProposalSelectionUpdate_SameUni()" << endl;
	double disp_val = sd_->min_disps_[cid_out_] + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX) / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_])); // generates a random double in the range[sd_->min_disp_, sd_->max_disp_]
	EigenMatlab::AssignByBooleans(Dnew, &sd_->masks_unknowns_[cid_out_], disp_val);
	sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew);

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, Dnew, &pass);
		if (!passes) {
			cout << "StereoReconstruction::InitDisparityProposal() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::InitDisparityProposal() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// temporary
void StereoReconstruction::Stereo_optim_sameuni(Matrix<double, Dynamic, 1> *Dproposal) {
	assert(Dproposal->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = true;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	cout << "StereoReconstruction::Stereo_optim_sameuni()" << endl;

	int num_pixels = sd_->num_pixels_[cid_out_];
	int num_pixels_used = sd_->num_used_pixels_[cid_out_];

	// create initial arrays

	Matrix<double, Dynamic, 1> energy(sd_->max_iters_, 1);
	energy.setZero();
	sd_->map_.setZero();
	sd_->count_updated_vals_.setZero();
	sd_->count_unlabelled_vals_.setZero();
	sd_->count_unlabelled_regions_.setZero();
	sd_->count_unlabelled_after_QPBOP_.setZero();
	sd_->timings_data_term_eval_.setZero();
	sd_->timings_smoothness_term_eval_.setZero();
	sd_->timings_qpbo_fuse_time_.setZero();
	sd_->timings_iteration_time_.setZero();


	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	Eigen::Matrix<double, Dynamic, 1> D(num_pixels, 1);
	InitRandomProposal(&D);

	int iter = sd_->average_over_; // starting value for iter (not +1 because 0-indexed instead of 1-indexed like Matlab)

	// initialize energies of phantom iterations to the maximum value. Phantom iterations are front-padded results to ensure can always average over the correct number of iterations; making them the maximum value ensures convergence is not achieved until enough iterations have been performed that these fall off the grid, so to speak. Another solution would be to simply run the loop for a minimum number of iterations before checking for convergence.
	for (int r = 0; r < sd_->average_over_; r++) {
		energy(r, 0) = DBL_MAX; // for float, would be FLT_MAX
	}
	energy(iter, 0) = DBL_MAX / 1e20; // ensures initial while check for loop entry does not meet convergence test by setting current iteration's energy significantly lower than earlier energies, so stabilization has not yet been achieved.

	// the new (proposal) depth map
	Eigen::Matrix<double, Dynamic, 1> Dnew(num_pixels, 1);

	// for the loop condition: note that energy is guaranteed not to increase, so convergence is checked by determining ratio of current energy to energy average_over iterations ago, then comparing to a convergence threshold that is dependent on the number of iterations over which we are averaging convergence, yielding a check that is truly an "averaging" of convergence.  More generally, if energy is falling, loop will continue.  Once has stabilized sufficiently, or once we exceed the maximum number of iterations, the loop will terminate.
	double t_loop;
	while ((1 - (energy(iter, 0) / energy(iter - sd_->average_over_, 0)) > sd_->converge_) &&
		(iter < sd_->max_iters_)) {
		t_loop = (double)getTickCount();
		iter++;

		Dnew = D; // copy over vals so get known disparities; important for each iteration because all used values are included in optimization calculation in order to include smoothness terms, even if no pairwise terms are included for known values, then results for unknown pixels only are collected, but need optimization to not think it's choosing an "illegal" known value swap that results in a high degree of smoothness, throwing off unknown pixel fuse results

		DisparityProposalSelectionUpdate_SameUni(&Dnew);

		sd_->timings_data_term_eval_(0, iter) = (double)getTickCount() - t_loop; // time taken for proposal generation

		if (debug) {
			DisplayImages::DisplayGrayscaleImage(&D, sd_->heights_[cid_out_], sd_->widths_[cid_out_]);
			DisplayImages::DisplayGrayscaleImage(&Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_]);
		}

		//LoadEigenMatrixBasic("MatlabD1", D);
		//LoadEigenMatrixBasic("MatlabD2", Dnew);

		/*
		if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[sd_->cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(sd_->cid_out_, &Dnew, &pass);
		if (!passes) {
		cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks failed for Dnew" << endl;
		}
		else cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks passed for Dnew" << endl;
		}
		*/

		// Fuse the depths
		Eigen::Matrix<bool, Dynamic, 1> Dswap(num_pixels, 1); // matrix of booleans for each pixel where, if true, the value from Dnew should replace the current value in Dcurr
		map<int, map<int, int>> label_mappings; // empty here
		op_->FuseProposals(&D, &Dnew, &Dswap, iter, &energy, label_mappings, true);
		Dswap = (sd_->known_depths_[cid_out_].array()).select(Matrix<bool, Dynamic, 1>::Constant(num_pixels, false), Dswap);// sets Dswap coefficients to false for positions where known_depths_ coefficents are true; do this before swapping Dnew into D
		D = (Dswap.array()).select(Dnew, D); // sets D coefficients to Dnew coefficient values for positions where Dswap coefficients are true
		Eigen::Matrix<unsigned short, Dynamic, 1> iter_mat(num_pixels, 1);
		iter_mat.setConstant(iter);
		sd_->map_ = (Dswap.array()).select(iter_mat, sd_->map_); // sets maps_ coefficients to iter value for positions where Dswap coefficients are true
		iter_mat.resize(0, 1);

		cout << "Energy for this iteration: " << energy(iter, 0) << "; energy for energy(iter - sd_->average_over_, 0) " << energy(iter - sd_->average_over_, 0) << "; convergence occurs at " << sd_->converge_ << endl;
	}

	(*Dproposal) = D;

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, Dproposal, &pass);
		if (!passes) {
			cout << "StereoReconstruction::Stereo_optim_sameuni() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_]);
		}
		else cout << "StereoReconstruction::Stereo_optim_sameuni() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(Dproposal, sd_->heights_[cid_out_], sd_->widths_[cid_out_]);
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Optimization::Stereo_optim_sameuni() running time = " << t*1000. / getTickFrequency() << " ms" << endl << endl;
	}
}


// performs depth test on depth_map values of all pixels of label for which label_flags is true and updates pass with the result by pixel
// depth_map holds depth map values for pixels of the current segment label being investigated (others are ignored)
// label_flags indicates whether pixel belongs to the current segment label being investigated
// pass indicates whether pixel passes or fails the depth test against neighboring pixels of the same segment
void StereoReconstruction::ConductDepthTest(Matrix<double, Dynamic, Dynamic> *depth_map, Matrix<bool, Dynamic, Dynamic> *label_flags, Matrix<bool, Dynamic, Dynamic> *pass) {
	pass->setConstant(false);
	double depth;
	double depth_threshold = GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT / sd_->agisoft_to_world_scales_[cid_out_];
	for (int c = 1; c < ((*depth_map).cols() - 1); c++) {
		for (int r = 1; r < ((*depth_map).rows() - 1); r++) {
			if (!(*label_flags)(r, c)) continue;
			depth = (*depth_map)(r, c);
			bool pixpass = true;
			for (int i = -1; i <= 1; i += 2) {
				for (int j = -1; j <= 1; j += 2) {
					if ((abs(depth - (*depth_map)(r + j, c + i)) > depth_threshold) &&
						((*label_flags)(r + j, c + i)))
						pixpass = false;
				}
			}
			if (pixpass) (*pass)(r, c) = true;
		}
	}
}

// ensures neighboring pixels belonging to the same segment are within GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT WS units of each other in depth
// for each segment, zero out pixel depths that fail the test, partition the result into blobs, find the largest blob, zero out all others, then call sd_->SmoothDisparityMapFromTrusted() with wide kernel on the result to find better values for the zeroed out pixels - holds base label pixels as constant and uses them as the only values in the smoothing computation
void StereoReconstruction::CleanProposalSegments(Matrix<double, Dynamic, 1> *D) {
	bool debug = false;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	//if (debug) DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);

	Matrix<double, Dynamic, Dynamic> disp_map = (*D);
	disp_map.resize(sd_->heights_[cid_out_], sd_->widths_[cid_out_]);

	Matrix<double, Dynamic, Dynamic> dm = DepthMap::ConvertDisparityMapToDepthMap(D);
	dm.resize(sd_->heights_[cid_out_], sd_->widths_[cid_out_]);

	// find all segment labels
	map<unsigned int, int> label_counts = GetLabelCounts(&sd_->segs_[cid_out_]);

	Matrix<double, Dynamic, Dynamic> dm_label(sd_->heights_[cid_out_], sd_->widths_[cid_out_]); // depth map values from dm present for pixels of the current segment label being investigated, and 0. for all others
	Matrix<bool, Dynamic, Dynamic> label_flags(sd_->heights_[cid_out_], sd_->widths_[cid_out_]); // bool indicates whether pixel belongs to the current segment label being investigated
	Matrix<bool, Dynamic, Dynamic> pass(sd_->heights_[cid_out_], sd_->widths_[cid_out_]); // bool indicates whether pixel passes or fails the depth test against neighboring pixels of the same segment
	Matrix<bool, Dynamic, Dynamic> label_base_flags(sd_->heights_[cid_out_], sd_->widths_[cid_out_]); // bool indicates whether pixel belongs to the current segment label_base being investigated
	Matrix<bool, Dynamic, Dynamic> inpaintMask(sd_->heights_[cid_out_], sd_->widths_[cid_out_]); // bool indicates whether pixel's disparity should be inpainted in the current step
	for (map<unsigned int, int>::iterator it = label_counts.begin(); it != label_counts.end(); ++it) {
		unsigned int label = (*it).first;
		if (label == 0) continue;
	
		dm_label.setZero();
		label_flags.setConstant(false);

		label_flags = (sd_->segs_[cid_out_].array() == label).select(Matrix<bool, Dynamic, Dynamic>::Constant(sd_->heights_[cid_out_], sd_->widths_[cid_out_], true), label_flags);
		EigenMatlab::AssignByBooleans(&dm_label, &label_flags, &dm);
		ConductDepthTest(&dm_label, &label_flags, &pass); // identify depth discontinuities in the current segment

		if (debug) {
			cout << "label_flags" << endl;
			DisplayImages::DisplayGrayscaleImage(&label_flags, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
			cout << "dm_label" << endl;
			DisplayImages::DisplayGrayscaleImage(&dm_label, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
			cout << "pass" << endl; 
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}

		// partition the result into blobs by first zeroing out depths at discontinuities
		Mat passcv = Mat::zeros(dm_label.rows(), dm_label.cols(), CV_8UC1);
		EigenOpenCV::eigen2cv(pass, passcv);
		Mat mask = Mat::zeros(passcv.size(), CV_8UC1);
		cv::threshold(passcv, mask, 0, 255, THRESH_BINARY); // just to make sure eigen2cv didn't assign a value of 1 for true instead of 255
		map<unsigned int, int> seg_label_counts;
		Matrix<unsigned int, Dynamic, Dynamic> seg = EigenOpenCV::SegmentUsingBlobs(mask, seg_label_counts); // segment now that depth discontinuities have been zeroed, creating the breaks between the blobs for segmentation

		if (debug) {
			cout << "seg" << endl;
			Matrix<unsigned int, Dynamic, Dynamic> tmp1 = seg;
			tmp1.resize(passcv.rows*passcv.cols, 1);
			Matrix<unsigned int, Dynamic, 1> tmp2 = tmp1;
			DisplayImages::DisplaySegmentedImage(&tmp2, passcv.rows, passcv.cols, sd_->orientations_[cid_out_]);
		}

		// find the largest blob
		unsigned int label_base = -1;
		int largest_count = -1;
		for (map<unsigned int, int>::iterator it = seg_label_counts.begin(); it != seg_label_counts.end(); ++it) {
			unsigned int label = (*it).first;
			if (label == 0) continue;
			int count = (*it).second;
			if (count > largest_count) {
				label_base = label;
				largest_count = count;
			}
		}
		if (label_base == -1) {
			if (debug) cout << "no base label within this segment from which to propogate disparity values" << endl;
			continue; // no base label within this segment from which to propogate disparity values
		}
		label_base_flags = seg.array() == label_base; // bool indicates whether pixel belongs to the current segment label_base being investigated

		// zero out disparities in D for pixels in this segment that aren't part of the largest blob since the disparities are discontinuous with the largest blob
		inpaintMask.setConstant(true);
		inpaintMask = (label_flags.array() == false).select(Matrix<bool, Dynamic, Dynamic>::Constant(sd_->heights_[cid_out_], sd_->widths_[cid_out_], false), inpaintMask);
		inpaintMask = (label_base_flags.array() == true).select(Matrix<bool, Dynamic, Dynamic>::Constant(sd_->heights_[cid_out_], sd_->widths_[cid_out_], false), inpaintMask);
		disp_map = (inpaintMask.array() == true).select(Matrix<double, Dynamic, Dynamic>::Zero(sd_->heights_[cid_out_], sd_->widths_[cid_out_]), disp_map);
		//disp_map = (label_flags.array() == true && label_base_flags.array() == false).select(Matrix<double, Dynamic, Dynamic>::Zero(sd_->heights_[cid_out_], sd_->widths_[cid_out_]), disp_map);
		if (debug) {
			cout << "label_flags" << endl;
			DisplayImages::DisplayGrayscaleImage(&label_flags, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
			cout << "label_base_flags" << endl;
			DisplayImages::DisplayGrayscaleImage(&label_base_flags, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
			cout << "disparity map to update with fill" << endl;
			DisplayImages::DisplayGrayscaleImage(&disp_map, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}

		// fill the zeros in the segment back in based on data from the largest blob area in the segment; since fill algorithm doesn’t advanc e around corners, function returns T/F whether a disparity was updated from zero and keep running it in a loop on the same segment until it returns false
		bool made_change = true;
		while (made_change)
			made_change = sd_->FillSegmentDisparityZeros(cid_out_, label, &disp_map);

		if (debug) {
			cout << "disparity map updated with fill" << endl;
			DisplayImages::DisplayGrayscaleImage(&disp_map, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
	}

	disp_map.resize(sd_->heights_[cid_out_] * sd_->widths_[cid_out_], 1);
	(*D) = disp_map;
	disp_map.resize(sd_->heights_[cid_out_], sd_->widths_[cid_out_]);
	
	if (debug) DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);

	//sd_->SnapDisparitiesToValidRanges(cid_out_, D); // don't snap it here because valid ranges cannot always be trusted since camera poses are not entirely accurate - wait to snap later instead
	// truncate disparities to allowed range
	//(*D) = D->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	//(*D) = D->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max                   
	//if (debug) cout << "snapped and truncated version" << endl;
	//if (debug) DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Optimization::CleanProposalSegments() running time = " << t*1000. / getTickFrequency() << " ms" << endl << endl;
	}
}

// use valid disp range valid maximums, where one exists, as additional disparity proposal for smoothstar phase with valid randoms where no valid range is available
void StereoReconstruction::InitSegPlnProposal(Matrix<double, Dynamic, 1> *D) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);
	bool debug = false;
	
	if (debug) cout << endl << "StereoReconstruction::InitSegPlnProposal()" << endl;
	InitDisparityProposal(D);
	Matrix<double, Dynamic, 1> sgpln = sd_->D_segpln_.cast<double>();
	EigenMatlab::AssignByTruncatedBooleans(D, &sd_->masks_unknowns_[cid_out_], &sgpln); // sd_->D_segpln_ is size num_unknown_pixels_out_ x map size
	sd_->SnapDisparitiesToValidRanges(cid_out_, D);

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, D, &pass);
		if (!passes) {
			cout << "StereoReconstruction::InitValidMaximumProposal() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::InitValidMaximumProposal() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// use valid disp range valid maximums, where one exists, as additional disparity proposal for smoothstar phase with valid randoms where no valid range is available
void StereoReconstruction::InitValidMaximumProposal(Matrix<double, Dynamic, 1> *D) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	float *pDM = sd_->depth_maps_[cid_out_].data();
	double *pD = D->data();
	double *pCD = sd_->crowd_disparity_proposal_[cid_out_].data();
	bool *pM = sd_->masks_[cid_out_].data();
	bool *pU = sd_->masks_unknowns_[cid_out_].data();
	double disp_val;
	bool unknown;
	int idx_full = 0;
	int unk_pix_idx;
	for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
		for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
			if (!*pM++) { // masked out
				*pD++ = 0.;
				pU++;
				pDM++;
				idx_full++;
				continue;
			}
			unknown = true;
			if ((!*pU) &&
				(*pDM != 0.)) { // depth is known
				disp_val = 1. / static_cast<double>(*pDM);
				unknown = false;
			}
			else disp_val = 0.;
			if (unknown) { // depth is unknown
				unk_pix_idx = sd_->unknown_maps_fwd_[cid_out_](idx_full, 0);
				disp_val = static_cast<double>(sd_->GetMaxValidDisparity(cid_out_, unk_pix_idx));
				if (disp_val == 0)
					disp_val = static_cast<double>(sd_->GenerateRandomValidDisparity(cid_out_, unk_pix_idx)); // generates disparity values that safely conform to the masks of all other input images
				if (disp_val == 0) { // occurs when no valid disparity range exists for the unknown pixel
					float disp_perc = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // generates a random double in the range[0, 1]
					disp_perc = 1 - disp_perc * GLOBAL_RANDDISP_MAXPERCDEPTH;
					disp_val = static_cast<double>(sd_->GenerateValidDisparityAtPerc(cid_out_, unk_pix_idx, disp_perc)); // if possible, generates disparity values that safely conform to the masks of all other input images; if not possible because no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
				}
			}
			*pD++ = disp_val;
			pU++;
			pDM++;
			idx_full++;
		}
	}

	// truncate disparities to allowed range
	(*D) = D->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	(*D) = D->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, D, &pass);
		if (!passes) {
			cout << "StereoReconstruction::InitValidMaximumProposal() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::InitValidMaximumProposal() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// use valid disp range valid maximums, where one exists, as additional disparity proposal for smoothstar phase with valid randoms where no valid range is available
void StereoReconstruction::InitValidMinimumProposal(Matrix<double, Dynamic, 1> *D) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	float *pDM = sd_->depth_maps_[cid_out_].data();
	double *pD = D->data();
	double *pCD = sd_->crowd_disparity_proposal_[cid_out_].data();
	bool *pM = sd_->masks_[cid_out_].data();
	bool *pU = sd_->masks_unknowns_[cid_out_].data();
	double disp_val;
	bool unknown;
	int idx_full = 0;
	int unk_pix_idx;
	for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
		for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
			if (!*pM++) { // masked out
				*pD++ = 0.;
				pU++;
				pDM++;
				idx_full++;
				continue;
			}
			unknown = true;
			if ((!*pU) &&
				(*pDM != 0.)) { // depth is known
				disp_val = 1. / static_cast<double>(*pDM);
				unknown = false;
			}
			else disp_val = 0.;
			if (unknown) { // depth is unknown
				unk_pix_idx = sd_->unknown_maps_fwd_[cid_out_](idx_full, 0);
				disp_val = static_cast<double>(sd_->GetMinValidDisparity(cid_out_, unk_pix_idx));
				if (disp_val == 0)
					disp_val = static_cast<double>(sd_->GenerateRandomValidDisparity(cid_out_, unk_pix_idx)); // generates disparity values that safely conform to the masks of all other input images
				if (disp_val == 0) { // occurs when no valid disparity range exists for the unknown pixel
					float disp_perc = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // generates a random double in the range[0, 1]
					disp_perc = 1 - disp_perc * GLOBAL_RANDDISP_MAXPERCDEPTH;
					disp_val = static_cast<double>(sd_->GenerateValidDisparityAtPerc(cid_out_, unk_pix_idx, disp_perc)); // if possible, generates disparity values that safely conform to the masks of all other input images; if not possible because no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
				}
			}
			*pD++ = disp_val;
			pU++;
			pDM++;
			idx_full++;
		}
	}

	// truncate disparities to allowed range
	(*D) = D->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	(*D) = D->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, D, &pass);
		if (!passes) {
			cout << "StereoReconstruction::InitValidMaximumProposal() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::InitValidMaximumProposal() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// initializes a reference image disparity proposal for StereoReconstruction::Stereo_optim() by utilizing depth information where already known, crowd-sourced depth information where not, and random values where neither are available
void StereoReconstruction::InitCrowdSourcedProposal(Matrix<double, Dynamic, 1> *D) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	int idx = 0;
	float *pDM = sd_->depth_maps_[cid_out_].data();
	double *pD = D->data();
	bool *pM = sd_->masks_[cid_out_].data();
	bool *pU = sd_->masks_unknowns_[cid_out_].data();
	double depth_val, disp_val;
	bool unknown;
	int unk_pix_idx;
	for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
		for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
			if (!*pM++) { // masked out
				*pD++ = 0.;
				pDM++;
				pU++;
				idx++;
				continue;
			}
			unknown = *pU;
			depth_val = static_cast<double>(*pDM++);
			if ((!*pU) &&
				(depth_val != 0.)) { // depth is known and have valid value for it
				disp_val = 1. / depth_val;
			}
			else {
				disp_val = 0.; // will occur with depths with values in them when don't trust the data and are treating the pixel depth as unknown despite having a value for it
				unknown = true;
			}
			if ((unknown) ||
				(disp_val == 0.)) { // depth is unknown or invalid for a masked-in pixel
				unk_pix_idx = sd_->unknown_maps_fwd_[cid_out_](idx, 0);

				if (unk_pix_idx == -1) {  // will occur with known depths when don't trust the data and are treating the pixel depth as unknown despite having a value for it
					disp_val = sd_->min_disps_[cid_out_] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_])); // generates a random float in the range[sd_->min_disp_, sd_->max_disp_]; this approach may assign disparity values that do not conform to the masks of one or more other input images
				}
				else {
					disp_val = sd_->crowd_disparity_proposal_[cid_out_](unk_pix_idx, 0);// *pCD++; // first try camera crowd-sourced disparity value
					if (disp_val == 0) {
						float disp_perc = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // generates a random double in the range[0, 1]
						disp_perc = 1 - disp_perc * GLOBAL_RANDDISP_MAXPERCDEPTH;
						disp_val = static_cast<double>(sd_->GenerateValidDisparityAtPerc(cid_out_, unk_pix_idx, disp_perc)); // if possible, generates disparity values that safely conform to the masks of all other input images; if not possible because no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
					}
				}
			}
			*pD++ = disp_val;
			pU++;
			idx++;
		}
	}

	// truncate disparities to allowed range
	(*D) = D->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	(*D) = D->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->heights_[cid_out_] * sd_->widths_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, D, &pass);
		if (!passes) {
			cout << "StereoReconstruction::InitCrowdSourcedProposal() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::InitCrowdSourcedProposal() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// initializes a reference image disparity proposal for StereoReconstruction::Stereo_optim() by utilizing depth information where already available and random values where not
void StereoReconstruction::InitDisparityProposal(Matrix<double, Dynamic, 1> *D) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	float *pDM = sd_->depth_maps_[cid_out_].data();
	double *pD = D->data();
	bool *pM = sd_->masks_[cid_out_].data();
	bool *pU = sd_->masks_unknowns_[cid_out_].data();
	double depth_val, disp_val;
	bool unknown;
	int unk_pix_idx;
	int idx = 0;
	for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
		for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
			if (!*pM++) { // masked out
				*pD++ = 0.;
				pDM++;
				pU++;
				idx++;
				continue;
			}
			unknown = *pU;
			depth_val = static_cast<double>(*pDM++);
			if ((!*pU) &&
				(depth_val != 0.)) { // depth is known and have valid value for it
				disp_val = 1. / depth_val;
			}
			else {
				disp_val = 0.; // will occur with depths with values in them when don't trust the data and are treating the pixel depth as unknown despite having a value for it
				unknown = true;
			}
			if ((unknown) ||
				(disp_val == 0.)) { // depth is unknown or invalid for a masked-in pixel
				unk_pix_idx = sd_->unknown_maps_fwd_[cid_out_](idx, 0);

				if (unk_pix_idx == -1) { // will occur with known depths when don't trust the data and are treating the pixel depth as unknown despite having a value for it
					disp_val = sd_->min_disps_[cid_out_] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_])); // generates a random float in the range[sd_->min_disp_, sd_->max_disp_]; this approach may assign disparity values that do not conform to the masks of one or more other input images
				}
				else {
					float disp_perc = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // generates a random double in the range[0, 1]
					disp_perc = 1 - disp_perc * GLOBAL_RANDDISP_MAXPERCDEPTH;
					disp_val = static_cast<double>(sd_->GenerateValidDisparityAtPerc(cid_out_, unk_pix_idx, disp_perc)); // if possible, generates disparity values that safely conform to the masks of all other input images; if not possible because no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
				}
			}
			*pD++ = disp_val;
			pU++;
			idx++;
		}
	}

	// truncate disparities to allowed range
	(*D) = D->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	(*D) = D->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, D, &pass);
		if (!passes) {
			cout << "StereoReconstruction::InitDisparityProposal() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::InitDisparityProposal() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// initializes a reference image disparity proposal for StereoReconstruction::Stereo_optim() by utilizing depth information where already available and random values where not
void StereoReconstruction::InitCurrentGuessProposal(Matrix<double, Dynamic, 1> *D) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	float *pDM = sd_->depth_maps_[cid_out_].data();
	double *pD = D->data();
	bool *pM = sd_->masks_[cid_out_].data();
	double depth_val, disp_val;
	bool unknown;
	int unk_pix_idx;
	int idx = 0;
	for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
		for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
			if (!*pM++) { // masked out
				*pD++ = 0.;
				pDM++;
				idx++;
				continue;
			}
			unknown = true;
			depth_val = static_cast<double>(*pDM++);
			if (depth_val != 0.) { // depth is known
				disp_val = 1. / depth_val;
				unknown = false;
			}
			else disp_val = 0.; // will occur with known depths when don't trust the data and are treating the pixel depth as unknown despite having a value for it
			if ((unknown) ||
				(disp_val == 0.)) { // depth is unknown or invalid for a masked-in pixel
				unk_pix_idx = sd_->unknown_maps_fwd_[cid_out_](idx, 0);

				if (unk_pix_idx == -1) { // will occur with known depths when don't trust the data and are treating the pixel depth as unknown despite having a value for it
					disp_val = sd_->min_disps_[cid_out_] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_])); // generates a random float in the range[sd_->min_disp_, sd_->max_disp_]; this approach may assign disparity values that do not conform to the masks of one or more other input images
				}
				else {
					//float disp_perc = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // generates a random double in the range[0, 1]
					//disp_perc = 1 - disp_perc * GLOBAL_RANDDISP_MAXPERCDEPTH;
					//disp_val = static_cast<double>(sd_->GenerateValidDisparityAtPerc(cid_out_, unk_pix_idx, disp_perc)); // if possible, generates disparity values that safely conform to the masks of all other input images; if not possible because no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
					disp_val = static_cast<double>(sd_->GenerateRandomValidDisparity(cid_out_, unk_pix_idx)); // if possible, generates disparity values that safely conform to the masks of all other input images; if not possible because no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
				}
			}
			*pD++ = disp_val;
			idx++;
		}
	}

	// truncate disparities to allowed range
	(*D) = D->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	(*D) = D->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max

	sd_->SnapDisparitiesToValidRanges(cid_out_, D);

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, D, &pass);
		if (!passes) {
			cout << "StereoReconstruction::InitDisparityProposal() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::InitDisparityProposal() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// updates Dproposal with the optimized disparity proposal
// D_sameuni and D_segpln only need relevant data when proposal_method is SMOOTH_STAR
void StereoReconstruction::Stereo_optim_old(const GLOBAL_PROPOSAL_METHOD proposal_method, Matrix<double, Dynamic, 1> *Dproposal, const Matrix<double, Dynamic, 1> *D_crowdsourced, const Matrix<double, Dynamic, 1> *D_validmax, const Matrix<double, Dynamic, 1> *D_segpln) {
	assert(Dproposal->rows() == sd_->num_pixels_[cid_out_]);
	
	bool debug = false;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	int num_pixels = sd_->num_pixels_[cid_out_];
	int num_pixels_used = sd_->num_used_pixels_[cid_out_];

	// create initial arrays
	
	Matrix<double, Dynamic, 1> energy(sd_->max_iters_, 1);
	energy.setZero();
	sd_->map_.setZero();
	sd_->count_updated_vals_.setZero();
	sd_->count_unlabelled_vals_.setZero();
	sd_->count_unlabelled_regions_.setZero();
	sd_->count_unlabelled_after_QPBOP_.setZero();
	sd_->timings_data_term_eval_.setZero();
	sd_->timings_smoothness_term_eval_.setZero();
	sd_->timings_qpbo_fuse_time_.setZero();
	sd_->timings_iteration_time_.setZero();
	

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	Eigen::Matrix<double, Dynamic, 1> D(num_pixels, 1);
	InitDisparityProposal(&D);

	int iter = sd_->average_over_; // starting value for iter (not +1 because 0-indexed instead of 1-indexed like Matlab)

	// initialize energies of phantom iterations to the maximum value. Phantom iterations are front-padded results to ensure can always average over the correct number of iterations; making them the maximum value ensures convergence is not achieved until enough iterations have been performed that these fall off the grid, so to speak. Another solution would be to simply run the loop for a minimum number of iterations before checking for convergence.
	for (int r = 0; r < sd_->average_over_; r++) {
		energy(r, 0) = DBL_MAX; // for float, would be FLT_MAX
	}
	energy(iter, 0) = DBL_MAX / 1e20; // ensures initial while check for loop entry does not meet convergence test by setting current iteration's energy significantly lower than earlier energies, so stabilization has not yet been achieved.

	// the new (proposal) depth map
	Eigen::Matrix<double, Dynamic, 1> Dnew(num_pixels, 1);

	bool include_smoothness_terms = true;

	// for the loop condition: note that energy is guaranteed not to increase, so convergence is checked by determining ratio of current energy to energy average_over iterations ago, then comparing to a convergence threshold that is dependent on the number of iterations over which we are averaging convergence, yielding a check that is truly an "averaging" of convergence.  More generally, if energy is falling, loop will continue.  Once has stabilized sufficiently, or once we exceed the maximum number of iterations, the loop will terminate.
	double t_loop;
	while ((1. - (energy(iter, 0) / energy(iter - sd_->average_over_, 0)) > sd_->converge_) &&
		(iter < sd_->max_iters_)) {
		t_loop = (double)getTickCount();
		iter++;

		Dnew = D; // copy over vals so get known disparities; important for each iteration because all used values are included in optimization calculation in order to include smoothness terms, even if no pairwise terms are included for known values, then results for unknown pixels only are collected, but need optimization to not think it's choosing an "illegal" known value swap that results in a high degree of smoothness, throwing off unknown pixel fuse results
		
		switch (proposal_method) {
		case PERC_UNI:
			DisparityProposalSelectionUpdate_PercUni(&Dnew);
			include_smoothness_terms = false;
			break;
		case SEG_PLN:
			DisparityProposalSelectionUpdate_SegPln(iter, &Dnew);
			include_smoothness_terms = true;
			break;
		default: // SMOOTH_STAR
			DisparityProposalSelectionUpdate_SmoothStar(iter - sd_->average_over_, D_crowdsourced, D_validmax, D_segpln, &D, &Dnew);
			//DisparityProposalSelectionUpdate_Smooth_WithOrig(iter - sd_->average_over_, D_existing, &D, &Dnew);
			break;
		}

		sd_->timings_data_term_eval_(0, iter) = (double)getTickCount() - t_loop; // time taken for proposal generation

		if (debug) {
			cout << "D" << endl;
			DisplayImages::DisplayGrayscaleImage(&D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
			cout << "Dnew" << endl;
			DisplayImages::DisplayGrayscaleImage(&Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}

		//LoadEigenMatrixBasic("MatlabD1", D);
		//LoadEigenMatrixBasic("MatlabD2", Dnew);

		/*
		if (debug) {
			Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
			bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, &Dnew, &pass);
			if (!passes) {
				cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks failed for Dnew" << endl;
			}
			else cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks passed for Dnew" << endl;
		}
		*/

		// Fuse the depths
		Eigen::Matrix<bool, Dynamic, 1> Dswap(num_pixels, 1); // matrix of booleans for each pixel where, if true, the value from Dnew should replace the current value in Dcurr
		map<int, map<int, int>> label_mappings; // empty here
		op_->FuseProposals(&D, &Dnew, &Dswap, iter, &energy, label_mappings, include_smoothness_terms);
		Dswap = (sd_->known_depths_[cid_out_].array()).select(Matrix<bool, Dynamic, 1>::Constant(num_pixels, false), Dswap);// sets Dswap coefficients to false for positions where known_depths_ coefficents are true; do this before swapping Dnew into D
		D = (Dswap.array()).select(Dnew, D); // sets D coefficients to Dnew coefficient values for positions where Dswap coefficients are true
		Eigen::Matrix<unsigned short, Dynamic, 1> iter_mat(num_pixels, 1);
		iter_mat.setConstant(iter);
		sd_->map_ = (Dswap.array()).select(iter_mat, sd_->map_); // sets maps_ coefficients to iter value for positions where Dswap coefficients are true
		iter_mat.resize(0, 1);

		double energy_curr = energy(iter, 0);
		double energy_prev = energy(iter - sd_->average_over_, 0);
		double compare = 1. - (energy_curr / energy_prev);
		cout << "Energy for this iteration: " << energy_curr << "; energy for energy(iter - sd_->average_over_, 0) " << energy_prev << "; test against convergence value (1 - (energy(iter, 0) / energy(iter - sd_->average_over_, 0)) " << compare << "; convergence occurs at " << sd_->converge_ << endl;
	}
	
	(*Dproposal) = D;

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, Dproposal, &pass);
		if (!passes) {
			cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(Dproposal, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Optimization::Stereo_optim() running time = " << t*1000. / getTickFrequency() << " ms" << endl << endl;
	}
}

/*
// updates Dproposal with the optimized disparity proposal
// D_sameuni and D_segpln only need relevant data when proposal_method is SMOOTH_STAR
// map<int, map<int, int>> label_mappings; // for cid_out_, map of cid_in => label_out => label_in
void StereoReconstruction::Stereo_optim(Matrix<double, Dynamic, 1> *Dproposal, const Matrix<double, Dynamic, 1> *D_crowdsourced, const Matrix<double, Dynamic, 1> *D_validmax, map<int, map<int, int>> label_mappings) {
	assert(Dproposal->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = true;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	int num_pixels = sd_->num_pixels_[cid_out_];
	int num_pixels_used = sd_->num_used_pixels_[cid_out_];

	// create initial arrays

	Matrix<double, Dynamic, 1> energy(sd_->max_iters_, 1);
	energy.setZero();
	sd_->map_.setZero();
	sd_->count_updated_vals_.setZero();
	sd_->count_unlabelled_vals_.setZero();
	sd_->count_unlabelled_regions_.setZero();
	sd_->count_unlabelled_after_QPBOP_.setZero();
	sd_->timings_data_term_eval_.setZero();
	sd_->timings_smoothness_term_eval_.setZero();
	sd_->timings_qpbo_fuse_time_.setZero();
	sd_->timings_iteration_time_.setZero();


	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	Eigen::Matrix<double, Dynamic, 1> D(num_pixels, 1);
	InitCurrentGuessProposal(&D);

	if (sd_->num_unknown_pixels_[cid_out_] == 0) {
		(*Dproposal) = D;
		if (debug) cout << "no unknown pixels found for this camera - retaining current values for knowns and skipping optimization for this camera" << endl;
		if (timing) {
			t = (double)getTickCount() - t;
			cout << "Optimization::Stereo_optim() running time = " << t*1000. / getTickFrequency() << " ms" << endl << endl;
		}
		return;
	}

	// initialize energies of phantom iterations to the maximum value. Phantom iterations are front-padded results to ensure can always average over the correct number of iterations; making them the maximum value ensures convergence is not achieved until enough iterations have been performed that these fall off the grid, so to speak. Another solution would be to simply run the loop for a minimum number of iterations before checking for convergence.
	for (int r = 0; r < sd_->average_over_; r++) {
		energy(r, 0) = DBL_MAX; // for float, would be FLT_MAX
	}

	// the new (proposal) depth map
	Eigen::Matrix<double, Dynamic, 1> Dnew(num_pixels, 1);

	bool include_smoothness_terms = true;

	// for the loop condition: note that energy is guaranteed not to increase, so convergence is checked by determining ratio of current energy to energy average_over iterations ago, then comparing to a convergence threshold that is dependent on the number of iterations over which we are averaging convergence, yielding a check that is truly an "averaging" of convergence.  More generally, if energy is falling, loop will continue.  Once has stabilized sufficiently, or once we exceed the maximum number of iterations, the loop will terminate.
	for (int iter = 0; iter < 2; iter++) {

		if (iter == 0)
			Dnew = (*D_crowdsourced);
		else
			Dnew = (*D_validmax);

		// clean proposals
		//CleanProposalSegments(&D);
		//CleanProposalSegments(&Dnew);
		UpdateProposalsFromColorMask(&D, &Dnew);

		if (debug) {
			cout << "after UpdateProposalsFromColorMask()" << endl;
			cout << "D" << endl;
			DisplayImages::DisplayGrayscaleImage(&D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
			cout << "Dnew" << endl;
			DisplayImages::DisplayGrayscaleImage(&Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}

		// Fuse the depths
		Eigen::Matrix<bool, Dynamic, 1> Dswap(num_pixels, 1); // matrix of booleans for each pixel where, if true, the value from Dnew should replace the current value in Dcurr
		op_->FuseProposals(&D, &Dnew, &Dswap, iter, &energy, label_mappings, include_smoothness_terms);
		Dswap = (sd_->known_depths_[cid_out_].array()).select(Matrix<bool, Dynamic, 1>::Constant(num_pixels, false), Dswap);// sets Dswap coefficients to false for positions where known_depths_ coefficents are true; do this before swapping Dnew into D
		D = (Dswap.array()).select(Dnew, D); // sets D coefficients to Dnew coefficient values for positions where Dswap coefficients are true
		Eigen::Matrix<unsigned short, Dynamic, 1> iter_mat(num_pixels, 1);
		iter_mat.setConstant(iter);
		sd_->map_ = (Dswap.array()).select(iter_mat, sd_->map_); // sets maps_ coefficients to iter value for positions where Dswap coefficients are true
		iter_mat.resize(0, 1);
	}

	(*Dproposal) = D;

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, Dproposal, &pass);
		if (!passes) {
			cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(Dproposal, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Optimization::Stereo_optim() running time = " << t*1000. / getTickFrequency() << " ms" << endl << endl;
	}
}
*/

// updates Dproposal with the optimized disparity proposal
// D_sameuni and D_segpln only need relevant data when proposal_method is SMOOTH_STAR
// map<int, map<int, int>> label_mappings; // for cid_out_, map of cid_in => label_out => label_in
void StereoReconstruction::Stereo_optim(Matrix<double, Dynamic, 1> *Dproposal, const Matrix<double, Dynamic, 1> *D_crowdsourced, const Matrix<double, Dynamic, 1> *D_validmax, map<int, map<int, int>> label_mappings) {
	assert(Dproposal->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	if (debug) cout << "StereoReconstruction::Stereo_optim()" << endl;

	int num_pixels = sd_->num_pixels_[cid_out_];
	int num_pixels_used = sd_->num_used_pixels_[cid_out_];

	// create initial arrays

	Matrix<double, Dynamic, 1> energy(sd_->max_iters_, 1);
	energy.setZero();
	sd_->map_.setZero();
	sd_->count_updated_vals_.setZero();
	sd_->count_unlabelled_vals_.setZero();
	sd_->count_unlabelled_regions_.setZero();
	sd_->count_unlabelled_after_QPBOP_.setZero();
	sd_->timings_data_term_eval_.setZero();
	sd_->timings_smoothness_term_eval_.setZero();
	sd_->timings_qpbo_fuse_time_.setZero();
	sd_->timings_iteration_time_.setZero();


	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	Eigen::Matrix<double, Dynamic, 1> D(num_pixels, 1);
	InitCurrentGuessProposal(&D);

	if (sd_->num_unknown_pixels_[cid_out_] == 0) {
		(*Dproposal) = D;
		if (debug) cout << "no unknown pixels found for this camera - retaining current values for knowns and skipping optimization for this camera" << endl;
		if (timing) {
			t = (double)getTickCount() - t;
			cout << "Optimization::Stereo_optim() running time = " << t*1000. / getTickFrequency() << " ms" << endl << endl;
		}
		return;
	}

	// initialize energies of phantom iterations to the maximum value. Phantom iterations are front-padded results to ensure can always average over the correct number of iterations; making them the maximum value ensures convergence is not achieved until enough iterations have been performed that these fall off the grid, so to speak. Another solution would be to simply run the loop for a minimum number of iterations before checking for convergence.
	for (int r = 0; r < sd_->average_over_; r++) {
		energy(r, 0) = DBL_MAX; // for float, would be FLT_MAX
	}

	// the new (proposal) depth map
	Eigen::Matrix<double, Dynamic, 1> Dnew(num_pixels, 1);

	bool include_smoothness_terms = true;

	Matrix<bool, Dynamic, 1> reproj_known(num_pixels, 1);

	// for the loop condition: note that energy is guaranteed not to increase, so convergence is checked by determining ratio of current energy to energy average_over iterations ago, then comparing to a convergence threshold that is dependent on the number of iterations over which we are averaging convergence, yielding a check that is truly an "averaging" of convergence.  More generally, if energy is falling, loop will continue.  Once has stabilized sufficiently, or once we exceed the maximum number of iterations, the loop will terminate.
	// fuse against all comparison proposals
	int iter = 0;
	for (vector<int>::iterator it = sd_->use_cids_[cid_out_].begin(); it != sd_->use_cids_[cid_out_].end(); ++it) {
		int cid = (*it);
		if (cid == cid_out_) continue;

		if (debug) cout << "Fuse against comparison for cid " << cid << endl;

		sd_->BuildComparisonDisparityProposal(cid_out_, cid, &Dnew, &reproj_known);
		UpdateProposalEmptiesToExisting(&D, &Dnew);

		// clean proposals
		//CleanProposalSegments(&D);
		//CleanProposalSegments(&Dnew);
		UpdateProposalsFromColorMask(&D, &Dnew);

		if (debug) {
			cout << "after UpdateProposalsFromColorMask()" << endl;
			cout << "D" << endl;
			DisplayImages::DisplayGrayscaleImage(&D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
			cout << "Dnew" << endl;
			DisplayImages::DisplayGrayscaleImage(&Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}

		// Fuse the depths
		if (debug) cout << "Fuse against max valid disparities" << endl;
		Eigen::Matrix<bool, Dynamic, 1> Dswap(num_pixels, 1); // matrix of booleans for each pixel where, if true, the value from Dnew should replace the current value in Dcurr
		op_->FuseProposals(&D, &Dnew, &Dswap, iter, &energy, label_mappings, include_smoothness_terms);
		Dswap = (sd_->known_depths_[cid_out_].array()).select(Matrix<bool, Dynamic, 1>::Constant(num_pixels, false), Dswap);// sets Dswap coefficients to false for positions where known_depths_ coefficents are true; do this before swapping Dnew into D
		D = (Dswap.array()).select(Dnew, D); // sets D coefficients to Dnew coefficient values for positions where Dswap coefficients are true
		Eigen::Matrix<unsigned short, Dynamic, 1> iter_mat(num_pixels, 1);
		iter_mat.setConstant(iter);
		sd_->map_ = (Dswap.array()).select(iter_mat, sd_->map_); // sets maps_ coefficients to iter value for positions where Dswap coefficients are true
		iter_mat.resize(0, 1);

		iter++;
	}

	
	// reinstate ************************************************************************************************************************
	// Fuse against valid maximum
	Dnew = (*D_validmax);
	UpdateProposalsFromColorMask(&D, &Dnew);
	Eigen::Matrix<bool, Dynamic, 1> Dswap(num_pixels, 1); // matrix of booleans for each pixel where, if true, the value from Dnew should replace the current value in Dcurr
	op_->FuseProposals(&D, &Dnew, &Dswap, iter, &energy, label_mappings, include_smoothness_terms);
	Dswap = (sd_->known_depths_[cid_out_].array()).select(Matrix<bool, Dynamic, 1>::Constant(num_pixels, false), Dswap);// sets Dswap coefficients to false for positions where known_depths_ coefficents are true; do this before swapping Dnew into D
	D = (Dswap.array()).select(Dnew, D); // sets D coefficients to Dnew coefficient values for positions where Dswap coefficients are true
	Eigen::Matrix<unsigned short, Dynamic, 1> iter_mat(num_pixels, 1);
	iter_mat.setConstant(iter);
	sd_->map_ = (Dswap.array()).select(iter_mat, sd_->map_); // sets maps_ coefficients to iter value for positions where Dswap coefficients are true
	iter_mat.resize(0, 1);
	

	(*Dproposal) = D;

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, Dproposal, &pass);
		if (!passes) {
			cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::Stereo_optim() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(Dproposal, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Optimization::Stereo_optim() running time = " << t*1000. / getTickFrequency() << " ms" << endl << endl;
	}
}


// updates a reference image disparity proposal for StereoReconstruction::Stereo_optim() with random values where depth is 0 and masked-in
void StereoReconstruction::UpdateProposalEmptiesToRandoms(Matrix<double, Dynamic, 1> *D) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	double *pD = D->data();
	bool *pM = sd_->masks_[cid_out_].data();
	double disp_val;
	bool unknown;
	int unk_pix_idx;
	int idx = 0;
	for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
		for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
			if (!*pM++) { // masked out
				*pD++ = 0.;
				idx++;
				continue;
			}
			disp_val = *pD;
			if (disp_val == 0.) { // depth is unknown or invalid for a masked-in pixel
				unk_pix_idx = sd_->unknown_maps_fwd_[cid_out_](idx, 0);

				if (unk_pix_idx == -1) { // will occur with known depths when don't trust the data and are treating the pixel depth as unknown despite having a value for it
					disp_val = sd_->min_disps_[cid_out_] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_])); // generates a random float in the range[sd_->min_disp_, sd_->max_disp_]; this approach may assign disparity values that do not conform to the masks of one or more other input images
				}
				else {
					disp_val = static_cast<double>(sd_->GenerateRandomValidDisparity(cid_out_, unk_pix_idx)); // if possible, generates disparity values that safely conform to the masks of all other input images; if not possible because no valid disparity exists (among quantized label positions), generates a random disparity in within the min/max disparity range
				}

				*pD = disp_val;
			}
			pD++;
			idx++;
		}
	}

	// truncate disparities to allowed range
	(*D) = D->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	(*D) = D->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, D, &pass);
		if (!passes) {
			cout << "StereoReconstruction::UpdateProposalEmptiesToRandoms() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::UpdateProposalEmptiesToRandoms() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(D, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// updates a reference image disparity proposal Dnew for StereoReconstruction::Stereo_optim() with values from Dcurr where disparity is 0 and masked-in; assumes Dcurr is completely filled in for unknown pixels with guesses or random values
void StereoReconstruction::UpdateProposalEmptiesToExisting(Matrix<double, Dynamic, 1> *Dcurr, Matrix<double, Dynamic, 1> *Dnew) {
	assert(D->rows() == sd_->num_pixels_[cid_out_]);

	bool debug = false;

	// initialize disparity map values; unavailable values initialized to a random value between sd_->min_disp_ and sd_->max_disp_ (note that Eigen has a function setRandom(), but no documentation is given on the range to which it sets the values; with boost, one bookmarked page suggests I can set this range for the command)
	double *pDcurr = Dcurr->data();
	double *pDnew = Dnew->data();
	bool *pM = sd_->masks_[cid_out_].data();
	double disp_val;
	bool unknown;
	int unk_pix_idx;
	int idx = 0;
	for (int c = 0; c < sd_->widths_[cid_out_]; c++) {
		for (int r = 0; r < sd_->heights_[cid_out_]; r++) {
			if (!*pM++) { // masked out
				*pDnew++ = 0.;
				*pDcurr++;
				idx++;
				continue;
			}
			disp_val = *pDnew;
			if (disp_val == 0.) { // depth is unknown or invalid for a masked-in pixel
				*pDnew = *pDcurr;
			}
			pDnew++;
			pDcurr++;
			idx++;
		}
	}

	// truncate disparities to allowed range
	(*Dnew) = Dnew->cwiseMax(sd_->min_disps_[cid_out_]); // truncate at min
	(*Dnew) = Dnew->cwiseMin(sd_->max_disps_[cid_out_]); // truncate at max

	if (debug) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, Dnew, &pass);
		if (!passes) {
			cout << "StereoReconstruction::UpdateProposalEmptiesToRandoms() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::UpdateProposalEmptiesToRandoms() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}
}

// updates Dnew with the result
// one random percentage which is applied to the valid range for each pixel
void StereoReconstruction::DisparityProposalSelectionUpdate_PercUni(Eigen::Matrix<double, Dynamic, 1> *Dnew) {
	assert(Dnew->rows() == sd_->num_pixels_[cid_out_]);
	// random fronto-parallel - a random number is chosen and assigned to all pixels, so they all get the same value
	bool debug = true;
	if (debug) cout << endl << "StereoReconstruction::DisparityProposalSelectionUpdate_PercUni()" << endl;
	float disp_perc = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)); // generates a random double in the range[0, 1]
	disp_perc = 1 - disp_perc * GLOBAL_RANDDISP_MAXPERCDEPTH;

	if (debug) cout << "using percent " << disp_perc * 100 << "%" << endl;

	int h = sd_->heights_[cid_out_];
	int w = sd_->widths_[cid_out_];
	int idx;
	int unk_idx = 0;
	double *pD = Dnew->data();
	for (int c = 0; c < w; c++) {
		for (int r = 0; r < h; r++) {
			idx = PixIndexFwdCM(Point(c, r), h);
			if (!sd_->masks_unknowns_[cid_out_](idx, 0)) {
				pD++;
				continue;
			}
			*pD = static_cast<double>(sd_->GenerateValidDisparityAtPerc(cid_out_, unk_idx, disp_perc));
			unk_idx++;
			pD++;
		}
	}
	sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew); // actually not necessary for new unknowns...what about knowns?
}

// updates Dnew with the result
void StereoReconstruction::DisparityProposalSelectionUpdate_SegPln(const int n, Eigen::Matrix<double, Dynamic, 1> *Dnew) {
	assert(Dnew->rows() == sd_->num_pixels_[cid_out_]);
	bool debug = true;
	if (debug) cout << endl << "StereoReconstruction::DisparityProposalSelectionUpdate_SegPln()" << endl;
	int p = (n - 1) % sd_->D_segpln_.cols();
	Matrix<double, Dynamic, 1> sgpln = sd_->D_segpln_.block(0, p, sd_->D_segpln_.rows(), 1).cast<double>();
	EigenMatlab::AssignByTruncatedBooleans(Dnew, &sd_->masks_unknowns_[cid_out_], &sgpln); // sd_->D_segpln_ is size num_unknown_pixels_out_ x map size
	sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew);
}

// convert D's values from the range[0, 1] to a disparity value in the range[sd_->min_disp_, sd_->max_disp_]
void StereoReconstruction::ExpandNormalizedDisparityMap(Matrix<double, Dynamic, 1> *D) {
	(*D) *= (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_]);
	(*D) = D->array() + sd_->min_disps_[cid_out_];
}

// updates Dnew with the result for the new disparity map proposal
// arg iter is the current iteration
// arg Dcurr is the current disparity map proposal
// sameunit_proposal is the proposal previously generated by the random fronto-parallel method
// segpln_proposal is the proposal previously generated by the prototypical segment-based stereo proposals method
// Methodology: D_crowdsourced has values based on disparities from other cameras, with the largest (closest) disparity winning in any races. Since there may be unaccounted for occlusions, the actual disparities could be closer. To account for that, validmax_proposal has values based on the closest valid disparity for each pixel according to the confluence of masks for all other cameras. This value may be too close, especially if there are convex surfaces, but provides a counter-proposal to validmax_proposal values that may be occluded.  SmoothStar optimizes between these two proposals and smoothed versions, always snapping disparity values to valid ranges.
void StereoReconstruction::DisparityProposalSelectionUpdate_SmoothStar(const int iter, const Eigen::Matrix<double, Dynamic, 1> *D_crowdsourced, const Eigen::Matrix<double, Dynamic, 1> *D_validmax, const Eigen::Matrix<double, Dynamic, 1> *D_segpln, const Eigen::Matrix<double, Dynamic, 1> *Dcurr, Eigen::Matrix<double, Dynamic, 1> *Dnew) {
	assert(Dnew->rows() == sd_->num_pixels_[cid_out_]);
	bool debug = true;
	bool debug_check = false;
	if (debug) cout << endl << "StereoReconstruction::DisparityProposalSelectionUpdate_SmoothStar()" << endl;
	int num_pixels = sd_->num_pixels_[cid_out_];

	int h = sd_->heights_[cid_out_];
	int w = sd_->widths_[cid_out_];
	int idx;
	int unk_idx = 0;
	double *pD = Dnew->data();
	
	// for near to far and far to near cases	
	int p = ((iter - 1) % 6);
	switch (p) {

	case 0: // crowdsourced
		if (debug) cout << "case " << p << ": D_crowdsourced" << endl;
		(*Dnew) = (*D_crowdsourced);
		break;

	case 1: // validmax
		if (debug) cout << "case " << p << ": validmax_proposal" << endl;
		(*Dnew) = (*D_validmax);
		(*Dnew) = Dnew->cwiseMax(sd_->min_disps_[cid_out_]); // enforce the minimum on the disparity values through truncation
		break;
	/*
	case 2: // segpln
		if (debug) cout << "case " << p << ": segpln_proposal" << endl;
		(*Dnew) = (*D_segpln);
		(*Dnew) = Dnew->cwiseMax(sd_->min_disps_[cid_out_]); // enforce the minimum on the disparity values through truncation
		break;
	*/
	default: // interpolate disparity over rows and columns
		if (debug) cout << "case " << p << ": interpolate disparity" << endl;

		// Dnew = (D - vals.d_min) / vals.d_step;  // changes Dnew from disparity values to a number in the range [0,1] that represents a disparity from d_min (0) to d_max (1), which is important to do before interpolation, but then must un-normalize again after interpolation
		(*Dnew) = Dcurr->array() - sd_->min_disps_[cid_out_];
		double factor = 1. / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_]);
		(*Dnew) *= factor;

		if (iter % 2) { // iter % 2 returns 1 (true) if iter is odd, 0 (false) if iter is even
			// interpolate disparities over rows - each unknown value becomes the average of the values that were immediately above and below, but only if a used pixel exists there
			// Dnew(2:end - 1, : ) = (Dnew(1:end - 2, : ) + Dnew(3:end, : )) / 2;
			
			bool used_above, used_below;
			int idx_above, idx_below, idx;
			int h = sd_->heights_[cid_out_];
			int w = sd_->widths_[cid_out_];
			for (int s = 0; s < GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER; s++) {
				for (int c = 0; c < w; c++) {
					for (int r = 1; r < (h - 1); r++) {
						idx = PixIndexFwdCM(Point(c, r), h);
						if (!sd_->masks_unknowns_[cid_out_](idx, 0)) continue;

						idx_above = idx - 1;
						idx_below = idx + 1;
						used_above = sd_->masks_[cid_out_](idx_above, 0);
						used_below = sd_->masks_[cid_out_](idx_below, 0);

						if ((sd_->masks_int_[cid_out_](idx, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL) ||
							(sd_->masks_int_[cid_out_](idx_above, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL) ||
							(sd_->masks_int_[cid_out_](idx_below, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							continue; // on segmentation boundary

						if ((sd_->masks_int_[cid_out_](idx_above, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_above, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_above = false;
						if ((sd_->masks_int_[cid_out_](idx_below, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_below, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_below = false;
						if ((used_above) &&
							(used_below))
							(*Dnew)(idx, 0) = 0.5*(*Dnew)(idx_above, 0) + 0.5*(*Dnew)(idx_below, 0);
						else if (used_above)
							(*Dnew)(idx, 0) = (*Dnew)(idx_above, 0);
						else if (used_below)
							(*Dnew)(idx, 0) = (*Dnew)(idx_below, 0);
					}
				}
			}
		}
		else {
			// interpolate disparities over columns - each unknown value becomes the average of the values that were immediately left and right, but only if a used pixel exists there
			// Dnew(:, 2 : end - 1) = (Dnew(:, 1 : end - 2) + Dnew(:, 3 : end)) / 2;

			bool used_left, used_right;
			int idx_left, idx_right, idx;
			int h = sd_->heights_[cid_out_];
			int w = sd_->widths_[cid_out_];
			for (int s = 0; s < GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER; s++) {
				for (int c = 1; c < (w - 1); c++) {
					for (int r = 0; r < h; r++) {
						idx = PixIndexFwdCM(Point(c, r), h);
						if (!sd_->masks_unknowns_[cid_out_](idx, 0)) continue;

						idx_left = idx - h;
						idx_right = idx + h;
						used_left = sd_->masks_[cid_out_](idx_left, 0);
						used_right = sd_->masks_[cid_out_](idx_right, 0);

						if ((sd_->masks_int_[cid_out_](idx, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL) ||
							(sd_->masks_int_[cid_out_](used_left, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL) ||
							(sd_->masks_int_[cid_out_](used_right, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							continue; // on segmentation boundary

						if ((sd_->masks_int_[cid_out_](idx_left, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_left, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_left = false;
						if ((sd_->masks_int_[cid_out_](idx_right, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_right, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_right = false;

						if ((used_left) &&
							(used_right))
							(*Dnew)(idx, 0) = 0.5*(*Dnew)(idx_left, 0) + 0.5*(*Dnew)(idx_right, 0);
						else if (used_left)
							(*Dnew)(idx, 0) = (*Dnew)(idx_left, 0);
						else if (used_right)
							(*Dnew)(idx, 0) = (*Dnew)(idx_right, 0);
					}
				}
			}
		}

		ExpandNormalizedDisparityMap(Dnew);
		sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable

		break;
	}

	if (debug_check) {
		Matrix<bool, Dynamic, 1> pass(sd_->num_pixels_[cid_out_], 1);
		bool passes = sd_->TestDisparitiesAgainstMasks(cid_out_, Dnew, &pass);
		if (!passes) {
			cout << "StereoReconstruction::DisparityProposalSelectionUpdate_SmoothStar() TestDisparitiesAgainstMasks failed" << endl;
			DisplayImages::DisplayGrayscaleImage(&pass, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
		}
		else cout << "StereoReconstruction::DisparityProposalSelectionUpdate_SmoothStar() TestDisparitiesAgainstMasks passed" << endl;
		DisplayImages::DisplayGrayscaleImage(Dnew, sd_->heights_[cid_out_], sd_->widths_[cid_out_], sd_->orientations_[cid_out_]);
	}

	(*Dnew) = Dnew->cwiseMax(sd_->min_disps_[cid_out_]); // enforce the minimum on the disparity values through truncation
}

// updates Dnew with the result for the new disparity map proposal
// arg iter is the current iteration
// arg Dcurr is the current disparity map proposal
// sameunit_proposal is the proposal previously generated by the random fronto-parallel method
// segpln_proposal is the proposal previously generated by the prototypical segment-based stereo proposals method
// Methodology: D_crowdsourced has values based on disparities from other cameras, with the largest (closest) disparity winning in any races. Since there may be unaccounted for occlusions, the actual disparities could be closer. To account for that, validmax_proposal has values based on the closest valid disparity for each pixel according to the confluence of masks for all other cameras. This value may be too close, especially if there are convex surfaces, but provides a counter-proposal to validmax_proposal values that may be occluded.  SmoothStar optimizes between these two proposals and smoothed versions, always snapping disparity values to valid ranges.
void StereoReconstruction::DisparityProposalSelectionUpdate_SmoothStar_Comparison(const int iter, const Eigen::Matrix<double, Dynamic, 1> *proposal1, const Eigen::Matrix<double, Dynamic, 1> *proposal2, const Eigen::Matrix<double, Dynamic, 1> *Dcurr, Eigen::Matrix<double, Dynamic, 1> *Dnew) {
	assert(Dnew->rows() == sd_->num_pixels_[cid_out_]);
	bool debug = false;
	if (debug) cout << endl << "StereoReconstruction::DisparityProposalSelectionUpdate_SmoothStar()" << endl;
	int num_pixels = sd_->num_pixels_[cid_out_];

	int p = ((iter - 1) % 6);
	switch (p) {
	case 0: // validmin
		if (debug) cout << "case 0: proposal1" << endl;
		(*Dnew) = (*proposal1);
		break;
	case 1: // validmin
		if (debug) cout << "case 1: proposal2" << endl;
		(*Dnew) = (*proposal2);
		break;
	default: // interpolate disparity over rows and columns
		if (debug) cout << "case 2: interpolate disparity" << endl;

		// Dnew = (D - vals.d_min) / vals.d_step;  // changes Dnew from disparity values to a number in the range [0,1] that represents a disparity from d_min (0) to d_max (1), which is important to do before interpolation, but then must un-normalize again after interpolation
		(*Dnew) = Dcurr->array() - sd_->min_disps_[cid_out_];
		double factor = 1. / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_]);
		(*Dnew) *= factor;

		if (iter % 2) { // iter % 2 returns 1 (true) if iter is odd, 0 (false) if iter is even
			// interpolate disparities over rows - each unknown value becomes the average of the values that were immediately above and below, but only if a used pixel exists there
			// Dnew(2:end - 1, : ) = (Dnew(1:end - 2, : ) + Dnew(3:end, : )) / 2;

			bool used_above, used_below;
			int idx_above, idx_below, idx;
			int h = sd_->heights_[cid_out_];
			int w = sd_->widths_[cid_out_];
			for (int s = 0; s < GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER; s++) {
				for (int c = 0; c < w; c++) {
					for (int r = 1; r < (h - 1); r++) {
						idx = PixIndexFwdCM(Point(c, r), h);
						if (!sd_->masks_unknowns_[cid_out_](idx, 0)) continue;
						if ((sd_->masks_int_[cid_out_](idx, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							continue; // on segmentation boundary

						idx_above = idx - 1;
						idx_below = idx + 1;
						used_above = sd_->masks_[cid_out_](idx_above, 0);
						used_below = sd_->masks_[cid_out_](idx_below, 0);
						if ((sd_->masks_int_[cid_out_](idx_above, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_above, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_above = false;
						if ((sd_->masks_int_[cid_out_](idx_below, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_below, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_below = false;
						if ((used_above) &&
							(used_below))
							(*Dnew)(idx, 0) = 0.5*(*Dnew)(idx_above, 0) + 0.5*(*Dnew)(idx_below, 0);
						else if (used_above)
							(*Dnew)(idx, 0) = (*Dnew)(idx_above, 0);
						else if (used_below)
							(*Dnew)(idx, 0) = (*Dnew)(idx_below, 0);
					}
				}
			}
		}
		else {
			// interpolate disparities over columns - each unknown value becomes the average of the values that were immediately left and right, but only if a used pixel exists there
			// Dnew(:, 2 : end - 1) = (Dnew(:, 1 : end - 2) + Dnew(:, 3 : end)) / 2;

			bool used_left, used_right;
			int idx_left, idx_right, idx;
			int h = sd_->heights_[cid_out_];
			int w = sd_->widths_[cid_out_];
			for (int s = 0; s < GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER; s++) {
				for (int c = 1; c < (w - 1); c++) {
					for (int r = 0; r < h; r++) {
						idx = PixIndexFwdCM(Point(c, r), h);
						if (!sd_->masks_unknowns_[cid_out_](idx, 0)) continue;
						if ((sd_->masks_int_[cid_out_](idx, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							continue; // on segmentation boundary

						idx_left = idx - h;
						idx_right = idx + h;
						used_left = sd_->masks_[cid_out_](idx_left, 0);
						used_right = sd_->masks_[cid_out_](idx_right, 0);
						if ((sd_->masks_int_[cid_out_](idx_left, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_left, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_left = false;
						if ((sd_->masks_int_[cid_out_](idx_right, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_right, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_right = false;

						if ((used_left) &&
							(used_right))
							(*Dnew)(idx, 0) = 0.5*(*Dnew)(idx_left, 0) + 0.5*(*Dnew)(idx_right, 0);
						else if (used_left)
							(*Dnew)(idx, 0) = (*Dnew)(idx_left, 0);
						else if (used_right)
							(*Dnew)(idx, 0) = (*Dnew)(idx_right, 0);
					}
				}
			}
		}

		ExpandNormalizedDisparityMap(Dnew);
		//sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable

		break;
	}

	(*Dnew) = Dnew->cwiseMax(sd_->min_disps_[cid_out_]); // enforce the minimum on the disparity values through truncation
}

// updates Dnew with the result for the new disparity map proposal
// arg iter is the current iteration
// arg Dcurr is the current disparity map proposal
// sameunit_proposal is the proposal previously generated by the random fronto-parallel method
// segpln_proposal is the proposal previously generated by the prototypical segment-based stereo proposals method
// Methodology: D_crowdsourced has values based on disparities from other cameras, with the largest (closest) disparity winning in any races. Since there may be unaccounted for occlusions, the actual disparities could be closer. To account for that, validmax_proposal has values based on the closest valid disparity for each pixel according to the confluence of masks for all other cameras. This value may be too close, especially if there are convex surfaces, but provides a counter-proposal to validmax_proposal values that may be occluded.  SmoothStar optimizes between these two proposals and smoothed versions, always snapping disparity values to valid ranges.
void StereoReconstruction::DisparityProposalSelectionUpdate_Smooth_WithOrig(const int iter, const Eigen::Matrix<double, Dynamic, 1> *Dproposal_orig, const Eigen::Matrix<double, Dynamic, 1> *Dcurr, Eigen::Matrix<double, Dynamic, 1> *Dnew) {
	assert(Dnew->rows() == sd_->num_pixels_[cid_out_]);
	bool debug = true;
	if (debug) cout << endl << "DisparityProposalSelectionUpdate_Smooth_WithOrig()" << endl;
	int num_pixels = sd_->num_pixels_[cid_out_];

	int p = ((iter - 1) % 5);
	switch (p) {
	case 0: // proposal_orig
		if (debug) cout << "case 0: Dproposal_orig" << endl;
		(*Dnew) = (*Dproposal_orig);
		break;
	default: // interpolate disparity over rows and columns
		if (debug) cout << "case 1: interpolate disparity" << endl;

		// Dnew = (D - vals.d_min) / vals.d_step;  // changes Dnew from disparity values to a number in the range [0,1] that represents a disparity from d_min (0) to d_max (1), which is important to do before interpolation, but then must un-normalize again after interpolation
		(*Dnew) = Dcurr->array() - sd_->min_disps_[cid_out_];
		double factor = 1. / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_]);
		(*Dnew) *= factor;

		if (iter % 2) { // iter % 2 returns 1 (true) if iter is odd, 0 (false) if iter is even
			// interpolate disparities over rows - each unknown value becomes the average of the values that were immediately above and below, but only if a used pixel exists there
			// Dnew(2:end - 1, : ) = (Dnew(1:end - 2, : ) + Dnew(3:end, : )) / 2;

			bool used_above, used_below;
			int idx_above, idx_below, idx;
			int h = sd_->heights_[cid_out_];
			int w = sd_->widths_[cid_out_];
			for (int s = 0; s < GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER; s++) {
				for (int c = 0; c < w; c++) {
					for (int r = 1; r < (h - 1); r++) {
						idx = PixIndexFwdCM(Point(c, r), h);
						if (!sd_->masks_unknowns_[cid_out_](idx, 0)) continue;
						if ((sd_->masks_int_[cid_out_](idx, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							continue; // on segmentation boundary

						idx_above = idx - 1;
						idx_below = idx + 1;
						used_above = sd_->masks_[cid_out_](idx_above, 0);
						used_below = sd_->masks_[cid_out_](idx_below, 0);
						if ((sd_->masks_int_[cid_out_](idx_above, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_above, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_above = false;
						if ((sd_->masks_int_[cid_out_](idx_below, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_below, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_below = false;
						if ((used_above) &&
							(used_below))
							(*Dnew)(idx, 0) = 0.5*(*Dnew)(idx_above, 0) + 0.5*(*Dnew)(idx_below, 0);
						else if (used_above)
							(*Dnew)(idx, 0) = (*Dnew)(idx_above, 0);
						else if (used_below)
							(*Dnew)(idx, 0) = (*Dnew)(idx_below, 0);
					}
				}
			}
		}
		else {
			// interpolate disparities over columns - each unknown value becomes the average of the values that were immediately left and right, but only if a used pixel exists there
			// Dnew(:, 2 : end - 1) = (Dnew(:, 1 : end - 2) + Dnew(:, 3 : end)) / 2;

			bool used_left, used_right;
			int idx_left, idx_right, idx;
			int h = sd_->heights_[cid_out_];
			int w = sd_->widths_[cid_out_];
			for (int s = 0; s < GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER; s++) {
				for (int c = 1; c < (w - 1); c++) {
					for (int r = 0; r < h; r++) {
						idx = PixIndexFwdCM(Point(c, r), h);
						if (!sd_->masks_unknowns_[cid_out_](idx, 0)) continue;
						if ((sd_->masks_int_[cid_out_](idx, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							continue; // on segmentation boundary

						idx_left = idx - h;
						idx_right = idx + h;
						used_left = sd_->masks_[cid_out_](idx_left, 0);
						used_right = sd_->masks_[cid_out_](idx_right, 0);
						if ((sd_->masks_int_[cid_out_](idx_left, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_left, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_left = false;
						if ((sd_->masks_int_[cid_out_](idx_right, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
							(sd_->masks_int_[cid_out_](idx_right, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
							used_right = false;

						if ((used_left) &&
							(used_right))
							(*Dnew)(idx, 0) = 0.5*(*Dnew)(idx_left, 0) + 0.5*(*Dnew)(idx_right, 0);
						else if (used_left)
							(*Dnew)(idx, 0) = (*Dnew)(idx_left, 0);
						else if (used_right)
							(*Dnew)(idx, 0) = (*Dnew)(idx_right, 0);
					}
				}
			}
		}

		ExpandNormalizedDisparityMap(Dnew);
		//sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable

		break;
	}

	(*Dnew) = Dnew->cwiseMax(sd_->min_disps_[cid_out_]); // enforce the minimum on the disparity values through truncation
}

// updates Dnew with the result for the new disparity map proposal
// arg iter is the current iteration
// arg Dcurr is the current disparity map proposal
// sameunit_proposal is the proposal previously generated by the random fronto-parallel method
// segpln_proposal is the proposal previously generated by the prototypical segment-based stereo proposals method
// Methodology: D_crowdsourced has values based on disparities from other cameras, with the largest (closest) disparity winning in any races. Since there may be unaccounted for occlusions, the actual disparities could be closer. To account for that, validmax_proposal has values based on the closest valid disparity for each pixel according to the confluence of masks for all other cameras. This value may be too close, especially if there are convex surfaces, but provides a counter-proposal to validmax_proposal values that may be occluded.  SmoothStar optimizes between these two proposals and smoothed versions, always snapping disparity values to valid ranges.
void StereoReconstruction::DisparityProposalSelectionUpdate_Smooth(const int iter, const Eigen::Matrix<double, Dynamic, 1> *Dcurr, Eigen::Matrix<double, Dynamic, 1> *Dnew) {
	assert(Dnew->rows() == sd_->num_pixels_[cid_out_]);
	bool debug = false;
	if (debug) cout << endl << "DisparityProposalSelectionUpdate_Smooth()" << endl;
	int num_pixels = sd_->num_pixels_[cid_out_];

	// Dnew = (D - vals.d_min) / vals.d_step;  // changes Dnew from disparity values to a number in the range [0,1] that represents a disparity from d_min (0) to d_max (1), which is important to do before interpolation, but then must un-normalize again after interpolation
	(*Dnew) = Dcurr->array() - sd_->min_disps_[cid_out_];
	double factor = 1. / (sd_->max_disps_[cid_out_] - sd_->min_disps_[cid_out_]);
	(*Dnew) *= factor;

	if (iter % 2) { // iter % 2 returns 1 (true) if iter is odd, 0 (false) if iter is even
		// interpolate disparities over rows - each unknown value becomes the average of the values that were immediately above and below, but only if a used pixel exists there
		// Dnew(2:end - 1, : ) = (Dnew(1:end - 2, : ) + Dnew(3:end, : )) / 2;

		bool used_above, used_below;
		int idx_above, idx_below, idx;
		int h = sd_->heights_[cid_out_];
		int w = sd_->widths_[cid_out_];
		for (int s = 0; s < GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER; s++) {
			for (int c = 0; c < w; c++) {
				for (int r = 1; r < (h - 1); r++) {
					idx = PixIndexFwdCM(Point(c, r), h);
					if (!sd_->masks_unknowns_[cid_out_](idx, 0)) continue;
					if ((sd_->masks_int_[cid_out_](idx, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
						(sd_->masks_int_[cid_out_](idx, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
						continue; // on segmentation boundary

					idx_above = idx - 1;
					idx_below = idx + 1;
					used_above = sd_->masks_[cid_out_](idx_above, 0);
					used_below = sd_->masks_[cid_out_](idx_below, 0);
					if ((sd_->masks_int_[cid_out_](idx_above, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
						(sd_->masks_int_[cid_out_](idx_above, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
						used_above = false;
					if ((sd_->masks_int_[cid_out_](idx_below, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
						(sd_->masks_int_[cid_out_](idx_below, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
						used_below = false;
					if ((used_above) &&
						(used_below))
						(*Dnew)(idx, 0) = 0.5*(*Dnew)(idx_above, 0) + 0.5*(*Dnew)(idx_below, 0);
					else if (used_above)
						(*Dnew)(idx, 0) = (*Dnew)(idx_above, 0);
					else if (used_below)
						(*Dnew)(idx, 0) = (*Dnew)(idx_below, 0);
				}
			}
		}
	}
	else {
		// interpolate disparities over columns - each unknown value becomes the average of the values that were immediately left and right, but only if a used pixel exists there
		// Dnew(:, 2 : end - 1) = (Dnew(:, 1 : end - 2) + Dnew(:, 3 : end)) / 2;

		bool used_left, used_right;
		int idx_left, idx_right, idx;
		int h = sd_->heights_[cid_out_];
		int w = sd_->widths_[cid_out_];
		for (int s = 0; s < GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER; s++) {
			for (int c = 1; c < (w - 1); c++) {
				for (int r = 0; r < h; r++) {
					idx = PixIndexFwdCM(Point(c, r), h);
					if (!sd_->masks_unknowns_[cid_out_](idx, 0)) continue;
					if ((sd_->masks_int_[cid_out_](idx, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
						(sd_->masks_int_[cid_out_](idx, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
						continue; // on segmentation boundary

					idx_left = idx - h;
					idx_right = idx + h;
					used_left = sd_->masks_[cid_out_](idx_left, 0);
					used_right = sd_->masks_[cid_out_](idx_right, 0);
					if ((sd_->masks_int_[cid_out_](idx_left, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
						(sd_->masks_int_[cid_out_](idx_left, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
						used_left = false;
					if ((sd_->masks_int_[cid_out_](idx_right, 0) >= GLOBAL_MIN_MASKSEG_LINEVAL) &&
						(sd_->masks_int_[cid_out_](idx_right, 0) <= GLOBAL_MAX_MASKSEG_LINEVAL))
						used_right = false;

					if ((used_left) &&
						(used_right))
						(*Dnew)(idx, 0) = 0.5*(*Dnew)(idx_left, 0) + 0.5*(*Dnew)(idx_right, 0);
					else if (used_left)
						(*Dnew)(idx, 0) = (*Dnew)(idx_left, 0);
					else if (used_right)
						(*Dnew)(idx, 0) = (*Dnew)(idx_right, 0);
				}
			}
		}
	}

	ExpandNormalizedDisparityMap(Dnew);
	//sd_->SnapDisparitiesToValidRanges(cid_out_, Dnew); // might think we don't want to do this because as a result we will not be able to smooth properly because smooth smooth attempts will be snapped back. But actually we need it to combat smoothing across fg/bg depth discontinuities; actually, no longer need to worry about smoothing across fg/bg depth discontinuities since fg/bg segmentation is now reliable

	(*Dnew) = Dnew->cwiseMax(sd_->min_disps_[cid_out_]); // enforce the minimum on the disparity values through truncation
}