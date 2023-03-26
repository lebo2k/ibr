#ifndef StereoReconstruction_H
#define StereoReconstruction_H

#include "Globals.h"
#include "Calibration.h"
#include "Sensor.h"
#include "DepthMap.h"
#include "Scene.h"
#include "Interpolation.h"
#include "Optimization.h"
#include "StereoData.h"

/*
	input images' projection matrices should be relative to output image's projection matrix

	use range of depths in scene +20% on either side
	to determine number of discretized disparity labels, project pixel 0,0 into all views and ensure that minimum space between samples is 0.5 pixels

	prior: planar ... planar = true
	smoothness kernel: truncated linear ... smoothness_kernel = 1
	graph connectivity: 4 connected (bi-directional) ... connect = 4
	optimization algorithm: QPBO-R ... improve = 2
	proposal method: smooth* ... proposal_method = 3
	visibility = true (use geometric visbility constraint)
	compress_graph = false (compression makes graph smaller, but is slower)
	independent = false (use strongly-connected, rather than independent, regions)
*/

//Stereo Depth Reconstruction
class StereoReconstruction {

private:
	
	void ExpandNormalizedDisparityMap(Matrix<double, Dynamic, 1> *D);// convert D's values from the range[0, 1] to a disparity value in the range[sd_->min_disp_, sd_->max_disp_]
	void InitValidMaximumProposal(Matrix<double, Dynamic, 1> *D); // use valid disp range valid maximums, where one exists, as additional disparity proposal for smoothstar phase with valid randoms where no valid range is available
	void InitValidMinimumProposal(Matrix<double, Dynamic, 1> *D); // use valid disp range valid minimums, where one exists, as additional disparity proposal for smoothstar phase with valid randoms where no valid range is available
	void InitCrowdSourcedProposal(Matrix<double, Dynamic, 1> *D); // initializes a reference image disparity proposal for StereoReconstruction::Stereo_optim() by utilizing depth information where already known, crowd-sourced depth information where not, and random values where neither are available
	void InitDisparityProposal(Matrix<double, Dynamic, 1> *D); // initializes a reference image disparity proposal for StereoReconstruction::Stereo_optim() by utilizing depth information where already "known" and random values where not
	void InitSegPlnProposal(Matrix<double, Dynamic, 1> *D);
	void InitCurrentGuessProposal(Matrix<double, Dynamic, 1> *D); // initializes a reference image disparity proposal for StereoReconstruction::Stereo_optim() by utilizing depth information where already available, even if it's just a guess, and random values where not
	void UpdateProposalEmptiesToRandoms(Matrix<double, Dynamic, 1> *D); // updates a reference image disparity proposal for StereoReconstruction::Stereo_optim() with random values where depth is 0 and masked-in
	void UpdateProposalEmptiesToExisting(Matrix<double, Dynamic, 1> *Dcurr, Matrix<double, Dynamic, 1> *Dnew); // updates a reference image disparity proposal Dnew for StereoReconstruction::Stereo_optim() with values from Dcurr where depth is 0 and masked-in
	void CleanProposalSegments(Matrix<double, Dynamic, 1> *D); // ensures neighboring pixels belonging to the same segment are within GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT WS units of each other in depth
	void ConductDepthTest(Matrix<double, Dynamic, Dynamic> *depth_map, Matrix<bool, Dynamic, Dynamic> *label_flags, Matrix<bool, Dynamic, Dynamic> *pass); // performs depth test on depth_map values of all pixels of label for which label_flags is true and updates pass with the result by pixel



	// temporary
	void Stereo_optim_sameuni(Matrix<double, Dynamic, 1> *Dproposal);
	void DisparityProposalSelectionUpdate_SameUni(Eigen::Matrix<double, Dynamic, 1> *Dnew);
	void InitRandomProposal(Matrix<double, Dynamic, 1> *D);



public:

	std::string scene_name_;
	StereoData *sd_;
	Optimization *op_;
	Segmentation *seg_;

	int cid_out_; // ID of reference (output) camera

	// Constructors / destructor
	StereoReconstruction(std::string scene_name);
	~StereoReconstruction();

	void Init(std::map<int, Mat> imgsT, std::map<int, Mat> imgMasks, std::map<int, Mat> imgMasks_valid, std::map<int, MatrixXf> depth_maps, std::map<int, Matrix3d> Ks, std::map<int, Matrix3d> Kinvs, std::map<int, Matrix4d> RTs, std::map<int, Matrix4d> RTinvs, std::map<int, Matrix<double, 3, 4>> Ps, std::map<int, Matrix<double, 4, 3>> Pinvs, std::map<int, float> min_depths, std::map<int, float> max_depths, std::map<int, std::map<unsigned int, Matrix<unsigned int, Dynamic, 1>>> unknown_segs, std::map<int, float> agisoft_to_world_scales, Matrix4d AgisoftToWorld, Matrix4d WorldToAgisoft, std::vector<int> exclude_cam_ids, int max_num_cams, Scene *scene);
	void Init_MiddleburyCones(int cid_ref);

	void ReconstructAll(std::vector<int> cids, Scene *scene, bool save_as_computed, bool all_max = false); // reconstruct depth maps for cameras with IDs in cids; if all_max, assumes all pixels in all segments should be assigned valid maximum disparities without optimization
	void Stereo(int cid_out, bool all_max = false); // updates depth map for camera cid_out in sd_->depth_maps_; analysis to utilize only those cameras whose IDs are given in use_cids; list must include output camera ID; if all_flat, assumes all pixels in all segments should be assigned valid maximum disparities without optimization
	void Stereo_optim_old(const GLOBAL_PROPOSAL_METHOD proposal_method, Matrix<double, Dynamic, 1> *Dproposal, const Matrix<double, Dynamic, 1> *D_crowdsourced = NULL, const Matrix<double, Dynamic, 1> *D_validmax = NULL, const Matrix<double, Dynamic, 1> *D_segpln = NULL); // tiven a binary energy function for a stereo problem, and a set of proposals(or proposal index), fuses these proposals until convergence of the energy; updates Dproposal with the outcome
	void Stereo_optim(Matrix<double, Dynamic, 1> *Dproposal, const Matrix<double, Dynamic, 1> *D_crowdsourced, const Matrix<double, Dynamic, 1> *D_validmax, map<int, map<int, int>> label_mappings);
	void DisparityProposalSelectionUpdate_PercUni(Eigen::Matrix<double, Dynamic, 1> *Dnew); // updates Dnew with the result
	void DisparityProposalSelectionUpdate_SegPln(const int n, Eigen::Matrix<double, Dynamic, 1> *Dnew); // updates Dnew with the result
	void DisparityProposalSelectionUpdate_SmoothStar(const int iter, const Eigen::Matrix<double, Dynamic, 1> *D_crowdsourced, const Eigen::Matrix<double, Dynamic, 1> *D_validmax, const Eigen::Matrix<double, Dynamic, 1> *D_segpln, const Eigen::Matrix<double, Dynamic, 1> *Dcurr, Eigen::Matrix<double, Dynamic, 1> *Dnew); // updates Dnew with the result
	void DisparityProposalSelectionUpdate_SmoothStar_Comparison(const int iter, const Eigen::Matrix<double, Dynamic, 1> *proposal1, const Eigen::Matrix<double, Dynamic, 1> *proposal2, const Eigen::Matrix<double, Dynamic, 1> *Dcurr, Eigen::Matrix<double, Dynamic, 1> *Dnew);
	void DisparityProposalSelectionUpdate_Smooth_WithOrig(const int iter, const Eigen::Matrix<double, Dynamic, 1> *Dproposal_orig, const Eigen::Matrix<double, Dynamic, 1> *Dcurr, Eigen::Matrix<double, Dynamic, 1> *Dnew);
	void DisparityProposalSelectionUpdate_Smooth(const int iter, const Eigen::Matrix<double, Dynamic, 1> *Dcurr, Eigen::Matrix<double, Dynamic, 1> *Dnew);
	void ImproveKnowns();
	void Stereo_optim_singlecompare(Matrix<double, Dynamic, 1> *Dcurr, Matrix<double, Dynamic, 1> *Dproposal, Matrix<double, Dynamic, 1> *Dnew);

	void UpdateProposalsFromColorMask(Matrix<double, Dynamic, 1> *Dcurr, Matrix<double, Dynamic, 1> *Dnew); // pixels in imgMasks_color_[cid] that are (255, 0, 0) signify that we should assign the max valid disparity to the pixel and treat it as known; we do this by updating both proposal this way rather than changing the knowns list for the camera because the valid ranges are not available earlier when the knowns lists is set
	
	void SyncDisparityData(std::vector<int> cids, int max_num_cams, Scene *scene); // syncs disparity data across cameras as best as possible
	void SmoothDisparityData(int cid_out);
};

#endif