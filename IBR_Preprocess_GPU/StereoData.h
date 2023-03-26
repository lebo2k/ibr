#ifndef StereoData_H
#define StereoData_H

#include "Globals.h"
#include "DisplayImages.h"
#include "Interpolation.h"
#include "Camera.h"
#include "Scene.h"

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// time function includes (for sleep)
#include <chrono>
#include <thread>

using namespace std;
using namespace Eigen;


// data class for stereo reconstruction
class StereoData {

private:

	
	void UpdateDisps(int cid_specific = -1); // updates disps_, num_disps_, min_disp_, max_disp_, and disp_step_ members based on cid_out_, use_cids_ and other data
	void InitPout2ins(); // transforms the reference frame and sets the extrinsics matrices and output extrinsics matrix such that the output camera extrinsics matrix is the identity
	void InitUnknownAndUsedDisparityCoordinates(int cid, Matrix<double, Dynamic, 1> *Xuseds, Matrix<double, Dynamic, 1> *Yuseds, Matrix<double, Dynamic, 1> *Xunknowns, Matrix<double, Dynamic, 1> *Yunknowns, Matrix<unsigned int, Dynamic, 1> *Iunknowns); // for camera with ID cid, updates Xunknowns and Yunknowns to hold coordinates of a grid of size height x width that only includes positions for which Mask is true (the count of these positions is also given in num_mask_true to speed the function
	Matrix<float, 3, 4> StereoData::Pss1Toss2(int cid1, int cid2); // computes a projection matrix that transforms coordinates directly from cid1 screen space to cid2 screen space
	bool CheckCreateFaceOcclusion(int cid, Point p); // conducts test so that we don't build faces on pixels unless all pixels in neighborhood are within reasonable depth distance of each other (ensuring it represents a shared edge and not an occlusion edge); returns true if it passes this test, false otherwise

public:

	// depth and disparity members
	std::map<int, float> min_depths_; // map of camera ID => minimum depth value for scene in camera space, so it's in Agisoft units, from the perspective of the camera with the given ID as the reference camera; included for all cameras, not just those used, so can update disparity info if use_cids_ list changes; may not include padding added when computing disparity ranges
	std::map<int, float> max_depths_; // maximum depth value for scene in camera space, so it's in Agisoft units, from the perspective of the camera with the given ID as the reference camera; included for all cameras, not just those used, so can update disparity info if use_cids_ list changes; may not include padding added when computing disparity ranges
	map<int, float> min_disps_; // minimum disparity value for scene in camera space, from the perspective of the camera with the given ID as the reference camera; only takes cameras in use_cids into account
	map<int, float> max_disps_; // maximum disparity value for scene in camera space, from the perspective of the camera with the given ID as the reference camera; only takes cameras in use_cids into account
	map<int, float> disp_steps_; // change in disparity value between adjacent disparity labels, from the perspective of the camera with the given ID as the reference camera; max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp; only takes cameras in use_cids into account
	map<int, int> nums_disps_; // number of disparities, from the perspective of the camera with the given ID as the reference camera; only takes cameras in use_cids into account
	map<int, ArrayXd> disps_; // num_disps_ array of descending disparities to sample at in rendering, from the perspective of the camera with the given ID as the reference camera; only takes cameras in use_cids into account

	// scene and output members
	Matrix4d AgisoftToWorld_; // transform from Agisoft space to the world space we've defined in the chunk containing our scene in the Agisoft UI; necessary because Agisoft's camera extrinsic matrices and depth map values are in Agisoft's default space, not the transformed chunk space that takes effect once the coordinate system is updated by setting markers, entering positions for them, and updating the coordinate system
	Matrix4d WorldToAgisoft_; // inverse of AgisoftToWorld_
	map<int, float> agisoft_to_world_scales_; // map of camera ID => agisoft_to_world scale factor
	map<int, GLOBAL_AGI_CAMERA_ORIENTATION> orientations_; // map of camera ID => orientation
	map<int, vector<int>> use_cids_; // camera IDs for cameras to be used in analysis; must include output camera ID
	map<int, string> fn_imgs_; // map of camera ID => imgT filename without path
	map<int, int> heights_; // height in pixels of output display
	map<int, int> widths_; // width in pixels of output display
	std::map<int, Mat> imgsT_; // map of camera ID => image for input texture images of type CV_8UC3
	std::map<int, Mat> imgMasks_; // map of camera ID => image mask for input texture images of type CV_8UC1
	std::map<int, Mat> imgMasks_valid_; // map of camera ID => valid disparity image mask for input texture images of type CV_8UC1
	map<int, Mat> imgMasks_color_; // map of camera ID => image mask for input texture images of type CV_8UC3
	map<int, bool> closeup_xmins_, closeup_xmaxs_, closeup_ymins_, closeup_ymaxs_; // map of camera ID => closeup boolean; true in cases where photo is a close-up that doesn't fully capture the object within the screen space on the indicated side (value assigned by testing for valid masked-in pixels along the appropriate screen space side's edge)
	map<int, Matrix<unsigned int, Dynamic, Dynamic>> segs_; // map of camera ID => image segmentation based on blob application to thresholded imgMask; size is height x width like imgMasks_
	map<int, map<unsigned int, int>> seglabel_counts_; // map of camera ID => unsorted vector of segmentation labels represented in the image in segs_[cid]
	std::map<int, Matrix<int, Dynamic, 1>> masks_int_; // map of camera ID => image mask for input texture images; masks have height*width rows
	std::map<int, Matrix<bool, Dynamic, 1>> masks_; // map of camera ID => image mask for input texture images; masks have height*width rows and value of true for masked-in pixels and false for masked-out pixels; essentially yields used versus un-used pixels
	std::map<int, Matrix<bool, Dynamic, 1>> masks_valid_; // map of camera ID => image mask for input texture images; masks have height*width rows and value of true for masked-in pixels and false for masked-out pixels; dictates which pixels denote valid disparities for epilines and which not (used to build valid disparity ranges); can differ from masks_ to account for occluding background objects in an image that are masked red in the color version
	std::map<int, Matrix<bool, Dynamic, 1>> masks_dilated_; // map of camera ID => dilated image mask for input texture images; masks have height*width rows and 1 col and value of true for masked-in pixels and false for masked-out pixels; masks are dilated for use as approximate masks to accomodate error in camera pose when testing reprojection against masks
	std::map<int, MatrixXf> depth_maps_; // map of camera ID => depth image for input texture images; depth maps are height x width of the associated image
	std::map<int, Matrix3f> Ks_; // map of camera ID => 3x3 camera intrinsics matrix for input images
	std::map<int, Matrix4f> RTs_; // map of camera ID => 4x4 camera extrinsics matrix for input images
	std::map<int, Matrix3f> Kinvs_; // map of camera ID => 3x3 inverse camera intrinsics matrix for input images
	std::map<int, Matrix4f> RTinvs_; // map of camera ID => 4x4 inverse camera extrinsics matrix for input images
	std::map<int, Matrix<float, 3, 4>> Ps_; // map of camera ID => projection matrix that transforms WS to screen space in the associated input camera
	std::map<int, Matrix<float, 4, 3>> Pinvs_; // map of camera ID => inverse projection matrix that transforms screen space in the input camera to WS
	map<int, map<int, Matrix<float, 3, 4>>> Pout2ins_; // map of source camera ID => map of destination camera ID => projection matrix that transforms screen space in the source camera to screen space in the associated destination camera
	std::map<int, Eigen::Matrix<float, Dynamic, 3>> As_; // map of camera ID => BGR image of size (num_pixels, 3)
	std::map<int, Eigen::Matrix<float, Dynamic, 3>> Aunknowns_; // map of camera ID => BGR image of size (num_unknown_disparities_[camera ID], 3) containing only pixels for which disparities are unknown
	std::map<int, int> num_pixels_; // map of camera ID => number of pixels in the image
	std::map<int, int> num_used_pixels_; // map of camera ID => number of pixels used in the image; count excludes pixels masked out but not those already known with high confidence
	std::map<int, int> num_unknown_pixels_; // map of camera ID => number of pixels unknown in the image; count excludes pixels masked out as well as those for which depths/disparities are already known with high confidence
	std::map<int, int> num_known_pixels_; // map of camera ID => number of pixels known in the image; count excludes pixels masked out as well as those for which depths/disparities are unknown with high confidence
	std::map<int, Matrix<int, Dynamic, 1>> used_maps_fwd_; // map of camera ID => pixel index position holds coefficient with index into compact array of used pixels, or -1 for an unused pixel
	std::map<int, Matrix<int, Dynamic, 1>> used_maps_bwd_; // map of camera ID => compact used pixel index position holds coefficient with index into full image index positions
	std::map<int, Matrix<int, Dynamic, 1>> unknown_maps_fwd_; // map of camera ID => pixel index position holds coefficient with index into compact array of unknown pixels, or -1 for a known/unused pixel
	std::map<int, Matrix<int, Dynamic, 1>> unknown_maps_bwd_; // map of camera ID => compact unknown pixel index position holds coefficient with index into full image index positions
	std::map<int, Matrix<bool, Dynamic, 1>> known_depths_; // map of camera ID => pixel index position holds coefficient with true if depth is known to high degree of confidence, false otherwise; masked-out pixels are generally considered unknown (since they generally have depth values of 0.)
	std::map<int, Matrix<bool, Dynamic, 1>> masks_unknowns_; // map of camera ID => image mask that has values of true only where the pixel is masked-in and the depth is unknown, so a combination of masks_ and known_depths_ data
	std::map<int, Matrix<bool, Dynamic, 1>> masks_unknowns_orig_; // map of camera ID => image mask that has values of true only where the pixel is masked-in and the depth is unknown, so a combination of masks_ and known_depths_ data
	std::map<int, Matrix<double, Dynamic, 1>> Xuseds_; // map of camera ID => x coordinates of pixels with used depths only, given in full pixel image index representation
	std::map<int, Matrix<double, Dynamic, 1>>  Yuseds_; // map of camera ID => y coordinates of pixels with used depths only, given in full pixel image index representation
	std::map<int, Matrix<double, Dynamic, 1>> Xunknowns_; // map of camera ID => x coordinates of pixels with unknown depths only, given in full pixel image index representation
	std::map<int, Matrix<double, Dynamic, 1>>  Yunknowns_; // map of camera ID => y coordinates of pixels with unknown depths only, given in full pixel image index representation
	std::map<int, Matrix<unsigned int, Dynamic, 1>> Iunknowns_; // for pixels with unknown depths, column-major indices into full image for compact unknown pixel representation (compact number of them of full-image pixel indices)
	map<int, Matrix<bool, Dynamic, Dynamic>> unknown_disps_valid_; // map of camera ID => used to determine whether reprojection is masked in or out of destination screen space for cameras not otherwise included in stereo reconstruction; first dimension (rows) is unknown pixel's location index among unknown indexing coordinates; second dimension (cols) is the quantized disparity label; data structure applies to the reference image against all other images
	std::map<int, bool> valid_cam_poses_; // map of camera ID => flag indicating whether the camera's pose is considered sufficiently accurate relative to the reference camera
	map<int, Matrix<double, Dynamic, 1>> crowd_disparity_proposal_; // map of camera ID => Matrix<double, Dynamic, 1> crowd_disparity_proposal_: disparity proposal for unknown pixels only of reference camera; known pixels from all other cameras are projected into the reference screen space.All surrounding integer pixel coordinates that are unknown and masked - in receive disparity information from projected pixels, with the largest disparity(smallest depth) winning races; all other pixels receive a value of 0
	map<int, Matrix<double, Dynamic, 1>> disparity_maps_; // map of camera ID => current best disparity map; number of rows equals the number of pixels in the image
	map<int, map<int, Point3f>> meshes_vertices_; // map of camera ID => map of vertex index => x,y,z world space position of vertex
	map<int, map<int, vector<int>>> meshes_vertex_faces_; // map of camera ID => map of vertex index => vector of IDs of faces that contain the vertex
	map<int, map<int, Vec3i>> meshes_faces_; // map of camera ID => map of face index => counter-clockwise ordered vertex indices in the face (triangles only)
	map<int, map<int, Point3f>> meshes_vertex_normals_; // map of camera ID => map of vertex index => normal vector
	map<int, map<int, Point2f>> meshes_texcoords_; // map of camera ID => map of vertex index => texture coordinate; usually, texture coordinates are allowed to vary from face to face for a vertex, but we constrain them here since texture is fronto-parallel projected onto mesh; note that vertex indices and vertex texture coordinate indices are the same and each vertex has only one texture coordinate regardless of face
	map<int, bool> stereo_computed_; // map of camera ID => flag indicating whether stereo reconstruction has been performed on the camera

	// segmentations
	Matrix<float, Dynamic, Dynamic> D_segpln_; // disparity value map structure generated by SegmentPlanar(); size num_unknown_pixels_out_ x b

	// optimization settings
	int max_iters_; // maximum number of optimization iterations, if doesn't converge first
	int average_over_; // number of iterations over which to average when checking convergence
	double converge_; // loop until percentage decrease in energy per loop is less than this value (so, if converge_==101, loop once)

	// stereo output metadata
	Matrix<unsigned short, Dynamic, 1> map_; // height*width x 1; contains iteration number at which each disparity map coefficient was set during optimization
	Matrix<unsigned int, 1, Dynamic> count_updated_vals_; // number of updated values for each iteration (ms)
	Matrix<unsigned int, 1, Dynamic> count_unlabelled_vals_; // number of unlabelled values for each iteration (ms)
	Matrix<unsigned int, 1, Dynamic> count_unlabelled_regions_; // number of independent regions for each iteration (ms)
	Matrix<unsigned int, 1, Dynamic> count_unlabelled_after_QPBOP_; // number of pixels unlabelled after QPBOP for each iteration (ms)
	Matrix<unsigned int, 1, Dynamic> timings_data_term_eval_; // timing of data term evaluation for each iteration (ms) ... (note: not cumulative like for ojw)
	Matrix<unsigned int, 1, Dynamic> timings_smoothness_term_eval_; // timing of smoothness term evaluation for each iteration (ms) ... (note: not cumulative like for ojw)
	Matrix<unsigned int, 1, Dynamic> timings_qpbo_fuse_time_; // timing of qpbo fuse step for each iteration (ms) ... (note: not cumulative like for ojw)
	Matrix<unsigned int, 1, Dynamic> timings_iteration_time_; // total time for each iteration (ms)

	bool init_to_middlebury; // true if initialized to Middlebury cones sequence, false otherwise

	// Constructors and destructor
	StereoData();
	~StereoData();

	// Initialization
	void Init(std::map<int, Mat> imgsT, std::map<int, Mat> imgMasks, std::map<int, Mat> imgMasks_valid, std::map<int, MatrixXf> depth_maps, std::map<int, Matrix3d> Ks, std::map<int, Matrix3d> Kinvs, std::map<int, Matrix4d> RTs, std::map<int, Matrix4d> RTinvs, std::map<int, Matrix<double, 3, 4>> Ps, std::map<int, Matrix<double, 4, 3>> Pinvs, std::map<int, float> min_depths, std::map<int, float> max_depths, std::map<int, std::map<unsigned int, Matrix<unsigned int, Dynamic, 1>>> unknown_segs, std::map<int, float> agisoft_to_world_scales, Matrix4d AgisoftToWorld, Matrix4d WorldToAgisoft, std::vector<int> exclude_cam_ids, int max_num_cams, Scene *scene);
	void Init_MiddleburyCones(int cid_ref);
	void ClearStatistics(int cid_ref);
	void BuildMeshes(int cid_specific = -1); // builds data structures meshes_vertices_, meshes_faces_, and meshes_vertex_normals_
	void BuildTextureCoordinates(int cid_specific = -1); // builds data structure meshes_texcoords_
	void InitPixelData(int cid_specific = -1);// std::map<int, std::map<unsigned int, Matrix<unsigned int, Dynamic, 1>>> unknown_segs); // initializes values in num_used_pixels_, used_map_fwd_, and used_map_bwd_ and many others
	void SpecifyPixelData(int cid, Matrix<bool, Dynamic, 1> *mask_unknowns);
	void DilateMask(int cid); // dilates camera cid's mask to update an entry in masks_dilated_; requires that masks_ and heights_ and widths_ are set

	void CleanDepths(int cid_ref, Matrix<float, Dynamic, 1> *depth_map);
	void CleanDepths_Pair(int cid_src, Matrix<float, Dynamic, 1> *src_depth_map, Matrix<double, Dynamic, 4> *WC_known, Matrix<bool, Dynamic, 1> *known_mask, int cid_dest);

	void CrossCheckValidDisparityRanges(int cid_ref); // attenuates camera cid_ref's valid disparity ranges by projection them into other cameras' valid disparity ranges and culling resulting invalids
	void BuildAllValidDisparityRanges(); // builds data structure unknown_disps_valid_ for camera cid_ref
	void BuildValidDisparityRanges(int cid_ref); // builds data structure unknown_disps_valid_ for camera cid_ref
	void BuildValidDisparityRanges_alt(int cid_ref); // builds data structure unknown_disps_valid_ for camera cid_ref; different implementation
	void BuildCrowdDisparityProposal(int cid_ref); // builds data structure crowd_disparity_proposal_
	void BuildComparisonDisparityProposal(int cid_ref, int cid, Matrix<double, Dynamic, 1> *Dproposal, Matrix<bool, Dynamic, 1> *reproj_known); // creates a disparity proposal (updating Dproposal) that uses disparity values from camera cid1 except where cid projects to cid_ref, in which case it uses the cid's disparities for boolean optimization against cid_ref's current values; reproj_known is updated to hold values reflecting whether reprojected pixel disparities are considered known or unknown for cid (if any pixel neighboring the floating point SS coordinates is known, the disparity is considered known)
	void ReplaceInvalids(int cid_ref, int cid, Matrix<bool, Dynamic, 1> *invalids, Matrix<double, Dynamic, 3> *reproj_ss_pts);

	map<int, int> MapSegmentationLabelsAcrossImages(int cid_src, int cid_dest, Matrix<double, Dynamic, 1> *disparity_map_src); // project points with non-zero, masked-in depth values from cid_src to cid_dest, screen space and give each pixel a "vote" for its label.  Majority vote wins on label mapping, but there is a minimum threshold percentage vote.  Also, if two labels in cid1 both claim to map to the same label in cid2, the one with more votes wins.  Not all labels will be mapped from or to.  The mapping can be used to help determine occlusions when calculating photoconsistency in optimization.  Returns map of label_src => label_dest

	void MaskOutSegmentationOcclusionEdges(int cid); // don't trust depth data on pixels with segmentation value 0 (on the lines) unless all pixels in neighborhood are within reasonable depth distance of each other (ensuring it represents a shared edge and not an occlusion edge), so mask out untrusted pixels

	// Utilities
	static void MaskImg(Mat *imgT, Matrix<bool, Dynamic, Dynamic> *mask, Mat *imgT_masked);
	float GenerateRandomValidDisparity(int cid, int unk_pix_idx); // for camera cid, given an unknown pixel's index (in unknown pixel indexing), generate a random floating point disparity value (snapped to quantized disparity label positions) that is valid with respect to image masks for all other cameras in the scene; if no valid disparity exists (among quantized label positions), the function returns a disparity of 0; requires that unknown_disps_valid_ is built first
	float GenerateValidDisparityAtPerc(int cid, int unk_pix_idx, float perc);
	void SnapDisparitiesToValidRanges(int cid, Eigen::Matrix<double, Dynamic, 1> *disps);  // updates disps so that each relevant value is snapped to the closest valid value for the pixel in camera cid, if one exists; if one doesn't exist, no change is made; if the closest valid range value is a tie, the tie goes to the value closer to the camera (a higher disparity); disps are disparities for all pixels, but the algorithm only considers changing disparities of unknown pixels, assuming known pixels are fine as is
	void SnapDisparityToValidRange(int cid, int idx_unk, double &disp_val); // updates disp_val to snap it to the closest valid value in the reference image for the unknown pixel with contracted unknown space pixel index idx_unk, if one exists; if one doesn't exist, no change is made
	void SnapDisparityToValidRangeAndSmooth(int cid, int idx_unk, double &disp_val, Matrix<double, Dynamic, 1> *disps); // updates disp_val to snap it to the closest valid value in the reference image for the unknown pixel with contracted unknown space pixel index idx_unk, if one exists; if one doesn't exist, no change is made; if possible, also snaps to within GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT WS units of all immediate segment label neighbors
	void PushToSnapNextFartherValidDisparity(int cid, int idx_unk, double &disp_val);
	float GetMaxValidDisparity(int cid, int unk_pix_idx); // uses unknown_disps_valid_ to find the maximum valid disparity value for an unknown pixel; returns 0 if no valid disparity exists for the unknown pixel whose index is given
	float GetMinValidDisparity(int cid, int unk_pix_idx); // uses unknown_disps_valid_ to find the minimum valid disparity value for an unknown pixel; returns 0 if no valid disparity exists for the unknown pixel whose index is given
	int GetMaxValidDisparityLabel(int cid, int unk_pix_idx); // uses unknown_disps_valid_ to find the maximum valid disparity label for an unknown pixel; returns 0 if no valid disparity exists for the unknown pixel whose index is given
	int GetMinValidDisparityLabel(int cid, int unk_pix_idx); // uses unknown_disps_valid_ to find the minimum valid disparity label for an unknown pixel; returns 0 if no valid disparity exists for the unknown pixel whose index is given
	void GetFirstMinMaxValidDisparity(int cid, int unk_pix_idx, float &max, float &min);
	void InverseProjectSStoWS(int cid, Matrix<double, 3, Dynamic> *Iss, Matrix<double, 1, Dynamic> *depths, Matrix<double, 4, Dynamic> *Iws); // like Camera::InverseProjectSStoWS() method, but column-major instead of row-major
	void InverseProjectSStoWS(int ss_width, int ss_height, Matrix<float, Dynamic, Dynamic> *depth_map, Matrix3d *Kinv, Matrix4d *RTinv, Matrix<float, 4, Dynamic> *Iws); // like Camera::InverseProjectSStoWS() method, but column-major instead of row-major
	void ProjectSS1toSS2(int cid_src, int cid_dest, Matrix<double, Dynamic, 1> *disparity_map_src, Matrix<double, Dynamic, 3> *Iss_src, Matrix<double, Dynamic, 3> *Iss_dest); // project points from cid_src's SS to cid_dest's SS, using the disparity map for cid_src disparity_map_src; results are stored with homogeneous SS coordinates in Iss_src and Iss_dest for all masked-in coordinates for which there is disparity information
	Matrix<float, 3, Dynamic> ConstructSSCoordsCM(int ss_w, int ss_h); // returns 3 by (ss_w*ss_h) data structure with all homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h assuming row-major order of indices
	Matrix<double, 3, Dynamic> ConstructSmoothedMaskedInSSCoordsCM(int cid, int ss_w, int ss_h); // returns 3 by (ss_w*ss_h) data structure with all homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h assuming row-major order of indices; smooths pixel locations along segment edges so create smoothed edge lines
	inline float DispLabelToValue(int cid, int disp_label) { return min_disps_[cid] + disp_steps_[cid] * (float)disp_label; }; // returns disparity value corresponding to given label
	int DispValueToLabel(int cid, float disp_val); // returns closest label to given disparity value; labels are truncated to between 1 and the maximum label number

	void SavePointCloud(string scene_name, vector<int> exclude_cids);
	void SaveMeshes(string scene_name, int cid_specific = -1); // like SaveMesh(), but also builds, decimates, cleans, builds texture coordinates for each mesh and can do multiple at once
	void DecimateMeshes(string scene_name, int cid_specific = -1); // downsamples each camera's mesh to have a number of face (approximately) equal to GLOBAL_TARGET_MESH_FACES
	void WriteAgisoftSparsePointCloud(string scene_name, vector<int> exclude_cids);
	void SaveValidRangesPointCloud(string scene_name);
	void SaveMesh(string scene_name, int cid); // mesh must be built first
	void RemoveIsolatedFacesFromMeshes(int cid_specific);

	// Debug / testing functions
	void TestProjectionMatricesByEpilines(int cid_ref); // view epilines of unknown pixels from reference image in each other image
	void DebugViewPoseAccuracy(int cid_ref); // in order to determine the proper dilation element size for masks when constraining depth values, view projections of known pixels from reference image in each other image after dilation to discern overlap visually
	bool TestDisparitiesAgainstMasks(const int cid_ref, Eigen::Matrix<double, Dynamic, 1> *disps, Matrix<bool, Dynamic, 1> *pass); // tests a set of unknown pixel disparity proposals for reference camera cid_ref against the masks for all other cameras to ensure the values are not out of bounds; disps are disparities for unknown pixels only; cid_ref is the camera ID to which the disparities apply; for each pixel, pass is updated to true if a disparity passes the test, false if it fails; returns true if all pass, false if any fail
	void ViewDepthMaps();

	void SyncDepthMaps(Scene *scene, vector<int> cid_order);
	void SmoothDisparityMap(int cid, Matrix<double, Dynamic, 1> *D, int kernel_size = GLOBAL_SMOOTH_KERNEL_SIZE, int smooth_iters = GLOBAL_SMOOTH_ITERS);
	void SmoothDisparityMapSegmentFromTrusted(int cid, int seg_label, Matrix<double, Dynamic, 1> *D, Matrix<bool, Dynamic, Dynamic> *trustMask, int kernel_size = GLOBAL_SMOOTH_KERNEL_SIZE); // apply gaussian smoothing on the given segment seg_label that works as follows: only pixels in the same segment can have an effect on value; also, only pixels with true value in trustMask can have an effect on value; only pixels with a false value in trustMask can be altered
	void SmoothDisparityMap_MeanNonZero(int cid, Matrix<double, Dynamic, 1> *D, int kernel_size = GLOBAL_SMOOTH_KERNEL_SIZE);
	void SmoothDisparityMap_Avg(int cid, Matrix<double, Dynamic, 1> *D, int kernel_size = GLOBAL_SMOOTH_KERNEL_SIZE, int smooth_iters = GLOBAL_SMOOTH_ITERS);
	void SmoothDisparityMaps();

	bool FillSegmentDisparityZeros(int cid, int seg_label, Matrix<double, Dynamic, Dynamic> *D); // fill pixels in segment with label seg_label that have zeros according to disocclusion fill algorithm from Zinger 2010; D is height x width

	void RevertOrigUnknowns(int cid);

	void UpdateDisparityMapFromDepthMap(int cid);
	void UpdateDepthMapFromDisparityMap(int cid);

	void LoadMeshes(string scene_name, int cid_specific = -1);
	int ParseObjFaceVertex(string fv);

	void CleanFacesAgainstMasks(int cid_mesh); // removes camera cid_mesh faces from meshes_faces_[cid_mesh] that violate any camera masks

	// Pixel type transformation functions

	// transforms coefficient index values in A from full image pixel indices to the corresponding compact used pixel indices for camera with ID cid
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> MapFulltoUsedIndices(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<int, Dynamic, 1> mask_rep = used_maps_fwd_[cid].replicate(stacked_num, 1);
		int num_rows_base = used_maps_fwd_[cid].rows();
		for (int i = 2; i <= stacked_num; i++) {
			mask_rep.block(num_rows_base*(i - 1), 0, num_rows_base, 1) = mask_rep.block(num_rows_base*(i - 1), 0, num_rows_base, 1).array() + (i - 1)*num_used_pixels_[cid];
		}
		return EigenMatlab::AccessByIndices(&mask_rep, A).cast<_Tp>();
	}

	// transforms coefficient index values in A from compact used pixel indices to the corresponding full image pixel indices for camera with ID cid
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> MapUsedToFullIndices(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<int, Dynamic, 1> mask_rep = used_maps_bwd_[cid].replicate(stacked_num, 1);
		int num_rows_base = used_maps_bwd_[cid].rows();
		for (int i = 2; i <= stacked_num; i++) {
			mask_rep.block(num_rows_base*(i - 1), 0, num_rows_base, 1) = mask_rep.block(num_rows_base*(i - 1), 0, num_rows_base, 1).array() + (i - 1)*(num_pixels_[cid]);
		}
		return EigenMatlab::AccessByIndices(&mask_rep, A).cast<_Tp>();
	}

	// transforms coefficient index values in A from full image pixel indices to the corresponding compact unknown pixel indices for camera with ID cid
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> MapFulltoUnknownIndices(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<int, Dynamic, 1> mask_rep = unknown_maps_fwd_[cid].replicate(stacked_num, 1);
		int num_rows_base = unknown_maps_fwd_[cid].rows();
		for (int i = 2; i <= stacked_num; i++) {
			mask_rep.block(num_rows_base*(i - 1), 0, num_rows_base, 1) = mask_rep.block(num_rows_base*(i - 1), 0, num_rows_base, 1).array() + (i - 1)*num_unknown_pixels_[cid];
		}
		return EigenMatlab::AccessByIndices(&mask_rep, A).cast<_Tp>();
	}

	// transforms coefficient index values in A from compact unknown pixel indices to the corresponding full image pixel indices for camera with ID cid
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> MapUnknownToFullIndices(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<int, Dynamic, 1> mask_rep = unknown_maps_bwd_[cid].replicate(stacked_num, 1);
		int num_rows_base = unknown_maps_bwd_[cid].rows();
		for (int i = 2; i <= stacked_num; i++) {
			mask_rep.block(num_rows_base*(i - 1), 0, num_rows_base, 1) = mask_rep.block(num_rows_base*(i - 1), 0, num_rows_base, 1).array() + (i - 1)*(num_pixels_[cid]);
		}
		return EigenMatlab::AccessByIndices(&mask_rep, A).cast<_Tp>();
	}

	// expands matrix from compact used pixel size to full image size
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, Dynamic, _cols, _options, _maxRows, _maxCols> ExpandUsedToFullSize(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<bool, Dynamic, _cols> mask_rep = masks_[cid].replicate(stacked_num, A->cols());
		Matrix<_Tp, Dynamic, _cols> B(mask_rep.rows(), mask_rep.cols());
		B.setZero();
		EigenMatlab::AssignByTruncatedBooleans(&B, &mask_rep, A);

		return B;
	}

	// truncate matrix from full image size to compact used pixel size
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> ContractFullToUsedSize(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<bool, Dynamic, 1> mask_rep = masks_[cid].replicate(stacked_num, 1);
		return EigenMatlab::TruncateByBooleansRows(A, &mask_rep);
	}

	// expands matrix from compact used pixel size to full image size
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, Dynamic, _cols, _options, _maxRows, _maxCols> ExpandUnknownToFullSize(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<bool, Dynamic, _cols> mask_rep = masks_unknowns_[cid].replicate(stacked_num, A->cols());
		Matrix<_Tp, Dynamic, _cols> B(mask_rep.rows(), mask_rep.cols());
		B.setZero();
		EigenMatlab::AssignByTruncatedBooleans(&B, &mask_rep, A);

		return B;
	}

	// truncate matrix from full image size to compact used pixel size
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> ContractFullToUnknownSize(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<bool, Dynamic, 1> mask_rep = masks_unknowns_[cid].replicate(stacked_num, 1);
		return EigenMatlab::TruncateByBooleansRows(A, &mask_rep);
	}

	// expands matrix from compact used pixel size to full image size
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, Dynamic, _cols, _options, _maxRows, _maxCols> ExpandUnknownToUsedSize(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<_Tp, _rows, _cols> Bfull = ExpandUnknownToFullSize(cid, A, stacked_num);
		Matrix<_Tp, _rows, _cols> Bused = ContractFullToUsedSize(cid, &Bfull, stacked_num);
		return Bused;
	}

	// truncate matrix from full image size to compact used pixel size
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	inline Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> ContractUsedToUnknownSize(int cid, const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, int stacked_num = 1) {
		Matrix<_Tp, _rows, _cols> Bfull = ExpandUsedToFullSize(cid, A, stacked_num);
		Matrix<_Tp, _rows, _cols> Bunk = ContractFullToUnknownSize(cid, &Bfull, stacked_num);
		return Bunk;
	}

};

#endif
