#ifndef StereoReconstruction_H
#define StereoReconstruction_H

#include "Globals.h"
#include "Calibration.h"
#include "Sensor.h"
#include "DepthMap.h"
#include "Camera.h"
#include "Scene.h"
#include "Interpolation.h"
#include "seg_ms/msImageProcessor.h"

/*
	input images' projection matrices should be relative to output image's projection matrix

	use range of depths in scene +20% on either side
	to determine number of discretized disparity labels, project pixel 0,0 into all views and ensure that minimum space between samples is 0.5 pixels

	prior: planar ... planar = true
	smoothness kernel: truncated linear ... smoothness_kernel = 1
	graph connectivity: 4 connected (bi-directional) ... connect = 4
	optimization algorithm: QPBO-R ... improve = 2
	proposal method: smooth* ... proposal_method = 3

*/

//Stereo Depth Reconstruction
class StereoReconstruction {

private:
	
	Matrix<unsigned int, Dynamic, Dynamic> seg_labels_; // segmentation labels for imgT_out_; set using mean-shift
	Matrix<float, Dynamic, 3> R_; // output image reshaped to rows=2*width*height and one column for each color channel; a 2-dimensional matrix where the second dimension (columns) has size equal to the number of color channels per pixel. The number of rows equals imgT_out's rows*cols*2, where all values are repeated vertically, as in Matlab's repmat[2 1]
	Matrix<unsigned int, 3, Dynamic> SEI_; // Mx3 uint32 array of smoothness clique indices; additional explanation: 3xi (i==((rows)*(cols-2)+(rows-2)*(cols)) of output image) matrix of pixel location indices displaying across its rows connectivity of the output image (first vertical connectivity, then horizonal connectivity); form necessary for 2nd order smoothness prior
	Matrix<int, Dynamic, 1> EW_; // data structure of smoothness edges that don't cross segmentation boundaries
	Matrix<float, Dynamic, Dynamic> D_; // disparity map structure

	// Initialization for stereo computations
	void InitDisps(); // initialize disp_labels_ data structure and min_disp_, max_disp_, and disp_step_ using args min_disp, max_disp and disp_step
	inline double DispLabelToValue(int disp_label) { return min_disp_ + disp_step_*(double)disp_label; }; // returns disparity value corresponding to given label
	int DispValueToLabel(double disp_val); // returns closest label to given disparity value; labels are truncated to between 1 and the maximum label number
	void TransformReferenceFrame(); // transforms the reference frame and sets the extrinsics matrices and output extrinsics matrix such that the output camera extrinsics matrix is the identity
	void ComputeMeanShiftSegmentation(); // segments output texture image imgT_out_ using mean-shift and places results in seg_labels_
	void InitEW(); // initializes data structure EW_, which identifies smoothness edges that don't cross segmentation boundaries

	// Segmentation
	void SegmentPlanar(); // generate piecewise-planar disparity proposals for stereo depth reconstruction

	// Optimization
	Matrix<float, Dynamic, Dynamic> Stereo_optim(GLOBAL_PROPOSAL_METHOD proposal_method); // tiven a binary energy function for a stereo problem, and a set of proposals(or proposal index), fuses these proposals until convergence of the energy
	inline int DisparityProposalSelectionUpdate_SameUni(int n) { return 1; };
	inline int DisparityProposalSelectionUpdate_SegPln(int n) {  }; // fix this
	inline int DisparityProposalSelectionUpdate_SmoothStar(int n) { return ((n - 1) % 6) + 1; };


public:

	// disparity members
	double min_depth_; // minimum depth value for scene
	double max_depth_; // maximum depth value for scene
	double min_disp_; // minimum disparity value for scene
	double max_disp_; // maximum disparity value for scene
	double disp_step_; // change in disparity value between adjacent disparity labels
	int num_disps_; // number of disparities
	ArrayXd disps_; // num_disps_ array of descending disparities to sample at in rendering
	
	// scene and output members
	int cid_out_; // ID of reference (output) camera
	int height_; // height in pixels of output display
	int width_; // width in pixels of output display
	std::map<int, Mat> imgsT_; // map of camera ID => image for input texture images of type CV_8UC3
	std::map<int, Mat> imgsD_; // map of camera ID => depth image for input texture images of type CV_8UC1
	Matrix3d K_; // camera intrinsics matrix
	Matrix3d Kinv_; // inverse of camera intrinsics matrix
	map<int, Matrix<double, 3, 4>> RTs_; // map of camera ID => 4x4 camera extrinsics matrix for input images; to be transformed into coordinate system centered on output camera
	map<int, Matrix<double, 3, 4>> Ps_; // map of camera ID => projection matrix that transforms screen space in the output camera to screen space in the associated input camera

	// energy parameters
	double disp_thresh_;
	double col_thresh_; // scalar noise parameter for data likelihood
	double occl_const_; // scalar occlusion cost
	double occl_val_; // scalar penalty energy cost for occluded pixels
	int lambda_l_; // scalar smoothness prior weight for cliques crossing segmentation boundaries
	int lambda_h_; // scalar smoothness prior weight for cliques not crossing segmentation boundaries

	// parameters for the mean-shift over-segmentation of the reference image
	int ms_seg_sigmaS_;
	float ms_seg_sigmaR_;
	int ms_seg_minRegion_;

	// optimization settings
	bool visibility_; // use geometric visbility constraint
	bool compress_graph_; // compression makes graph smaller, but is slower
	int max_iters_; // maximum number of iterations, if doesn't converge first
	double converge_; // loop until percentage decrease in energy per loop is less than this value (so, if converge_==101, loop once)
	int average_over_; // number of iterations over which to average when checking convergence
	bool independent_; // use strongly-connected, rather than independent, regions
	double window_; // half-size of window to use in window matching
	
	// IBR fuse depths settings
	double visibility_val_;

	// Constructors / destructor
	StereoReconstruction();
	~StereoReconstruction();

	void Init(std::map<int, Mat*> imgsT, std::map<int, Mat*> imgsD, std::map<int, Matrix<double, 3, 4>> RTs, double min_depth, double max_depth);

	Matrix<float, Dynamic, Dynamic> Stereo(int cid_out, Matrix3d K); // updates disparity map of size sz_out from perspective of camera with calibration K and extrinsics RT


};

#endif