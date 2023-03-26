#ifndef Camera_H
#define Camera_H

#include "Globals.h"
#include "Sensor.h"
#include "Calibration.h"
#include "DepthMap.h"

class Camera {

private:

	// Members
	Mat imgDdisc_; // type CV_8UC1 binary image of size width_, height_ corresponding to img_, with val=255 for pixels where there is a high depth discontinuity and 0 everywhere else
	Mat Iws_; // 4xn matrix of type CV_32F of ordered, columnated, world space homogeneous locations (x,y,z,1) of screen space points; the order of points in Iws_ corresponds to the related pixel, where pixel order is given by index = (row * width) + col

	// Initialization
	void ParseAgisoftCameraExtrinsics(std::string s, Mat *AgisoftToWorld_, Mat *AgisoftToWorldinv_); // parse string of 16 ordered doubles (in col then row order) into R_ and T_ matrices for camera extrinsics; it expects the transform being parsed is a camera location in world space, and therefore represents RTinv; see Scene.h for description of AgisoftToWorld
	static Mat ConstructSSCoords(int ss_w, int ss_h); // returns 3 by (ss_w*ss_h) data structure of type CV_32F with homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h
	void InitDepthMapDiscontinuities(); // called by InitDepthMap() to initialize imgDMDisc_
	bool GetNearestDepth(Point p, Point dir, int &dist, float &depth); // used by InpaintDepthMap(); traverses depth image imgD_ from p in direction step until encounters a pixel with a non-zero depth value that is not masked out in imgMask
	void InpaintDepthMap(); // inpaints depth map by searching from each missing pixel (u,v) in 8 canonical directions until reach a pixel with a depth value (ud, pd), then perform a weighted interpolation
	bool DetermineDepthMapDiscontinuity(Point p); // tests pixel at point p for high depth map discontinuity and returns result boolean

	// Update functions
	void UpdateCameraMatrices(); // updates P_, Pinv_, RT_, RTinv_, Kinv_; requires that K_, R_, and T_ are set
	void UpdatePos(); // updates member pos_
	void UpdateViewDir(); // updates member view_dir_

	// Convenience functions
	static inline int PixIndexFwd(Point pt, int width) { return (pt.y * width) + pt.x; }; // forward computation of index into pixel position data structures from pixel pt in image img_
	static inline Point PixIndexBwd(int idx, int width) { Point pt; pt.y = std::floor(idx / width); pt.x = idx - pt.y*width; return pt; }; // backward computation of pixel pt in image img_ from index into pixel position data structures
	bool GetTransformedPosSS(Mat *PosSS, Point pt_ss, cv::Size sizeTargetSS, Point &pt_ss_transformed); // given pointer to 3xn matrix PosSS that holds tranformed screen space positions where n is the number of pixels in width_*height_ and the order is column then row, updates the rounded transformed screen space coordinates for a given screen space pixel position and returns boolean whether is inside the target screen space or not
	Point3d Camera::GetTransformedPosWS(Mat *PosWS, Point pt_ss); // given pointer to 4xn matrix PosWS that holds tranformed world space positions where n is the number of pixels in width_*height_ and the order is column then row, returns the transformed world space coordinates for a given screen space pixel position

	// I/O
	char* GetFilename(std::string filepath, std::string scene_name);
	char* GetFilenameImage(std::string filepath, std::string scene_name, char* fn_img_chars);
	char* GetFilenameMatrix(std::string filepath, std::string scene_name, char* fn_mat_chars);

public:

	// Members
	int id_; // a unique identifier for this camera
	std::string fn_; // image filename
	std::string fn_mask_; // image mask filename
	int sensor_id_; // ID of associated sensor
	Mat imgT_; // this photo; type CV_8UC3
	Mat imgMask_; // binary image mask of type CV_8UC1 ("opaque" foreground with value 255, and "transparent" background with value 0)
	Mat imgD_; // depth map image of type CV_32F with same intrinsics (and extrinsics) as camera and img_
	Mat imgDinterpMask_; // binary image mask of type CV_8UC1 with values of 255 where imgD_ values were interpolated through inpainting and 0 everywhere else
	Mat RT_; // 4x4 camera extrinsics matrix; type CV_64F; [R | t] -- [0 | 1]
	Mat RTinv_; // 4x4 inverse camera extrinsics matrix; type CV_64F
	Mat P_; // 3x4 projection matrix, including intrinsics and extrinsics, where P=K[R|T]; type CV_64F; converts WS to SS
	Mat Pinv_; // 4x3 inverse projection matrix; type CV_64F; converts SS to WS
	Point3d pos_; // position of camera in world space
	Point3d view_dir_; // view direction of camera in world space
	int width_, height_; // image width and height in pixels
	Calibration calib_;
	GLOBAL_AGI_CAMERA_ORIENTATION orientation_;
	bool enabled_; // whether the camera is enabled in the Agisoft file; it is expected that any disabled camera has no depth map, but we do not rely on that fact in the code
	DepthMap *dm_;
	bool posed_; // true if pose has been calculated, false otherwise
	bool has_depth_map_; // true if the camera has a depth map, false otherwise


	// Constructors / destructor
	Camera();
	~Camera();

	// Initialization
	void Init(xml_node<> *camera_node, Mat *AgisoftToWorld_, Mat *AgisoftToWorldinv_); // see Scene.h for description of AgisoftToWorld
	void Camera::InitSensor(Sensor *sensor); // initializes projection matrices P and Pinv using the camera intrinsics matrix arg K
	void InitDepthMap(xml_node<> *depthmap_node, double agisoft_to_world_scale_, int depth_downscale); // initializes imgDM_ and calls InitDepthMapDiscontinuities(); depth downscale is the downward scale factor as given by Agisoft for the depth map from the original image; agisoft_to_world_scale_ is the scale factor associated with the change from Agisoft's space to our world space, given by the chunk transform in the xml file from Agisoft
	void InitWorldSpaceProjection(); // initializes Iws_ by computing the world space location that corresponds to each pixel in camera screen space
	void Downsample(float scale_factor, bool include_depth_map=true); // downsamples images and updates resolution info, camera intrinsics, and projection matrices accordingly; if include_depth_map flag is TRUE, also downsamples the associated depth map
	void Downsample(int target_width, int target_height, bool include_depth_map = true); // downsamples images and updates resolution info, camera intrinsics, and projection matrices accordingly; if include_depth_map flag is TRUE, also downsamples the associated depth map
	
	void UndistortPhotos(); // undistorts imgT_, imgD_, and imgMask_ images to correct for radial and tangential distortion

	// Warping
	static void InverseProjectSStoWS(int ss_width, int ss_height, Mat *imgD, Mat *Kinv, Mat *RTinv, Mat *Iws); // inverse projects screen space points (screen space dimensions ss_width x ss_height) with depths given by imgD from screen space to world space using Kinv and RTinv, returning a 4xn matrix of type 64F of the corresponding points in world space
	void Reproject(Mat *P_dest, Mat *imgT, Mat *imgD, Mat *imgMask); // reprojects the camera view into a new camera with projection matrix P_dest; only reprojects pixels for which there is depth info; imgT is modified to include texture, imgD to include depth values, and imgMask to include binary mask values (255 pixel is opaque and 0 pixel is transparent), and all must be same size and types

	// Static convenience functions
	static Point RoundSSPoint(Point2d ptd, int width, int height); // rounds the position of a sub-pixel point in screen space to an integer pixel point in screen space
	static Point3d GetCameraPositionWS(Mat *RTinv); // returns camera position in world space using RTinv inverse extrinsics matrix from argument
	static Point3d GetCameraViewDirectionWS(Mat *RTinv); // // returns camera view direction in world space using RTinv inverse extrinsics matrix from argument
	static Mat Extend_K(Mat *K); // converts 3x3 camera intrinsics matrix to 3x4 version with right column of [0 0 0]T
	static Mat Extend_Kinv(Mat *Kinv); // converts 3x3 inverse camera intrinsics matrix to 4x3 version with bottom row [0 0 1]
	static void ComputeProjectionMatrices(Mat *K, Mat *Kinv, Mat *RT, Mat *RTinv, Mat &P, Mat &Pinv); // updates P and Pinv to be 4x4 projection and inverse projection matrices, respectively, from camera intrinsics K and extrinsics RT
	void DeterminePointCloudBoundingVolume(Point3d &bv_min, Point3d &bv_max); // returns bounding volume around the world space point cloud Iws_ and updates bv_min and bv_max

	// I/O
	void Save_Iws(std::string scene_name);
	void Load_Iws(std::string scene_name);

	// Debugging
	void Print(); // debug printing
};

#endif