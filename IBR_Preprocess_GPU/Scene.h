#ifndef Scene_H
#define Scene_H

#include "Globals.h"
#include "Sensor.h"
#include "Camera.h"
#include "Interpolation.h"

/*
Note that to get location of camera in chunk space, use RTinv in Agisoft's chunk space = AgisoftToWorldinv_ * RTinv in Agisoft's world space, then read off the right column's first 3 values
So, RT in Agisoft's chunk space = RT in Agisoft's world space * AgisoftToWorld_
*/

class Scene {

private:
	
	void InitCamerasReprojection(); // initialize cameras for reprojection
	void CullCameras(); // deletes cameras without extrinsic matrices and depth maps
	std::vector<int> OrderCams(Point3d I0_pos, Point3d I0_view_dir, bool cull_backfacing = true); // returns an ordered list of camera IDs from member cameras according to the closest view to I0 (by view direction, not position); if cull_backfacing, ensures camera is looking in the same general direction as the given view_dir; otherwise does not
	void CleanDepths_Pair(int cid_src, Matrix<double, Dynamic, 4> *WC_known, Matrix<bool, Dynamic, 1> *known_mask, int cid_dest); // given source and destination camera IDs, updates the source camera's depth map so that any of the source camera's world space points, when reprojected into the destination camera's screen space, that falls on a masked out pixel has its corresponding depth value in the source camera's depth map set to zero to signify an unknown depth (since the depth we had for that pixel was found to be incorrect)
	void GenerateUnknownSegmentations(); // generates all segmentation label images
	void GenerateUnknownSegmentations_img(int cid, Mat *img); // generates a segmentation label image
	void RecordUnknownSegmentationLabels(int cid, IplImage* labelImg);

	void FilterCamerasByPoseAccuracy(); // camera poses may not be correct, so test input camera poses against reference camera: projections of reference camera pixels into each input camera's screen space should land inside masked pixels within an acceptable error factor, represented by applying a morphological dilation to the input camera mask with element size given by GLOBAL_MASK_DILATION; any failing cameras and their data are given a false flag for posed_; if a majority of the input cameras fail the test, the reference camera is considered to have a bad pose (majority rules for pose estimation)...otherwise, the cameras with false in valid_cam_poses are assumed to have bad poses; if exactly half pass, then assume only the group of cameras that include the lowest cid pass so we don't have two groups that do not match up well when rendering


public:

	std::string name_; // the name of the scene; must also be the name of the folder containing the scene's data
	std::map<int, Sensor*> sensors_;
	std::map<int, Camera*> cameras_;
	float depth_downscale_; // downsampling factor applied to camera images when computing depth map images
	Matrix4d AgisoftToWorld_; // transform from Agisoft space to the world space we've defined in the chunk containing our scene in the Agisoft UI; necessary because Agisoft's camera extrinsic matrices and depth map values are in Agisoft's default space, not the transformed chunk space that takes effect once the coordinate system is updated by setting markers, entering positions for them, and updating the coordinate system
	Matrix4d WorldToAgisoft_; // inverse of AgisoftToWorld_
	double agisoft_to_world_scale_; // scale factor associated with AgisoftToWorld_

	// Depth and disparity members
	Point3d bv_min_, bv_max_; // minimum and maximum bounding volume coordinates for the point cloud in the scene captured by all input images ... used to determine minimum and maximum depth bound in camera space
	std::map<int, float> min_depths_; // map of camera ID => minimum depth value for scene in camera space, so it's in Agisoft units, from the perspective of the camera with the given ID as the reference camera
	std::map<int, float> max_depths_; // maximum depth value for scene in camera space, so it's in Agisoft units, from the perspective of the camera with the given ID as the reference camera
	std::map<int, float> min_disps_; // minimum disparity value for scene in camera space, from the perspective of the camera with the given ID as the reference camera
	std::map<int, float> max_disps_; // maximum disparity value for scene in camera space, from the perspective of the camera with the given ID as the reference camera
	std::map<int, float> disp_steps_; // change in disparity value between adjacent disparity labels, from the perspective of the camera with the given ID as the reference camera; max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp
	std::map<int, int> num_disps_; // number of disparities, from the perspective of the camera with the given ID as the reference camera
	std::map<int, ArrayXd> disps_; // num_disps_ array of descending disparities to sample at in rendering, from the perspective of the camera with the given ID as the reference camera
	std::map<int, std::map<unsigned int, Matrix<unsigned int, Dynamic, 1>>> unknown_segs_; // map of camera ID => label for label segmentation of the image into blobs of pixels with unknown depths => pixels with unknown depths belonging to this label

	std::map<int, Matrix<bool, Dynamic, 1>> masks_dilated_; // map of camera ID => dilated image mask for input texture images; masks have height*width rows and 1 col and value of true for masked-in pixels and false for masked-out pixels; masks are dilated for use as approximate masks to accomodate error in camera pose when testing reprojection against masks
	
	// Constructors / destructor
	Scene();
	~Scene();

	void Scene::Init(std::string name);

	std::vector<int> GetClosestCams(Point3d I0, Point3d I0_view_dir, std::vector<int> exclude_cam_ids, int num = 0, bool cull_backfacing = true); // gets an ordered list of num closest cameras to view of I0 (closest first); if cull_backfacing, ensures camera is looking in the same general direction as the given view_dir; otherwise does not
	
	void UpdatePointCloudBoundingVolume(); // determines world space point cloud bounding box volume across all camera views and updates bv_min_ and bv_max_
	void UpdateCSMinMaxDepths(); // computes min and max depths for all cameras' points projected into each camera's CS as a reference camera in turn; updates min_depths_ and max_depths_
	void UpdateDisparities(); // need version here so that can save disparity information in asset bundle for export since are saving quantized disparities that must later be converted to depth values
	void UpdateCamerasWSPoints(); // update Iws_ for all cameras based on current depth data

	// I/O
	void SaveRenderingDataRLE();
	void LoadRenderingDataRLE();

	void UpdateDepthMapsFromStereoData(map<int, MatrixXf> depth_maps);
	void UpdateDepthMapsAndMasksFromStereoData(map<int, MatrixXf> depth_maps, map<int, Matrix<bool, Dynamic, 1>> masks, map<int, int> heights, map<int, int> widths); // masks also important in case expanded a mask because a pixel has no valid quantized disparity labels
	
	void Print(); // debug printing

	void TestMinMaxDepths();
	
	void CleanDepths(); // tests world space coordinates from each camera against masks from all other input cameras.  any world space point that, when reprojected into another camera space, does not fall on a pixel with a true value in that camera space is considered to have a rejected depth value.  the corresponding pixel in the corresponding depth map is assigned a depth of 0 to signify that we have no valid information for that pixel

	void UpdateDepthsByCrowdWisdom(); // updates depth map data for camera cid_ref using depth data from all other cameras; smallest cid_ref camera space depth wins races; 0 depths automatically overwritten

	Matrix<float, 3, Dynamic> ConstructSSCoordsCM(int ss_w, int ss_h, Mat* imgMask, Matrix<float, Dynamic, Dynamic> *depth_map);
	void InverseProjectSStoWS(int ss_width, int ss_height, Mat* imgMask, Matrix<float, Dynamic, Dynamic> *depth_map, Matrix3d *Kinv, Matrix4d *RTinv, Matrix<float, 4, Dynamic> *Iws);

	void DebugEpipolarGeometry();
	void DebugViewPoseAccuracy(int cid_ref);

	bool SimulateRegularMeshPixelColor(int cid, double xpos, double ypos, Vec3b &color); // returns true if a pixel color could be retrieved, false otherwise; updates color with the value if one is retrieved; color may not be retrieved if a valid simulated triangle exists at that position given where pixels are not masked and they don't vary in depth more than allowed for a single triangle

	void ExportDepthMapsEXR(string filepath);

	void SyncDepthMaps();

	void ExportSceneInfo(); // writes a file that, for each camera, exports camera ID, WS position, WS view direction, and WS up direction; also exports bounding volume info

	void ExportMaskedCameraImages(); // exports each camera's image after masking it
	
	void CleanFacesAgainstMasks(int cid_mesh);

};

#endif