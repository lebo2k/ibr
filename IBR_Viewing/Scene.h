#ifndef Scene_H
#define Scene_H

#include "Globals.h"
#include "Sensor.h"
#include "Camera.h"

/*
Note that to get location of camera in chunk space, use RTinv in Agisoft's chunk space = AgisoftToWorldinv_ * RTinv in Agisoft's world space, then read off the right column's first 3 values
So, RT in Agisoft's chunk space = RT in Agisoft's world space * AgisoftToWorld_
*/

class Scene {

private:
	
	void InitCamerasReprojection(); // initialize cameras for reprojection
	void CullCameras(); // deletes cameras without extrinsic matrices and depth maps
	std::vector<int> OrderCams(Point3d I0_pos, Point3d I0_view_dir); // returns an ordered list of camera IDs from member cameras according to the closest view to I0

public:

	std::string name_;
	std::map<int, Sensor*> sensors_;
	std::map<int, Camera*> cameras_;
	float depth_downscale_; // downsampling factor applied to camera images when computing depth map images
	Mat AgisoftToWorld_; // transform from Agisoft space to the world space we've defined in the chunk containing our scene in the Agisoft UI; necessary because Agisoft's camera extrinsic matrices and depth map values are in Agisoft's default space, not the transformed chunk space that takes effect once the coordinate system is updated by setting markers, entering positions for them, and updating the coordinate system
	Mat AgisoftToWorldinv_; // inverse of AgisoftToWorld_
	double agisoft_to_world_scale_; // scale factor associated with AgisoftToWorld_

	// Constructors / destructor
	Scene();
	~Scene();

	void Scene::Init(std::string agisoft_filename);

	std::vector<int> GetClosestCams(Point3d I0, Point3d I0_view_dir, std::vector<int> exclude_cam_ids, int num = 0); // gets an ordered list of num closest cameras to view of I0 (closest first)
	void DeterminePointCloudBoundingVolume(Point3d &bv_min, Point3d &bv_max); // determines point cloud bounding box volume across all camera views and updates bv_min and bv_max

	// I/O
	void SaveCamerasWSPointProjections(); // saves Iws_ for each camera

	void Print(); // debug printing
};

#endif