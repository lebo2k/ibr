#ifndef NVS_H
#define NVS_H

#include "Globals.h"
#include "Calibration.h"
#include "Sensor.h"
#include "DepthMap.h"
#include "Camera.h"
#include "Scene.h"
#include "StereoData.h"
#include "DisplayImages.h"

// New view synthesis
class NVS {

private:
	

	void Composite(std::map<int, Mat> imgTs, std::map<int, MatrixXf> imgDs, std::map<int, Mat> imgMs, std::vector<int> ordered_cam_ids, Mat *imgTresult, Mat *imgDresult, Mat *imgMresult, Mat *imgMprimary); // composites input new views to a single consolidated new view, putting the resulting texture image in imgT, the associated resulting depth image in imgD, and the associated mask imgM for opaque versus transparent values
	void Sharpen(Mat *imgT, Mat *imgD, Mat *imgM, std::vector<int> cam_ids, Matrix<float, 3, 4> Pvirtual, Point3d view_dir_virtual);
	float Etexture(Matrix<int, Dynamic, 3> patch1, Matrix<int, Dynamic, 3> patch2, int center_pixel_idx);
	float Ephoto(Vec3b color, map<int, Vec3b> sampled_colors);
	float EphotoMinZ(Vec3b color, map<int, map<int, Vec3b>> sampled_colors_across_depths, int &depth_idx_minz); // updates depth_idx_minz to hold the depth_idx at which Ephoto was minimized
	Vec3f Gradient_Ephoto(Vec3b color, map<int, Vec3b> sampled_colors);
	Vec3f Gradient_EphotoMinZ(Vec3b color, map<int, map<int, Vec3b>> sampled_colors_across_depths); // gradient of a f(x,y) = min(x,y) is
	inline bool pairCompareModes(const std::pair<float, Vec3b>& firstElem, const std::pair<float, Vec3b>& secondElem) { return firstElem.first < secondElem.first; };
	void DisplayEpigraph(map<int, map<int, Vec3b>> sampled_colors_across_depths, int max_cam_id);
	Vec3b AvgColors(map<int, Vec3b> colors);

public:

	Scene *scene_;
	StereoData *sd_;

	Calibration calib_;
	cv::Size view_size_;

	map<int, int> heights_; // height in pixels of output display
	map<int, int> widths_; // width in pixels of output display
	std::map<int, Mat> imgsT_; // map of camera ID => image for input texture images of type CV_8UC3
	std::map<int, Matrix<bool, Dynamic, 1>> masks_; // map of camera ID => image mask for input texture images; masks have height*width rows and value of true for masked-in pixels and false for masked-out pixels; essentially yields used versus un-used pixels
	map<int, Matrix<double, 3, 4>> Ps_; // map of camera ID => 3x4 projection matrix, including intrinsics and extrinsics, where P=K[R|T]; converts WS to SS
	map<int, Matrix<double, 4, 3>> Pinvs_; // map of camera ID => 4x3 inverse projection matrix; converts SS to WS
	map<int, Matrix<int, Dynamic, Dynamic>> As_blue_;
	map<int, Matrix<int, Dynamic, Dynamic>> As_green_;
	map<int, Matrix<int, Dynamic, Dynamic>> As_red_;

	// Constructors / destructor
	NVS();
	~NVS();

	void Init(Scene *scene, StereoData *sd, Calibration *calib, cv::Size view_size);

	// performs new view synthesis to create an image from a camera with pose given by camera extrinsics matrix RT, with intrinsics given by the calibration calib, and potentially adjusted screen space pixel size view_size
	Mat SynthesizeView(Mat *RTmat, std::vector<int> exclude_cam_ids, int num_cams = 0);

};

#endif