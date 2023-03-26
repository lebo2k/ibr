#ifndef NVS_H
#define NVS_H

#include "Globals.h"
#include "Calibration.h"
#include "Sensor.h"
#include "DepthMap.h"
#include "Camera.h"
#include "Scene.h"

// New view synthesis
class NVS {

private:
	
	void NVS::Blend(std::map<int, Mat> imgTs, std::map<int, Mat> imgDs, std::map<int, Mat> imgMs, std::vector<int> ordered_cam_ids, Mat *imgTresult, Mat *imgDresult, Mat *imgMresult); // blends input new views to a single consolidated new view, putting the resulting texture image in imgT, the associated resulting depth image in imgD, and the associated mask imgM for opaque versus transparent values

public:

	Scene *scene_;

	// Constructors / destructor
	NVS(std::string agisoft_filename); // agisoft_filename holds the filename for the Agisoft scene data
	~NVS();

	// performs new view synthesis to create an image from a camera with pose given by camera extrinsics matrix RT, with intrinsics given by the calibration calib, and potentially adjusted screen space pixel size view_size
	Mat SynthesizeView(Calibration *calib, Mat *RT, cv::Size view_size, std::vector<int> exclude_cam_ids, int num_cams = 0);

};

#endif