#ifndef DepthMap_H
#define DepthMap_H

#include "Globals.h"
#include "Calibration.h"

// OpenEXR includes
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfArray.h>
#include <ImfInputFile.h>
#include "namespaceAlias.h"
using namespace IMF;
using namespace IMATH_NAMESPACE;
//#include <ImfRgbaFile.h>
//#include <ImfOutputFile.h>
//#include <ImfChannelList.h>

/*
	Note on differences in intrinsics matrix K between a camera and its associated depth map:
	Depth map screen space is scaled by a factor w from camera screen space (believed to be always the same in both directions, though we don't count on that in this code where we determine wx and wy)
	fx_dm = fx_cam / wx
	fy_dm = fy_cam / wy
	cx_dm = cx_cam / wx
	cy_dm = cy_cam / wy
	width_dm = width_cam / wx
	height_dm = height_cam / wy
	cx_dm / width_dm = cx_cam / width_cam
	cy_dm / height_dm = cy_cam / height_cam
*/

/*
	Note that depth values given by Agisoft are in Agisoft's base space, not the world space we'd like that we defined in our Agisoft chunk.  The reason for that is that the camera intrinsics from Agisoft contain fx=f*kx and fy=f*ky, where kx and ky are scale factors that convert pixels to millimeters (not meters because focal length f is in mm and the distance units here cancel out so don't survive to interact with the world space's meters units).

	We leave the depth values here in Agisoft's units, but save the scaling factor agisoft_to_world_scale_ as a class member in case need to access our world ("chunk") space depths later.  The reason is that the depth values are used by Camera::Reproject() in camera space, not world space, so extrinsics have not been applied to convert from Agisoft space to world space, and therefore the Agisoft depth values should be used there.
*/

class DepthMap {

private:

	void readGZ1(const char fileName[], Array2D<float> &zPixels, int &width, int &height); // reads data from a single-channel .exr depth file into array zPixels and sets corresponding width and height values
	float DepthVal(Array2D<float> zPixels, int x, int y); // retrieves a depth value from array zPixels at location (x,y)
	void readHeader(const char fileName[]); // reads the header information from a .exr file named in the argument

public:
	
	std::string fn_; // image filename, including any path used by Agisoft
	Mat imgD_; // this photo; type CV_32F
	int cam_id_; // associated camera ID
	Calibration calib_;
	double max_depth_, min_nonzero_depth_; // max depth and min non-zero depth in Agisoft space unit values
	double max_depth_ws_, min_nonzero_depth_ws_; // max depth and min non-zero depth in world space unit values
	int width_, height_;
	int depth_downscale_; // downward scale factor as given by Agisoft for the depth map from the original image
	double agisoft_to_world_scale_; // agisoft_to_world_scale is the scale factor associated with the change from Agisoft's space to our world space, given by the chunk transform in the xml file from Agisoft

	// Constructors / destructor
	DepthMap();
	~DepthMap();

	void Init(xml_node<> *depthmap_node, double agisoft_to_world_scale, int depth_downscale); // set member values according to data in node argument from Agisoft doc.xml file; depth downscale is the downward scale factor as given by Agisoft for the depth map from the original image; agisoft_to_world_scale is the scale factor associated with the change from Agisoft's space to our world space, given by the chunk transform in the xml file from Agisoft
	static void DisplayDepthImage(Mat *img32F); // displays a grayscale visualization of the depth map image
	Mat GetDepthMapInWorldSpace(); // returns the depth map image converted to world space (since is stored in Agisoft space to make reprojection faster and easier)

	void Print(); // debug printing

};

#endif