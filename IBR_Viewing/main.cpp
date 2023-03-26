#include "Globals.h"
#include "NVS.h"
#include "StereoReconstruction.h"
#include "ArcBall.h"

// User Defined Variables
cv::Size sz;
Mat imgT;
NVS *nvs;
Calibration calib;
std::vector<int> exclude_cam_ids;
Mat RT = cv::Mat::zeros(4, 4, CV_64F);
Mat RTinv = cv::Mat::zeros(4, 4, CV_64F);

const float PI2 = 2.0*3.1415926535f;								// PI Squared

Matrix4fT   TransformAB = { 1.0f, 0.0f, 0.0f, 0.0f,	 // ArcBall's RT in ArcBall space (Z in, Y down, X right) in ArcBall format
0.0f, 1.0f, 0.0f, 0.0f,
0.0f, 0.0f, 1.0f, 0.0f,
0.0f, 0.0f, 0.0f, 1.0f };

Matrix3fT   LastRot = { 1.0f, 0.0f, 0.0f,					// NEW: Last Rotation
0.0f, 1.0f, 0.0f,
0.0f, 0.0f, 1.0f };

Matrix3fT   ThisRot = { 1.0f, 0.0f, 0.0f,					// NEW: This Rotation
0.0f, 1.0f, 0.0f,
0.0f, 0.0f, 1.0f };

ArcBallT    ArcBall(640.0f, 480.0f);				                // NEW: ArcBall Instance
Point2fT    MousePt;												// NEW: Current Mouse Point
bool isDragging = false;

// ArcBall space: X right, Y down, Z out
// Camera space: X right, Y down, Z in

Mat WScamaligned_from_AB = cv::Mat::eye(4, 4, CV_64F); // transform from ArcBall space to WS origin location with axes aligned with camera space
Mat AB_from_WScamaligned = cv::Mat::eye(4, 4, CV_64F); // transform from WS origin location with axes aligned with camera space to ArcBall space
Mat WScamaligned_from_WS = cv::Mat::eye(4, 4, CV_64F); // transform from WS to WS origin location with axes aligned with camera space
Mat WS_from_WScamaligned = cv::Mat::eye(4, 4, CV_64F); // transform from WS origin location with axes aligned with camera space to WS
Mat RTab; // ArcBall's RT in ArcBall space (Z in, Y down, X right) in OpenCV format
Point3d bv_min, bv_max; // bounding volume min and max locations in world space for our point cloud across all cameras
Mat WSoffset_from_WS = cv::Mat::eye(4, 4, CV_64F); // transform from a world space with the axis aligned to camera space rotation to the same but translated so that the origin is at half the z of the bounding volume of the point cloud in world space so we can rotate around the center of the product instead of its base (the product is already centered about x and y and positioned so that its bottom is at z = 0)
Mat WS_from_WSoffset = cv::Mat::eye(4, 4, CV_64F); // inverse of WSoffset_from_WS

double ws_z_offset; // world space Z offset, set to half of scene height, used to set rotation around center of scene rather than origin (scene is already centered around X and Y axes)

void onMouse(int event, int x, int y, int flags, void*)
{
	MousePt.s.X = (float)x;
	MousePt.s.Y = (float)y;

	if (event == CV_EVENT_LBUTTONUP || event == CV_EVENT_RBUTTONUP)
	{
		cout << "Mouse up at location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;
		isDragging = false;
	}
	else if (event == CV_EVENT_LBUTTONDOWN || event == CV_EVENT_RBUTTONDOWN)
	{
		cout << "Mouse down at location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;
		isDragging = true;										// Prepare For Dragging
		LastRot = ThisRot;										// Set Last Static Rotation To Last Dynamic One
		ArcBall.click(&MousePt);								// Update Start Vector And Prepare For Dragging
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		cout << "Dragging mouse to location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;
		
		Quat4fT     ThisQuat;

		ArcBall.drag(&MousePt, &ThisQuat);						// Update End Vector And Get Rotation As Quaternion
		Matrix3fSetRotationFromQuat4f(&ThisRot, &ThisQuat);		// Convert Quaternion Into Matrix3fT
		Matrix4fSetRotationFromMatrix3f(&TransformAB, &ThisRot);	// Set Our Final Transform's Rotation From This One

		RTab = Arc4fTToCV(&TransformAB); // converts ArcBall RT to OpenCV format, but still in ArcBall space

		// WS_from_WScamaligned transforms from WS origin location with axes aligned with camera space to WS.  WScamaligned_from_WS does the inverse.  Our camera space coordinate system is z in, x right, y up.
		RTinv.copyTo(WS_from_WScamaligned);
		WS_from_WScamaligned.at<double>(0, 3) = 0.;
		WS_from_WScamaligned.at<double>(1, 3) = 0.;
		WS_from_WScamaligned.at<double>(2, 3) = 0.;
		WScamaligned_from_WS = WS_from_WScamaligned.inv();
		
		/*
		// enforce minimum height
		// if resulting camera position in offset world space is below (GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset) in world space, rotate it around the axis perpendicular both to camera Z and world space Z (use cross product) that puts its world space Z at the minimum (GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset)
		Mat RT_offset = WS_from_WSoffset * WS_from_WScamaligned * WScamaligned_from_AB * RTab * AB_from_WScamaligned * WScamaligned_from_WS * WSoffset_from_WS;
		Mat RTinv_offset = RT_offset.inv();

		if (RTinv_offset.at<double>(2, 3) < GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset) {
			Mat Z_cs = cv::Mat::eye(4, 4, CV_64F);
			Mat Z_ws = 
		}
		*/

		RT = RT * WS_from_WSoffset * WS_from_WScamaligned * WScamaligned_from_AB * RTab * AB_from_WScamaligned * WScamaligned_from_WS * WSoffset_from_WS;
		
		// else if zooming: RT = RT * RTinv* RTzoom_incameraspace * RT; // RTzoom_incameraspace is an eye matrix with a certain value in (2,3) for translation in/out

		RTinv = RT.inv();

		cout << endl << "RT" << endl << RT << endl << endl;

		imgT = nvs->SynthesizeView(&calib, &RT, sz, exclude_cam_ids, 4); // synthesize a new view
		cv::imshow("image", imgT);
	}
}

int main (int argc, char *argv[]) {
	
	srand(static_cast <unsigned> (time(0))); // initialize random seed - should be only called once per run, at its start

	nvs = new NVS("doc.xml");
	nvs->scene_->SaveCamerasWSPointProjections();

	// Stereo reconstruction
	double max_depth = 0.;
	double min_nonzero_depth = 0.;
	std::map<int, Mat*> imgsT, imgsD;
	std::map<int, Matrix<double, 3, 4>> RTs;
	for (std::map<int, Camera*>::iterator it_in = nvs->scene_->cameras_.begin(); it_in != nvs->scene_->cameras_.end(); ++it_in) {
		int cid = (*it_in).first;

		// set up images
		imgsT[cid] = &(*it_in).second->imgT_;
		imgsD[cid] = &(*it_in).second->imgD_;

		// set up  RTins
		Matrix<double, 3, 4> RT = Convert4x4OpenCVExtrinsicsMatTo3x4EigenExtrinsicsMatrixd(&(*it_in).second->RT_);
		RTs[cid] = RT;

		if ((*it_in).second->dm_->max_depth_ws_ > max_depth) max_depth = (*it_in).second->dm_->max_depth_ws_;
		if (((*it_in).second->dm_->min_nonzero_depth_ws_ < min_nonzero_depth) ||
			(min_nonzero_depth == 0.))
			min_nonzero_depth = (*it_in).second->dm_->min_nonzero_depth_ws_;
	}
	int cid_out;
	StereoReconstruction *sr = new StereoReconstruction();	
	sr->Init(imgsT, imgsD, RTs, min_nonzero_depth, max_depth);
	for (std::map<int, Camera*>::iterator it_out = nvs->scene_->cameras_.begin(); it_out != nvs->scene_->cameras_.end(); ++it_out) { // reconstruct each camera as output depth map in turn
		cid_out = (*it_out).first;
		
		// set up K
		Matrix3d K = ConvertOpenCVMatToEigenMatrix3d(&(*it_out).second->calib_.K_);
		
		Matrix<float, Dynamic, Dynamic> matD_out = sr->Stereo(cid_out, K);
	}
	
	// NVS
	int cid = nvs->scene_->cameras_.begin()->first; // select a camera view to recreate
	calib = nvs->scene_->cameras_[cid]->calib_.Copy();
	sz = Size(nvs->scene_->cameras_[cid]->width_ * 0.5, nvs->scene_->cameras_[cid]->height_ * 0.5);
	nvs->scene_->DeterminePointCloudBoundingVolume(bv_min, bv_max);
	ws_z_offset = bv_max.z / 2.;
	WSoffset_from_WS.at<double>(2, 3) = -1. * ws_z_offset;
	WS_from_WSoffset.at<double>(2, 3) = ws_z_offset;
	//exclude_cam_ids.push_back(cid);
	nvs->scene_->cameras_[cid]->RT_.copyTo(RT); // starting position of camera
	RTinv = RT.inv();

	Mat imgT = nvs->SynthesizeView(&calib, &RT, sz, exclude_cam_ids, 4); // synthesize a new view
	
	// Set up ArcBall
	ArcBall.setBounds(sz.width, sz.height);
	AB_from_WScamaligned.at<double>(2, 2) = -1.; // ArcBall coordinate system is rotationally aligned with our camera space coordinate system in x and y (x right, y up) but ArcBall is z out while camera space is z in
	WScamaligned_from_AB = AB_from_WScamaligned.inv();

	// Interactivity with user
	cv::namedWindow("image", 1);
	cv::imshow("image", imgT);
	cv::setMouseCallback("image", onMouse, 0);
	for (;;)
	{
		int c = waitKey(0);

		if ((char)c == 27) // exit;
			break;
		else if ((char)c == 'r') { // reset view
			nvs->scene_->cameras_[cid]->RT_.copyTo(RT); // starting position of camera
			RTinv = RT.inv();
			imgT = nvs->SynthesizeView(&calib, &RT, sz, exclude_cam_ids, 4); // synthesize a new view
			cv::imshow("image", imgT);
		}
	}


	delete nvs;
	delete sr;

	return 0;
}