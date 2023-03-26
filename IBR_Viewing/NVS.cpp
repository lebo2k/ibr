#include "NVS.h"

// agisoft_filename holds the filename for the Agisoft scene data
NVS::NVS(std::string agisoft_filename) {
	scene_ = new Scene();
	scene_->Init("doc.xml");
}

NVS::~NVS() {
	delete scene_;
}

// performs new view synthesis to create an image from a camera with pose given by camera extrinsics matrix RT, with intrinsics given by the calibration calib, and potentially adjusted screen space pixel size view_size
// requires that RT is a 4x4 camera extrinsics matrix of type CV_64F
// num_cams gives the number of input cameras used to compute the new view; given the number, it will always attempt to use the cameras closest to view_pos for reconstruction but will take into account trying to find cameras from more unique positions, as per Scene::OrderCamsByDistance()
// exclude_cam_ids holds a list of camera IDs that should be excluded from the list used to sythesize the new view
Mat NVS::SynthesizeView(Calibration *calib, Mat *RT, cv::Size view_size, std::vector<int> exclude_cam_ids, int num_cams) {
	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	assert(RT->rows == 4 && RT->cols == 4 && RT->type() == CV_64F, "NVS::NVS() RT must be a 4x4 camera extrinsics matrix of type CV_64F");

	// recalibrate given view_size
	calib->RecalibrateNewSS(view_size);

	// compute P and Pinv
	Mat RTinv = RT->inv();
	Mat P = cv::Mat::zeros(3, 4, CV_64F);
	Mat Pinv = cv::Mat::zeros(4, 3, CV_64F);
	Camera::ComputeProjectionMatrices(&calib->K_, &calib->Kinv_, RT, &RTinv, P, Pinv);

	// get ordered list of closest cameras
	Point3d view_pos = Camera::GetCameraPositionWS(&RTinv);
	Point3d view_dir = Camera::GetCameraViewDirectionWS(&RTinv);
	std::vector<int> cam_ids = scene_->GetClosestCams(view_pos, view_dir, exclude_cam_ids, num_cams);

	std::map<int, Mat> imgTs; // texture images
	std::map<int, Mat> imgDs; // depth images
	std::map<int, Mat> imgMs; // masks

	for (int cam_num = 0; cam_num < cam_ids.size(); cam_num++) {
		int cid = cam_ids[cam_num];
		Mat imgT = cv::Mat::zeros(view_size, CV_8UC3);
		Mat imgD = cv::Mat::zeros(view_size, CV_32F);
		Mat imgMask = cv::Mat::zeros(view_size, CV_8UC1);

		scene_->cameras_[cid]->Reproject(&P, &imgT, &imgD, &imgMask);

		if (debug) {
			cout << "Displaying images for camera with ID " << cid << endl;
			display_mat(&imgT, "Texture image");
			DepthMap::DisplayDepthImage(&imgD);
			display_mat(&imgMask, "imgMask");
		}

		imgTs[cid] = imgT;
		imgDs[cid] = imgD;
		imgMs[cid] = imgMask;
	}

	Mat imgT = cv::Mat::zeros(view_size, CV_8UC3);
	Mat imgD = cv::Mat::zeros(view_size, CV_32F);
	Mat imgM = cv::Mat::zeros(view_size, CV_8UC1);
	Blend(imgTs, imgDs, imgMs, cam_ids, &imgT, &imgD, &imgM);
	Mat imgD_missing = cv::Mat(imgT.size(), CV_8UC1, 1); // non-zero pixels indicate the area missing depth values; initialize values to 1 so zeros here don't match zeros in imgM during compare operation below
	Mat imgT_inpaint_mask = cv::Mat::zeros(imgT.size(), CV_8UC1); // non-zero pixels indicate the area that needs to be inpainted
	compare(imgD, 0., imgD_missing, CMP_EQ);
	compare(imgM, imgD_missing, imgT_inpaint_mask, CMP_EQ);
	cv::inpaint(imgT, imgT_inpaint_mask, imgT, 3, CV_INPAINT_NS); // use radius ~= 3; use flag either CV_INPAINT_NS or CV_INPAINT_TELEA for different algorithms

	if (debug) {
		display_mat(&imgT, "Texture image result");
		display_mat(&imgD, "Depth image result");
		display_mat(&imgM, "Mask image result");
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "NVS::SynthesizeView() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}

	return imgT;
};

// blends input new views to a single consolidated new view, putting the resulting texture image in imgT, the associated resulting depth image in imgD, and the associated mask imgM for opaque versus transparent values
// imgTs is a map of reference camera ID => texture image new view
// imgDs is a map of reference camera ID => depth image new view
// imgDinterpMasks is a map of reference camera ID => depth interpolation binary mask for the new view where 255 means the depth value was interpolated (and so should be given lower priority during blending) and 0 means it was an original depth value
// imgMs is a map of reference camera ID => binary mask image for the the new view
// ordered_cam_ids is a vector of reference camera IDs ordered with the highest priority first
// reference cameras listed in imgTs, imgDs, and ordered_cam_ids must all match, though are not expected to be in the same order
// blending works as follows: first pixel color priority goes to the reference camera with the smallest depth; if multiple have the same smallest depth, priority goes to the earliest reference camera in the list among those with the smallest depth; if none have a depth value for the pixel, priority goes to the earliest reference camera in the list. Also, non-interpolated depths give camera higher priority than interpolated depths.  Also, consensus among cameras determines whether a given pixel is visible at all according to visibility masks imgMs.
void NVS::Blend(std::map<int, Mat> imgTs, std::map<int, Mat> imgDs, std::map<int, Mat> imgMs, std::vector<int> ordered_cam_ids, Mat *imgTresult, Mat *imgDresult, Mat *imgMresult) {
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	// validate arguments and set width and height values (necessary for both validation and function operations below)
	assert(imgTs.size() > 0, "NVS::Blend() imgTs must not be an empty data structure");
	assert(imgDs.size() > 0, "NVS::Blend() imgDs must not be an empty data structure");
	assert(imgMs.size() > 0, "NVS::Blend() imgMs must not be an empty data structure");
	int cid_first = imgTs.begin()->first;
	int width = imgTs[cid_first].cols;
	int height = imgTs[cid_first].rows;
	bool lists_match = true;
	for (std::map<int, Mat>::iterator it = imgTs.begin(); it != imgTs.end(); ++it) {
		assert((*it).second.type() == CV_8UC3, "NVS::Blend() imgTs must contain texture images of type CV_8UC3");
		assert((*it).second.cols == width && (*it).second.rows == height, "NVS::Blend() imgTs, imgDs, imgTresult, imgDresult, and imgMresult must all be of the same size");
		if (imgDs.find((*it).first) == imgDs.end()) lists_match = false; // reference cam ID must also exist in imgDs list
		if (imgMs.find((*it).first) == imgMs.end()) lists_match = false; // reference cam ID must also exist in imgMs list
		if (std::find(ordered_cam_ids.begin(), ordered_cam_ids.end(), (*it).first) == ordered_cam_ids.end()) lists_match = false; // reference cam ID must also exist in ordered_cam_ids list
	}
	for (std::map<int, Mat>::iterator it = imgDs.begin(); it != imgDs.end(); ++it) {
		assert((*it).second.type() == CV_32F, "NVS::Blend() imgDs must contain depth images of type CV_32F");
		assert((*it).second.cols == width && (*it).second.rows == height, "NVS::Blend() imgTs, imgDs, imgTresult, imgDresult, and imgMresult must all be of the same size");
		if (imgTs.find((*it).first) == imgTs.end()) lists_match = false; // reference cam ID must also exist in imgTs list
		if (imgMs.find((*it).first) == imgMs.end()) lists_match = false; // reference cam ID must also exist in imgMs list
		if (std::find(ordered_cam_ids.begin(), ordered_cam_ids.end(), (*it).first) == ordered_cam_ids.end()) lists_match = false; // reference cam ID must also exist in ordered_cam_ids list
	}
	for (std::map<int, Mat>::iterator it = imgMs.begin(); it != imgMs.end(); ++it) {
		assert((*it).second.type() == CV_8UC1, "NVS::Blend() imgMs must contain binary mask of type CV_8UC1");
		assert((*it).second.cols == width && (*it).second.rows == height, "NVS::Blend() imgTs, imgDs, imgTresult, imgDresult, and imgMresult must all be of the same size");
		if (imgTs.find((*it).first) == imgTs.end()) lists_match = false; // reference cam ID must also exist in imgTs list
		if (imgDs.find((*it).first) == imgDs.end()) lists_match = false; // reference cam ID must also exist in imgDs list
		if (std::find(ordered_cam_ids.begin(), ordered_cam_ids.end(), (*it).first) == ordered_cam_ids.end()) lists_match = false; // reference cam ID must also exist in ordered_cam_ids list
	}
	for (std::vector<int>::iterator it = ordered_cam_ids.begin(); it != ordered_cam_ids.end(); ++it) {
		if (imgTs.find(*it) == imgTs.end()) lists_match = false; // reference cam ID must also exist in imgTs list
		if (imgDs.find(*it) == imgDs.end()) lists_match = false; // reference cam ID must also exist in imgDs list
		if (imgMs.find(*it) == imgMs.end()) lists_match = false; // reference cam ID must also exist in imgMs list
	}
	assert(lists_match, "NVS::Blend() reference cameras listed in imgTs, imgDs, imgDinterpMasks, and ordered_cam_ids must all match, though are not expected to be in the same order");
	assert(imgTresult->cols == width && imgTresult->rows == height, "NVS::Blend() imgTs, imgDs, imgDinterpMasks, imgTresult, imgDresult, and imgMresult must all be of the same size");
	assert(imgDresult->cols == width && imgDresult->rows == height, "NVS::Blend() imgTs, imgDs, imgDinterpMasks, imgTresult, imgDresult, and imgMresult must all be of the same size");
	assert(imgMresult->cols == width && imgMresult->rows == height, "NVS::Blend() imgTs, imgDs, imgDinterpMasks, imgTresult, imgDresult, and imgMresult must all be of the same size");
	
	float min_depth, curr_depth;
	float visibility;
	vector<int> min_depth_cam_ids;
	int priority_cam_id;
	std::map<int, Vec3b*> pTs; // map of reference camera ID => pointer into associated imgTs data structure
	std::map<int, float*> pDs; // map of reference camera ID => pointer into associated imgDs data structure
	std::map<int, uchar*> pMs; // map of reference camera ID => pointer into associated imgMs data structure
	Vec3b* pTresult;
	float* pDresult;
	uchar* pMresult;
	for (int r = 0; r < height; r++) {
		// initialize data pointers for this row of the images
		pTresult = imgTresult->ptr<Vec3b>(r);
		pDresult = imgDresult->ptr<float>(r);
		pMresult = imgMresult->ptr<uchar>(r);
		for (std::map<int, Mat>::iterator it = imgTs.begin(); it != imgTs.end(); ++it) {
			pTs[(*it).first] = (*it).second.ptr<Vec3b>(r);
		}
		for (std::map<int, Mat>::iterator it = imgDs.begin(); it != imgDs.end(); ++it) {
			pDs[(*it).first] = (*it).second.ptr<float>(r);
		}
		for (std::map<int, Mat>::iterator it = imgMs.begin(); it != imgMs.end(); ++it) {
			pMs[(*it).first] = (*it).second.ptr<uchar>(r);
		}
		
		// for this pixel, determine relative depths of cameras
		for (int c = 0; c < width; c++) {
			// for this pixel, determine consensus among cameras on visibility - if not visible, skip the pixel
			visibility = 0;
			for (std::map<int, Mat>::iterator it = imgMs.begin(); it != imgMs.end(); ++it) {
				visibility += pMs[(*it).first][c];
			}
			visibility /= imgMs.size();
			if (visibility < 127.) {
				pMresult[c] = 0;
				continue; // if consensus is not visible, skip the pixel
			}
			else pMresult[c] = 255;

			// for this pixel, determine relative depths of cameras
			min_depth = 0.;
			for (std::map<int, Mat>::iterator it = imgDs.begin(); it != imgDs.end(); ++it) {
				curr_depth = pDs[(*it).first][c];
				if (curr_depth == 0) continue;
				if ((curr_depth <= min_depth) ||
					(min_depth == 0.)) {
					min_depth = curr_depth;
					min_depth_cam_ids.push_back((*it).first);
				}
			}

			priority_cam_id = -1;
			if (min_depth == 0.) priority_cam_id = ordered_cam_ids.front(); // choose first camera from ordered list
			else if (min_depth_cam_ids.size() == 1) priority_cam_id = min_depth_cam_ids.front(); // there is a clear winner for priority camera
			else { // min_depth_cam_ids.size() > 1; there is a tie for priority camera among a set of cameras
				for (std::vector<int>::iterator it = ordered_cam_ids.begin(); it != ordered_cam_ids.end(); ++it) {
					if (std::find(min_depth_cam_ids.begin(), min_depth_cam_ids.end(), (*it)) != min_depth_cam_ids.end()) {
						priority_cam_id = (*it);
						break;
					}
				}
				assert(priority_cam_id != -1, "NVS::Blend() priority_cam_id should have been found but was not");
			}

			// copy image data from priority camera
			pTresult[c] = pTs[priority_cam_id][c];
			pDresult[c] = pDs[priority_cam_id][c];

			// clear data structures for next pass
			min_depth_cam_ids.erase(min_depth_cam_ids.begin(), min_depth_cam_ids.end());
		}
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "NVS::Blend() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}
