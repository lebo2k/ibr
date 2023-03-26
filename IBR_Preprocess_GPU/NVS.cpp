#include "NVS.h"

// agisoft_filename holds the filename for the Agisoft scene data
NVS::NVS() {
}

NVS::~NVS() {
}

void NVS::Init(Scene *scene, StereoData *sd, Calibration *calib, cv::Size view_size) {
	bool debug = true;

	scene_ = scene;
	sd_ = sd;
	calib_ = calib->Copy();
	view_size_.height = view_size.height;
	view_size_.width = view_size.width;

	scene_->UpdatePointCloudBoundingVolume();

	// recalibrate given view_size
	calib_.RecalibrateNewSS(view_size_);

	for (std::map<int, Mat>::iterator it = sd_->imgsT_.begin(); it != sd_->imgsT_.end(); ++it) {
		int cid = (*it).first;

		// update image
		Mat imgTnew = cv::Mat::zeros(view_size_, CV_8UC3);
		Mat imgMnew = cv::Mat::zeros(view_size_, CV_8UC1);
		int interpolation;
		if (sd_->heights_[cid] > view_size_.height)
			interpolation = CV_INTER_AREA;
		else
			interpolation = CV_INTER_CUBIC;

		resize((*it).second, imgTnew, view_size_, 0.0, 0.0, interpolation); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
		resize(scene_->cameras_[cid]->imgMask_, imgMnew, view_size_, 0.0, 0.0, interpolation); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
		imgsT_[cid] = imgTnew;
		heights_[cid] = view_size.height;
		widths_[cid] = view_size.width;

		// convert it to eigen format
		Eigen::Matrix<float, Dynamic, 3> A;
		EigenOpenCV::cv2eigenImage<float>(&imgTnew, &A);
		Eigen::Matrix<bool, Dynamic, Dynamic> maskX;
		EigenOpenCV::cv2eigen(imgMnew, maskX);

		// resize mask
		maskX.resize(view_size.height*view_size.width, 1);
		Matrix<bool, Dynamic, 1> mask = maskX;
		masks_[cid] = mask;
		
		// split it by channel
		Matrix<int, Dynamic, Dynamic> blue = A.col(0).cast<int>();
		Matrix<int, Dynamic, Dynamic> green = A.col(1).cast<int>();
		Matrix<int, Dynamic, Dynamic> red = A.col(2).cast<int>();
		blue.resize(heights_[cid], widths_[cid]);
		green.resize(heights_[cid], widths_[cid]);
		red.resize(heights_[cid], widths_[cid]);
		As_blue_[cid] = blue;
		As_green_[cid] = green;
		As_red_[cid] = red;

		/*
		display_mat(&imgTnew, "imgTnew", scene->cameras_[cid]->orientation_);
		DisplayImages::DisplayBGRImage(&A, heights_[cid], widths_[cid]);
		DebugPrintMatrix(&blue, "blue");

		DisplayImages::DisplayGrayscaleImage(&blue, heights_[cid], widths_[cid]);
		DisplayImages::DisplayGrayscaleImage(&green, heights_[cid], widths_[cid]);
		DisplayImages::DisplayGrayscaleImage(&red, heights_[cid], widths_[cid]);


		Mat b2 = Mat::zeros(heights_[cid], widths_[cid], CV_8UC1);
		Matrix<unsigned char, Dynamic, Dynamic> bluetest = blue.cast<unsigned char>();
		EigenOpenCV::eigen2cv(bluetest, b2);
		display_mat(&b2, "b2", scene->cameras_[cid]->orientation_);

		Mat g2 = Mat::zeros(heights_[cid], widths_[cid], CV_8UC1);
		Matrix<unsigned char, Dynamic, Dynamic> greentest = green.cast<unsigned char>();
		EigenOpenCV::eigen2cv(greentest, g2);
		display_mat(&g2, "g2", scene->cameras_[cid]->orientation_);

		Mat r2 = Mat::zeros(heights_[cid], widths_[cid], CV_8UC1);
		Matrix<unsigned char, Dynamic, Dynamic> redtest = red.cast<unsigned char>();
		EigenOpenCV::eigen2cv(redtest, r2);
		display_mat(&r2, "r2", scene->cameras_[cid]->orientation_);
		*/

		// update projection matrices for the camera
		Matrix<double, 3, 4> P;
		Matrix<double, 4, 3> Pinv;
		Camera::ComputeProjectionMatrices(&calib_.K_, &calib_.Kinv_, &scene->cameras_[cid]->RT_, &scene->cameras_[cid]->RTinv_, &P, &Pinv);
		Ps_[cid] = P;
		Pinvs_[cid] = Pinv;
	}
}

// performs new view synthesis to create an image from a camera with pose given by camera extrinsics matrix RT, with intrinsics given by the calibration calib, and potentially adjusted screen space pixel size view_size
// requires that RT is a 4x4 camera extrinsics matrix of type CV_64F
// num_cams gives the number of input cameras used to compute the new view; given the number, it will always attempt to use the cameras closest to view_pos for reconstruction but will take into account trying to find cameras from more unique positions, as per Scene::OrderCamsByDistance()
// exclude_cam_ids holds a list of camera IDs that should be excluded from the list used to sythesize the new view
Mat NVS::SynthesizeView(Mat *RTmat, std::vector<int> exclude_cam_ids, int num_cams) {
	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	assert(RTmat->rows == 4 && RTmat->cols == 4 && RTmat->type() == CV_64F);

	// compute P and Pinv
	Matrix4d RT;
	EigenOpenCV::cv2eigen((*RTmat), RT);
	Matrix4d RTinv = RT.inverse();
	Matrix<double, 3, 4> P;
	Matrix<double, 4, 3> Pinv;
	Camera::ComputeProjectionMatrices(&calib_.K_, &calib_.Kinv_, &RT, &RTinv, &P, &Pinv);

	// get ordered list of closest cameras
	Point3d view_pos = Camera::GetCameraPositionWS(&RTinv);
	Point3d view_dir = Camera::GetCameraViewDirectionWS(&RTinv);
	std::vector<int> cam_ids = scene_->GetClosestCams(view_pos, view_dir, exclude_cam_ids, num_cams); // returns ordered list of cameras with closest first

	std::map<int, Mat> imgTs; // texture images
	std::map<int, MatrixXf> imgDs; // depth images
	std::map<int, Mat> imgMs; // masks

	for (int cam_num = 0; cam_num < cam_ids.size(); cam_num++) {
		int cid = cam_ids[cam_num];
		cout << "cid " << cid << endl;
		Mat imgT = cv::Mat::zeros(view_size_, CV_8UC3);
		MatrixXf imgD(view_size_.height, view_size_.width);
		Mat imgMask = cv::Mat::zeros(view_size_, CV_8UC1);

		//scene_->cameras_[cid]->Reproject(&P, &RT, &imgT, &imgD, &imgMask);
		scene_->cameras_[cid]->ReprojectMesh(view_dir, &P, &RT, &imgT, &imgD, &imgMask);

		if (debug) {
			//cout << "Displaying images for camera with ID " << cid << endl;
			//display_mat(&scene_->cameras_[cid]->imgT_, "Texture image", scene_->cameras_[cid]->orientation_);
			//DepthMap::DisplayDepthImage(&scene_->cameras_[cid]->dm_->depth_map_, scene_->cameras_[cid]->orientation_);
			//display_mat(&scene_->cameras_[cid]->imgMask_, "imgMask", scene_->cameras_[cid]->orientation_);
			cout << "Displaying reprojected images for camera with ID " << cid << endl;
			display_mat(&imgT, "Texture image", scene_->cameras_[cid]->orientation_);
			DepthMap::DisplayDepthImage(&imgD, scene_->cameras_[cid]->orientation_);
			display_mat(&imgMask, "imgMask", scene_->cameras_[cid]->orientation_);
		}

		imgTs[cid] = imgT;
		imgDs[cid] = imgD;
		imgMs[cid] = imgMask;
	}

	if (debug) cout << "completed cam_ids" << endl;

	Mat imgT = cv::Mat::zeros(view_size_, CV_8UC3);
	Mat imgD = cv::Mat::zeros(view_size_, CV_32F);
	Mat imgM = cv::Mat::zeros(view_size_, CV_8UC1);
	Mat imgMprimary = cv::Mat::zeros(view_size_, CV_8UC1);
	Composite(imgTs, imgDs, imgMs, cam_ids, &imgT, &imgD, &imgM, &imgMprimary);
	if (GLOBAL_INPAINT_VIEWS) {
		Mat imgD_missing = cv::Mat(imgT.size(), CV_8UC1, 1); // non-zero pixels indicate the area missing depth values; initialize values to 1 so zeros here don't match zeros in imgM during compare operation below
		Mat imgT_inpaint_mask = cv::Mat::zeros(imgT.size(), CV_8UC1); // non-zero pixels indicate the area that needs to be inpainted
		compare(imgD, 0., imgD_missing, CMP_EQ);
		compare(imgM, imgD_missing, imgT_inpaint_mask, CMP_EQ);
		cv::inpaint(imgT, imgT_inpaint_mask, imgT, 3, CV_INPAINT_NS); // use radius ~= 3; use flag either CV_INPAINT_NS or CV_INPAINT_TELEA for different algorithms
	}
	if (GLOBAL_SHARPEN) {
		if (debug) display_mat(&imgT, "Texture image result before sharpening");
		Sharpen(&imgT, &imgD, &imgM, cam_ids, P.cast<float>(), view_dir);
		if (debug) display_mat(&imgT, "Texture image after sharpening an iteration");
	}

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

// composites input new views to a single consolidated new view, putting the resulting texture image in imgT, the associated resulting depth image in imgD, and the associated mask imgM for opaque versus transparent values
// imgTs is a map of reference camera ID => texture image new view
// imgDs is a map of reference camera ID => depth image new view
// imgDinterpMasks is a map of reference camera ID => depth interpolation binary mask for the new view where 255 means the depth value was interpolated (and so should be given lower priority during blending) and 0 means it was an original depth value
// imgMs is a map of reference camera ID => binary mask image for the the new view
// ordered_cam_ids is a vector of reference camera IDs ordered by camera with closest view direction to virtual view direction first and farthest from virtual view direction last
// reference cameras listed in imgTs, imgDs, and ordered_cam_ids must all match, though are not expected to be in the same order
// compositing works as follows: first pixel color priority goes to the reference camera with the closest view direction to virtual camera (by dot product), and so on down the line by camera with pixels masked-out if no color is available from any camera.  Caveat: if a later camera generates a depth significantly closer to the virtual view than an earlier camera, the later camera's pixel color takes precedence
// imgMprimary is updated to be a mask of pixels that have colors from the primary camera (the one closest to the current virtual view) since these are the most trusted
void NVS::Composite(std::map<int, Mat> imgTs, std::map<int, MatrixXf> imgDs, std::map<int, Mat> imgMs, std::vector<int> ordered_cam_ids, Mat *imgTresult, Mat *imgDresult, Mat *imgMresult, Mat *imgMprimary) {
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	bool debug = true;

	// validate arguments and set width and height values (necessary for both validation and function operations below)
	assert(imgTs.size() > 0);
	assert(imgDs.size() > 0);
	assert(imgMs.size() > 0);
	int cid_first = imgTs.begin()->first;
	int width = imgTs[cid_first].cols;
	int height = imgTs[cid_first].rows;
	bool lists_match = true;
	for (std::map<int, Mat>::iterator it = imgTs.begin(); it != imgTs.end(); ++it) {
		assert((*it).second.type() == CV_8UC3);
		assert((*it).second.cols == width && (*it).second.rows == height);
		if (imgDs.find((*it).first) == imgDs.end()) lists_match = false; // reference cam ID must also exist in imgDs list
		if (imgMs.find((*it).first) == imgMs.end()) lists_match = false; // reference cam ID must also exist in imgMs list
		if (std::find(ordered_cam_ids.begin(), ordered_cam_ids.end(), (*it).first) == ordered_cam_ids.end()) lists_match = false; // reference cam ID must also exist in ordered_cam_ids list
	}
	for (std::map<int, MatrixXf>::iterator it = imgDs.begin(); it != imgDs.end(); ++it) {
		assert((*it).second.cols() == width && (*it).second.rows() == height);
		if (imgTs.find((*it).first) == imgTs.end()) lists_match = false; // reference cam ID must also exist in imgTs list
		if (imgMs.find((*it).first) == imgMs.end()) lists_match = false; // reference cam ID must also exist in imgMs list
		if (std::find(ordered_cam_ids.begin(), ordered_cam_ids.end(), (*it).first) == ordered_cam_ids.end()) lists_match = false; // reference cam ID must also exist in ordered_cam_ids list
	}
	for (std::map<int, Mat>::iterator it = imgMs.begin(); it != imgMs.end(); ++it) {
		assert((*it).second.type() == CV_8UC1);
		assert((*it).second.cols == width && (*it).second.rows == height);
		if (imgTs.find((*it).first) == imgTs.end()) lists_match = false; // reference cam ID must also exist in imgTs list
		if (imgDs.find((*it).first) == imgDs.end()) lists_match = false; // reference cam ID must also exist in imgDs list
		if (std::find(ordered_cam_ids.begin(), ordered_cam_ids.end(), (*it).first) == ordered_cam_ids.end()) lists_match = false; // reference cam ID must also exist in ordered_cam_ids list
	}
	for (std::vector<int>::iterator it = ordered_cam_ids.begin(); it != ordered_cam_ids.end(); ++it) {
		if (imgTs.find(*it) == imgTs.end()) lists_match = false; // reference cam ID must also exist in imgTs list
		if (imgDs.find(*it) == imgDs.end()) lists_match = false; // reference cam ID must also exist in imgDs list
		if (imgMs.find(*it) == imgMs.end()) lists_match = false; // reference cam ID must also exist in imgMs list
	}
	assert(lists_match);
	assert(imgTresult->cols == width && imgTresult->rows == height);
	assert(imgDresult->cols == width && imgDresult->rows == height);
	assert(imgMresult->cols == width && imgMresult->rows == height);
	assert(imgMprimary->cols == width && imgMprimary->rows == height);

	if (debug) cout << "NVS::Composite() passed assertions" << endl;

	int primary_cid = imgTs.begin()->first;
	imgMprimary->setTo(0);

	vector<int> min_depth_cam_ids;
	std::map<int, Vec3b*> pTs; // map of reference camera ID => pointer into associated imgTs data structure
	std::map<int, uchar*> pMs; // map of reference camera ID => pointer into associated imgMs data structure
	Vec3b* pTresult;
	float* pDresult;
	uchar* pMresult;
	uchar* pMprimary;
	double sig_depth_diff = 5*GLOBAL_MESH_EDGE_DISTANCE_MAX / scene_->agisoft_to_world_scale_;
	for (int r = 0; r < height; r++) {
		// initialize data pointers for this row of the images
		pTresult = imgTresult->ptr<Vec3b>(r);
		pMresult = imgMresult->ptr<uchar>(r);
		pDresult = imgDresult->ptr<float>(r);
		pMprimary = imgMprimary->ptr<uchar>(r);
		for (std::map<int, Mat>::iterator it = imgTs.begin(); it != imgTs.end(); ++it) {
			pTs[(*it).first] = (*it).second.ptr<Vec3b>(r);
		}
		for (std::map<int, Mat>::iterator it = imgMs.begin(); it != imgMs.end(); ++it) {
			pMs[(*it).first] = (*it).second.ptr<uchar>(r);
		}
		
		// for this pixel, determine relative depths of cameras
		for (int c = 0; c < width; c++) {
			/*
			// actually, can't do this because the masks can't and don't distinguish between masked-out pixels that were reprojected there and empty pixels revealed due to shifted occlusions
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
			*/
			pMresult[c] = 255;
			
			// for this pixel, go through each camera in order and attempt to assign a color if possible, stopping when a color has been assigned
			bool assigned = false;
			for (std::map<int, Mat>::iterator it = imgTs.begin(); it != imgTs.end(); ++it) {
				bool valid = true;
				int cid = (*it).first;
				if (!pMs[cid][c]) valid = false; // must not be masked out for this image
				if (pTs[cid][c] == Vec3b(0, 0, 0)) valid = false; // must have color
				float depth = imgDs[cid](r, c);
				if (depth <= 0.) valid = false; // must have positive virtual view CS depth
				if (valid) {
					// copy image data from priority camera, unless already assigned with a depth at or close to this reprojection's depth...if earlier's reprojected image's depth significantly larger, use current color and depth
					if (assigned) {
						if ((pDresult[c] - depth) > sig_depth_diff) {
							pTresult[c] = pTs[cid][c];
							pDresult[c] = depth;
							if (cid == primary_cid) pMprimary[c] = 255;
						}
					}
					else {
						pTresult[c] = pTs[cid][c];
						pDresult[c] = depth;
						if (cid == primary_cid) pMprimary[c] = 255;
						assigned = true;
					}
				}
			}
			
			// if no color could be assigned, changed the mask value to masked-out
			if (!assigned)
				pMresult[c] = 0;
			
			// clear data structures for next pass
			min_depth_cam_ids.erase(min_depth_cam_ids.begin(), min_depth_cam_ids.end());
		}
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "NVS::Composite() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

float NVS::Ephoto(Vec3b color, map<int, Vec3b>  sampled_colors) {
	Vec3b scolor;
	float ephoto = 0;
	for (map<int, Vec3b>::iterator it = sampled_colors.begin(); it != sampled_colors.end(); ++it) {
		scolor = (*it).second;
		ephoto += pow(static_cast<float>(scolor[0]) - static_cast<float>(color[0]), 2) + pow(static_cast<float>(scolor[1]) - static_cast<float>(color[1]), 2) + pow(static_cast<float>(scolor[2]) - static_cast<float>(color[2]), 2);
	}
	return ephoto;
}

Vec3f NVS::Gradient_Ephoto(Vec3b color, map<int, Vec3b> sampled_colors) {
	Vec3b scolor;
	Vec3f grad(0.0, 0.0, 0.0);
	for (map<int, Vec3b>::iterator it = sampled_colors.begin(); it != sampled_colors.end(); ++it) {
		scolor = (*it).second;
		grad[0] += static_cast<float>(color[0]) - static_cast<float>(scolor[0]);
		grad[1] += static_cast<float>(color[1]) - static_cast<float>(scolor[1]);
		grad[2] += static_cast<float>(color[2]) - static_cast<float>(scolor[2]);
	}
	grad[0] = grad[0] / sampled_colors.size();
	grad[1] = grad[1] / sampled_colors.size();
	grad[2] = grad[2] / sampled_colors.size();
	//normalize(grad);
	return grad;
}

// minimum Ephoto across depths
// updates depth_idx_minz to hold the depth_idx at which Ephoto was minimized
float NVS::EphotoMinZ(Vec3b color, map<int, map<int, Vec3b>> sampled_colors_across_depths, int &depth_idx_minz) {
	float min_ephoto, ephoto;
	bool first = true;
	int depth_idx;
	for (map<int, map<int, Vec3b>>::iterator it = sampled_colors_across_depths.begin(); it != sampled_colors_across_depths.end(); ++it) {
		depth_idx = (*it).first;
		ephoto = Ephoto(color, (*it).second);
		if ((first) ||
			(ephoto < min_ephoto)) {
			min_ephoto = ephoto;
			depth_idx_minz = depth_idx;
		}
		first = false;
	}
	return min_ephoto;
}

// compute gradient of Ephoto at z which minimizes Ephoto
Vec3f NVS::Gradient_EphotoMinZ(Vec3b color, map<int, map<int, Vec3b>> sampled_colors_across_depths) {
	int depth_idx_minz;
	float Ephoto = EphotoMinZ(color, sampled_colors_across_depths, depth_idx_minz);

	Vec3f grad = Gradient_Ephoto(color, sampled_colors_across_depths[depth_idx_minz]);

	return grad;
}

Vec3b NVS::AvgColors(map<int, Vec3b> colors) {
	float bavg = 0;
	float gavg = 0;
	float ravg = 0;
	for (map<int, Vec3b>::iterator it = colors.begin(); it != colors.end(); ++it) {
		int cid = (*it).first;
		Vec3b color = (*it).second;
		bavg += static_cast<float>(color[0]);
		gavg += static_cast<float>(color[1]);
		ravg += static_cast<float>(color[2]);
	}
	float num_colors = static_cast<float>(colors.size());
	bavg /= num_colors;
	gavg /= num_colors;
	ravg /= num_colors;

	Vec3b avgcolor;
	avgcolor[0] = min(255, round(bavg, 0));// bavg;
	avgcolor[1] = min(255, round(gavg, 0));// gavg;
	avgcolor[2] = min(255, round(ravg, 0));// ravg;

	return avgcolor;
}

void NVS::DisplayEpigraph(map<int, map<int, Vec3b>> samples, int max_cam_id) {
	int num_depths = samples.size();
	if (num_depths == 0) return;
	int pixel_block_size = 4;
	int h = (max_cam_id+1) * pixel_block_size;
	int w = num_depths * pixel_block_size;
	Mat img = Mat::zeros(h, w, CV_8UC3);

	//cout << "img size h " << h << " and w " << w << endl;
	
	int idx_depth = 0;
	for (map<int, map<int, Vec3b>>::iterator it1 = samples.begin(); it1 != samples.end(); ++it1) {
		for (map<int, Vec3b>::iterator it2 = (*it1).second.begin(); it2 != (*it1).second.end(); ++it2) {
			int cid = (*it2).first;
			Vec3b color = (*it2).second;

			for (int ih = 0; ih < pixel_block_size; ih++) {
				for (int iw = 0; iw < pixel_block_size; iw++) {
					//cout << "filling image at (" << cid + ih << ", " << idx_depth + iw << ")" << endl;
					img.at<Vec3b>(cid*pixel_block_size + ih, idx_depth*pixel_block_size + iw) = color;
				}
			}

		}
		idx_depth++;
	}

	display_mat(&img, "NVS::DisplayEpigraph()");
}

void NVS::Sharpen(Mat *imgT, Mat *imgD, Mat *imgM, std::vector<int> cam_ids, Matrix<float, 3, 4> Pvirtual, Point3d view_dir_virtual) {
	bool debug_build = false;
	bool debug_modes = false;
	bool debug_match = false;
	bool debug_iter = false;
	bool timing = true;
	bool timing_loop2 = false;
	double t, t_last, t_loop, t_all;

	if (timing) t_all = (double)getTickCount();

	float DEPTH_ERROR_RANGE = 0.05; // number of world space units (meters) within which depth values may vary due to expected algorithm error in one direction in NVS::Sharpen()
	float DEPTH_STEP = 0.002; // number of world space units (meters) to step on depth when searching pixels along epilines in NVS::Sharpen()
	int NUM_NVS_SHARPEN_ETEXTURE_ITERATIONS = 30; // number of Etexture patch-matching iterations to perform in NVS::Sharpen()
	int NVS_SHARPEN_PATCH_SIZE = 5; // must be odd and >=3; number of pixels on a side to a patch in NVS::Sharpen()
	float NVS_LAMDA = 0.25;
	int NVS_NUM_MODES = 4;
	int GD_NUM_ITERATIONS = 20;
	int GD_NUM_STEPS = 12;
	float GD_LAMBDA = 0.75; // multiplicative factor applied to gradient at each step; if gradient is normalized, this gives the BGR space color distance to travel at each iteration

	int patch_side = (NVS_SHARPEN_PATCH_SIZE - 1) / 2;
	int patch_pixels = NVS_SHARPEN_PATCH_SIZE * NVS_SHARPEN_PATCH_SIZE;
	int patch_idx_center = (patch_pixels - 1) / 2;

	// determine projection matrices
	map<int, Matrix<float, 4, 3>> Ps;
	Matrix<float, 1, 4> ext;
	ext << 0., 0., 0., 1.;
	Matrix4f Pvirtual_ext;
	Pvirtual_ext << Pvirtual, ext;
	Matrix4f Pvirtual_ext_inv = Pvirtual_ext.inverse();

	int max_cam_id = -1;
	for (vector<int>::iterator it = cam_ids.begin(); it != cam_ids.end(); ++it) {
		int cid = (*it);
		Matrix<float, 3, 4> P1 = Ps_[cid].cast<float>() * Pvirtual_ext_inv;
		Matrix<float, 4, 3> P2 = P1.transpose();
		Ps[cid] = P2;

		if (cid > max_cam_id) max_cam_id = cid;
	}

	uchar *pM;
	float *pD;
	Vec3b *pT;
	float h;
	int x, y;
	map<int, int> x_last, y_last; // map of camera ID => last coordinate
	Matrix<float, 1, 4> WC(1, 4); // data structure containing homogeneous pixel positions across columns (u,v,1)
	WC(0, 2) = 1;
	float depth_val, disp_val;
	int idx_full_CM_dest, idx_full_CM_src;
	Matrix<int, Dynamic, 3> patch(patch_pixels, 3);
	Matrix<int, Dynamic, Dynamic> patch_blue(NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
	Matrix<int, Dynamic, Dynamic> patch_green(NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
	Matrix<int, Dynamic, Dynamic> patch_red(NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);

	map<int, map<int, map<int, Vec3b>>> candidate_colors_all; // map of CM pixel index => map of depth index => map of camera ID => color; depths range from (depth_val - DEPTH_ERROR_RANGE) to (depth_val + DEPTH_ERROR_RANGE) with steps of DEPTH_STEP
	map<int, vector<Matrix<int, Dynamic, 3>>> patch_library_all; // map of CM pixel index => candidate color
	//map<int, vector<pair<float, Vec3b>>> color_modes_all; // map of CM pixel index => vector of pair of Ephoto and color mode (local minima centroids of gradient descent result colors)
	map<int, vector<Vec3b>> color_modes_all; // map of CM pixel index => vector of color modes

	if (timing) t_loop = (double)getTickCount();

	// determine candidate color modes by examining the epiline in each image for colors at valid depths near current depth value from imgD that are also within a given depth range. Order by Ephoto and retain up to 5 with the smallest photoconsistency energies as the modes.  Compute photoconsistency energy for each candidate color as the sum of the squared error when the color is compared against all colors at a depth z, then take the largest Ephoto for the candidate color across all the depths.

	// enumerate colors at each depth in question for each masked-in pixel; also build patch library for each masked-in pixel
	for (int r = patch_side; r < (imgT->rows - patch_side); r++) { // for each used pixel in imgT with enough room to get 5x5 patch
		pM = imgM->ptr<uchar>(r);
		pD = imgD->ptr<float>(r);
		pT = imgT->ptr<Vec3b>(r);
		for (int c = patch_side; c < (imgT->cols - patch_side); c++) { // for each used pixel in imgT with enough room to get 5x5 patch
			if (!pM[c]) continue; // for each masked-in pixel according to imgM

			idx_full_CM_dest = PixIndexFwdCM(Point(c, r), imgT->rows);

			if (debug_build) cout << "matching (c, r) of (" << c << ", " << r << ")" << endl;

			if (debug_build) {
				// draw the point in question on the virtual image and display it
				cv::Scalar color(0, 0, 255);
				cv::Mat outImg1(imgT->rows, imgT->cols, CV_8UC3);
				imgT->copyTo(outImg1);
				cv::circle(outImg1, Point(c, r), 3, color, -1, CV_AA);
				display_mat(&outImg1, "point in question");
			}

			vector<Matrix<int, Dynamic, 3>> patch_library;
			map<int, map<int, Vec3b>> candidate_colors_acrossdepths;

			WC(0, 0) = c;
			WC(0, 1) = r;
			depth_val = pD[c];
			int depth_idx = 0;
			for (vector<int>::iterator it = cam_ids.begin(); it != cam_ids.end(); ++it) {
				int cid = (*it);
				x_last[cid] = -1;
				y_last[cid] = -1;
			}
			for (float depth_val_curr = (depth_val - DEPTH_ERROR_RANGE); depth_val_curr <= (depth_val + DEPTH_ERROR_RANGE); depth_val_curr += DEPTH_STEP) {
				if (depth_val_curr == 0) continue;

				map<int, Vec3b> candidate_colors; // map of camera ID => color at the current depths for the current pixel

				disp_val = 1 / depth_val_curr;
				WC(0, 3) = disp_val;

				for (vector<int>::iterator it = cam_ids.begin(); it != cam_ids.end(); ++it) {
					int cid = (*it);

					if (debug_build) cout << "checking cid " << cid << endl;

					double dot = scene_->cameras_[cid]->view_dir_.ddot(view_dir_virtual);
					if (dot <= 0) continue; // only consider patches from cameras facing same general direction

					if ((depth_val_curr < scene_->min_depths_[cid]) ||
						(depth_val_curr > scene_->max_depths_[cid]))
						continue;

					Matrix<float, 1, 3> T = WC * Ps[cid];
					h = T(0, 2);
					x = round(T(0, 0) / h);
					y = round(T(0, 1) / h);

					//if (debug_build) cout << "found (x, y) of (" << x << ", " << y << ")" << endl;

					if ((x == x_last[cid]) &&
						(y == y_last[cid]))
						continue; // already investigated this pixel

					if ((x < 0) ||
						(x >= widths_[cid]) ||
						(y < 0) ||
						(y >= heights_[cid]))
						continue;

					//if (debug) {
					// draw the point in question on the source mask and display it
					//	if (debug) cout << "drawing mask for cid " << cid << endl;
					//	DisplayImages::DisplayGrayscaleImage(&masks_[cid], heights_[cid], widths_[cid]);
					//}

					idx_full_CM_src = PixIndexFwdCM(Point(x, y), heights_[cid]);
					if (!masks_[cid](idx_full_CM_src, 0)) continue; // masked-out pixel

					if (debug_build) {
						// draw the point in question on the source image and display it
						cout << "drawing cid " << cid << endl;
						cv::Scalar color(0, 0, 255);
						cv::Mat outImg2(heights_[cid], widths_[cid], CV_8UC3);
						imgsT_[cid].copyTo(outImg2);
						cv::circle(outImg2, Point(x, y), 3, color, -1, CV_AA);
						display_mat(&outImg2, "projection to source");
					}

					Vec3b color = imgsT_[cid].at<Vec3b>(y, x);
					candidate_colors[cid] = color;

					if (debug_build) {
						Mat imgCC = Mat::zeros(100, 100, CV_8UC3);
						imgCC.setTo(color);
						display_mat(&imgCC, "candidate color");
					}

					if ((y >= patch_side) &&
						(y < heights_[cid] - patch_side) &&
						(x >= patch_side) &&
						(x < widths_[cid] - patch_side)) {
						patch_blue = As_blue_[cid].block(y - patch_side, x - patch_side, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
						patch_green = As_green_[cid].block(y - patch_side, x - patch_side, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
						patch_red = As_red_[cid].block(y - patch_side, x - patch_side, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);

						//if (debug_build) DisplayImages::DisplayGrayscaleImage(&patch_blue, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
						//if (debug_build) DisplayImages::DisplayGrayscaleImage(&patch_green, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
						//if (debug_build) DisplayImages::DisplayGrayscaleImage(&patch_red, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);

						patch_blue.resize(patch_pixels, 1);
						patch_green.resize(patch_pixels, 1);
						patch_red.resize(patch_pixels, 1);
						patch.col(0) = patch_blue;
						patch.col(1) = patch_green;
						patch.col(2) = patch_red;
						patch_library.push_back(patch);
						patch_blue.resize(NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
						patch_green.resize(NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
						patch_red.resize(NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);

						if (debug_build) DisplayImages::DisplayBGRImage(&patch, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
					}

					x_last[cid] = x;
					y_last[cid] = y;
				}
				if (candidate_colors.size() != 0)
					candidate_colors_acrossdepths[depth_idx] = candidate_colors;
				depth_idx++;
			}
			if (patch_library.size() != 0)
				patch_library_all[idx_full_CM_dest] = patch_library;
			if (candidate_colors_acrossdepths.size() != 0)
				candidate_colors_all[idx_full_CM_dest] = candidate_colors_acrossdepths;
		}
	}

	if (timing) {
		t = (double)getTickCount() - t_loop;
		t_loop = (double)getTickCount();
		cout << "NVS::Sharpen() execution time for building list of candidate colors and patch libraries for all pixels " << t*1000. / getTickFrequency() << " ms" << endl;
	}

	/*
	// instead of using gradient descent to compute candidates for modes, order existing candidate colors by EphotoMinZ for each pixel and take the top 4 for each pixel
	for (int r = patch_side; r < (imgT->rows - patch_side); r++) { // for each used pixel in imgT with enough room to get 5x5 patch
		pM = imgM->ptr<uchar>(r);
		pT = imgT->ptr<Vec3b>(r);
		for (int c = patch_side; c < (imgT->cols - patch_side); c++) { // for each used pixel in imgT with enough room to get 5x5 patch
			if (!pM[c]) continue; // for each masked-in pixel according to imgM

			idx_full_CM_dest = PixIndexFwdCM(Point(c, r), imgT->rows);

			std::vector<std::pair<float, int>> candidate_colors_energies;
			map<int, Vec3b> candidate_colors;
			float Ephoto;
			int zatmin;
			int cc_idx = 0;
			for (map<int, map<int, Vec3b>>::iterator it1 = candidate_colors_all[idx_full_CM_dest].begin(); it1 != candidate_colors_all[idx_full_CM_dest].end(); ++it1) {
				for (map<int, Vec3b>::iterator it2 = (*it1).second.begin(); it2 != (*it1).second.end(); ++it2) {
					Vec3b curr_color = (*it2).second;
					Ephoto = EphotoMinZ(curr_color, candidate_colors_all[idx_full_CM_dest], zatmin);
					pair<float, int> colorpair;
					colorpair.first = Ephoto;
					colorpair.second = cc_idx;
					candidate_colors[cc_idx] = curr_color;
					candidate_colors_energies.push_back(colorpair);
					cc_idx++;
				}
			}
			std::sort(candidate_colors_energies.begin(), candidate_colors_energies.end());
			int modenum = 0;
			for (std::vector<std::pair<float, int>>::iterator it = candidate_colors_energies.begin(); it != candidate_colors_energies.end(); ++it) {
				if (modenum > 4) break;
				int i = (*it).second;
				pair<float, Vec3b> cpair;
				cpair.first = (*it).first;
				cpair.second = candidate_colors[i];
				color_modes_all[idx_full_CM_dest].push_back(cpair);
				
				if (modenum == 0) pT[c] = candidate_colors[i]; // assign the best color to this pixel

				modenum++;
			}
		}
	}
	*/

	/*
	// for debugging
	for (map<int, map<int, vector<Vec3b>>>::iterator it1 = candidate_colors_all.begin(); it1 != candidate_colors_all.end(); ++it1) {
		int idx_pix = (*it1).first;
		cout << endl << endl << "candidate_colors_all CM pixel index " << idx_pix << endl;
		for (map<int, vector<Vec3b>>::iterator it2 = (*it1).second.begin(); it2 != (*it1).second.end(); ++it2) {
			int idx_depth = (*it2).first;
			cout << ".....depth index " << idx_depth << endl;
			for (vector<Vec3b>::iterator it3 = (*it2).second.begin(); it3 != (*it2).second.end(); ++it3) {
				Vec3b c = (*it3);
				cout << "..........color " << static_cast<int>(c[0]) << ", " << static_cast<int>(c[1]) << ", " << static_cast<int>(c[2]) << endl;
			}
		}
		cin.ignore();
	}
	*/

	// since the space of colors in V and z is small, instead of gradient descent simply enumerate the average color at each depth
	// also, don't cluster the colors because it's less likely the are modes from different parts of the object (and none from other objects in the scene)
	for (int r = patch_side; r < (imgT->rows - patch_side); r++) { // for each used pixel in imgT with enough room to get 5x5 patch
		pM = imgM->ptr<uchar>(r);
		pT = imgT->ptr<Vec3b>(r);
		for (int c = patch_side; c < (imgT->cols - patch_side); c++) { // for each used pixel in imgT with enough room to get 5x5 patch
			if (!pM[c]) continue; // for each masked-in pixel according to imgM

			idx_full_CM_dest = PixIndexFwdCM(Point(c, r), imgT->rows);

			vector<Vec3b> modecolors;
			for (map<int, map<int, Vec3b>>::iterator it = candidate_colors_all[idx_full_CM_dest].begin(); it != candidate_colors_all[idx_full_CM_dest].end(); ++it) {
				int depth_idx = (*it).first;

				if ((*it).second.size() == 0) continue; // no colors at this depth, which shouldn't occur

				Vec3b avgcolor = AvgColors((*it).second);
				modecolors.push_back(avgcolor);
			}

			// sort modes by Ephoto and assign to this pixel
			//std::sort(modecolors.begin(), modecolors.end(), pairCompareModes);
			color_modes_all[idx_full_CM_dest] = modecolors;
		}
	}

	/*
	// perform gradient descent on color space using minz(Ephoto) values at each color to determine modes, then cluster the results into 4 modes; use 20 iterations beginning at random starting colors and perform 12 gradient descent steps for each iteration
	for (int r = patch_side; r < (imgT->rows - patch_side); r++) { // for each used pixel in imgT with enough room to get 5x5 patch
		pM = imgM->ptr<uchar>(r);
		pT = imgT->ptr<Vec3b>(r);
		for (int c = patch_side; c < (imgT->cols - patch_side); c++) { // for each used pixel in imgT with enough room to get 5x5 patch
			if (!pM[c]) continue; // for each masked-in pixel according to imgM

			idx_full_CM_dest = PixIndexFwdCM(Point(c, r), imgT->rows);

			if (debug_modes)
				DisplayEpigraph(candidate_colors_all[idx_full_CM_dest], max_cam_id);

			// perform gradient descent on this pixel
			Vec3b curr_color;
			Vec3f grad;
			vector<Vec3b> gd_results; // map of CM pixel index => gradient descent color results
			for (int iter = 0; iter < GD_NUM_ITERATIONS; iter++) {
				curr_color[0] = 255. * (static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)));
				curr_color[1] = 255. * (static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)));
				curr_color[2] = 255. * (static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)));
				if (debug_modes) {
					Mat imgCC = Mat::zeros(100, 100, CV_8UC3);
					imgCC.setTo(curr_color);
					display_mat(&imgCC, "gradient random initial value");
				}
				for (int step = 0; step < GD_NUM_STEPS; step++) {
					grad = Gradient_EphotoMinZ(curr_color, candidate_colors_all[idx_full_CM_dest]);
					if (debug_modes) cout << "grad " << grad[0] << ", " << grad[1] << ", " << grad[2] << endl;
					curr_color[0] = curr_color[0] - round(GD_LAMBDA * grad[0]); // subtract scaled gradient to move toward local minimum
					curr_color[1] = curr_color[1] - round(GD_LAMBDA * grad[1]); // subtract scaled gradient to move toward local minimum
					curr_color[2] = curr_color[2] - round(GD_LAMBDA * grad[2]); // subtract scaled gradient to move toward local minimum
					if (debug_modes) {
						Mat imgCC = Mat::zeros(100, 100, CV_8UC3);
						imgCC.setTo(curr_color);
						display_mat(&imgCC, "gradient step");
					}
				}
				gd_results.push_back(curr_color);

				if (debug_modes) {
					Mat imgCC = Mat::zeros(100, 100, CV_8UC3);
					imgCC.setTo(curr_color);
					display_mat(&imgCC, "gradient result color");
				}
			}

			// cluster gradient descent results to determine color modes for each pixel; limit to 4 modes
			int num_candidates = gd_results.size();
			Mat points = Mat::zeros(num_candidates, 3, CV_32FC3); // number of cols equals number of pixel color possibilities; number of rows equals 3 for BGR channels; values are float; (data down each column)
			int cc = 0;
			for (vector<Vec3b>::iterator it = gd_results.begin(); it != gd_results.end(); ++it) {
				Vec3f val;
				val[0] = static_cast<float>((*it)[0]);
				val[1] = static_cast<float>((*it)[1]);
				val[2] = static_cast<float>((*it)[2]);
				points.at<Vec3f>(cc, 0) = val;

				cc++;
			}
			Mat bestLabels;
			Mat modes;
			cv::kmeans(points, NVS_NUM_MODES, bestLabels,
				TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
				3, KMEANS_PP_CENTERS, modes); // centers will be a matrix of size kx1 and type CV_32FC3 where each column holds the BGR color of a mode center; can sort by number of appearances of each mode index in bestLabels, with more occurrences meaning a better mode

			Vec3b modecolor;
			float best_Ephoto;
			Vec3b bestcolor;
			bool firstcolor = true;
			std::vector<std::pair<float, Vec3b>> modecolors;
			for (int m = 0; m < NVS_NUM_MODES; m++) {
				Vec3f val = modes.at<Vec3f>(m, 0);
				modecolor[0] = static_cast<int>(val[0]);
				modecolor[1] = static_cast<int>(val[1]);
				modecolor[2] = static_cast<int>(val[2]);

				int zatmin;
				float Ephoto = EphotoMinZ(modecolor, candidate_colors_all[idx_full_CM_dest], zatmin);

				std::pair<float, Vec3b> modepair;
				modepair.first = Ephoto;
				modepair.second = modecolor;
				modecolors.push_back(modepair);

				if ((firstcolor) ||
					(Ephoto < best_Ephoto)) {
					bestcolor = modecolor;
					best_Ephoto = Ephoto;
					firstcolor = false;
				}
			}

			// sort modes by Ephoto and assign to this pixel
			//std::sort(modecolors.begin(), modecolors.end(), pairCompareModes);
			color_modes_all[idx_full_CM_dest] = modecolors;

			// assign the best color to this pixel
			//pT[c] = modecolors.begin()->second;
			pT[c] = bestcolor;
		}
	}
	*/
	
	if (timing) {
		t = (double)getTickCount() - t_loop;
		t_loop = (double)getTickCount();
		cout << "NVS::Sharpen() execution time for determining list of mode colors for all pixels " << t*1000. / getTickFrequency() << " ms" << endl;
	}

	// apply texture priors
	for (int iter = 0; iter < NUM_NVS_SHARPEN_ETEXTURE_ITERATIONS; iter++) {
		// for each used pixel in imgT with enough room to get 5x5 patch, find closest matching patch from library (minimize energy for all pixels except center pixel) and replace center pixel with the one that most closely matches the center pixel of the matching patch
		Matrix<int, Dynamic, 3> patch(patch_pixels, 3);
		Matrix<int, Dynamic, 3> matching_patch; // note: matching_patch only used for debugging, so remove it ***************************************************************************************
		Matrix<int, Dynamic, 3> display_patch; // remove this line ***************************************************************************************
		Vec3b color;
		int i;
		Vec3b *pT;
		for (int r = patch_side; r < (imgT->rows - patch_side); r++) {
			pM = imgM->ptr<uchar>(r);
			pT = imgT->ptr<Vec3b>(r);
			for (int c = patch_side; c < (imgT->cols - patch_side); c++) {
				if (!pM[c]) continue;

				idx_full_CM_dest = PixIndexFwdCM(Point(c, r), imgT->rows);

				// if there are no color modes or there is no patch library for the pixel, continue
				if ((color_modes_all.find(idx_full_CM_dest) == color_modes_all.end()) ||
					(patch_library_all.find(idx_full_CM_dest) == patch_library_all.end()))
					continue;

				if (timing_loop2) t_last = (double)getTickCount();

				// build this patch
				i = 0;
				for (int c_adj = -patch_side; c_adj <= patch_side; c_adj++) {
					for (int r_adj = -patch_side; r_adj <= patch_side; r_adj++) {
						color = imgT->at<Vec3b>(r + r_adj, c + c_adj);
						patch(i, 0) = color[0];
						patch(i, 1) = color[1];
						patch(i, 2) = color[2];
						i++;
					}
				}
				
				if (debug_match) {
					// display current patch
					cout << "current patch" << endl;
					display_patch = patch;
					DisplayImages::DisplayBGRImage(&display_patch, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
				}

				if (timing_loop2) {
					t = (double)getTickCount() - t_last;
					t_last = (double)getTickCount();
					cout << "NVS::Sharpen() execution time for patch building at (" << c << ", " << r << ") = " << t*1000. / getTickFrequency() << " ms" << endl;
				}

				// find closest match from patch library
				Vec3b lpatch_centercolor;
				float lowest_energy, energy;
				bool first = true;
				for (vector<Matrix<int, Dynamic, 3>>::iterator it = patch_library_all[idx_full_CM_dest].begin(); it != patch_library_all[idx_full_CM_dest].end(); ++it) {
					
					//if (debug_match) {
						// display candidate patch
					//	cout << "candidate patch" << endl;
					//	display_patch = (*it);
					//	DisplayImages::DisplayBGRImage(&display_patch, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
					//}
					
					energy = Etexture(patch, (*it), patch_idx_center);
					if ((first) ||
						(energy < lowest_energy)) {
						lowest_energy = energy;
						lpatch_centercolor[0] = (*it)(patch_idx_center, 0);
						lpatch_centercolor[1] = (*it)(patch_idx_center, 1);
						lpatch_centercolor[2] = (*it)(patch_idx_center, 2);

						matching_patch = (*it); // used for debugging, so remove this line ****************************************************************************************************
					}
					first = false;
				}

				if (debug_match) {
					// display matching patch
					cout << "matching patch" << endl;
					display_patch = matching_patch;
					DisplayImages::DisplayBGRImage(&display_patch, NVS_SHARPEN_PATCH_SIZE, NVS_SHARPEN_PATCH_SIZE);
				}

				if (timing_loop2) {
					t = (double)getTickCount() - t_last;
					t_last = (double)getTickCount();
					cout << "NVS::Sharpen() execution time for patch matching at(" << c << ", " << r << ") = " << t*1000. / getTickFrequency() << " ms" << endl;
				}

				// Vr = (Vr-1 + lamda*Tc)/(1+lamda) where Tc is the center pixel of the matching patch
				Vec3b currcolor = pT[c];
				Vec3b newcolor;
				newcolor[0] = static_cast<unsigned char>((static_cast<float>(currcolor[0]) + NVS_LAMDA*static_cast<float>(lpatch_centercolor[0])) / (1 + NVS_LAMDA));
				newcolor[1] = static_cast<unsigned char>((static_cast<float>(currcolor[1]) + NVS_LAMDA*static_cast<float>(lpatch_centercolor[1])) / (1 + NVS_LAMDA));
				newcolor[2] = static_cast<unsigned char>((static_cast<float>(currcolor[2]) + NVS_LAMDA*static_cast<float>(lpatch_centercolor[2])) / (1 + NVS_LAMDA));

				// determine best color from modes as one that is closest to Vr
				float sqdist, closest_sqdist;
				closest_sqdist = -1;
				Vec3b newcolor_ofmodes, modecolor;
				for (vector<Vec3b>::iterator it = color_modes_all[idx_full_CM_dest].begin(); it != color_modes_all[idx_full_CM_dest].end(); ++it) {
					modecolor = (*it);
					//sqdist = (modecolor[0] - lpatch_centercolor[0]) ^ 2 + (modecolor[1] - lpatch_centercolor[1]) ^ 2 + (modecolor[2] - lpatch_centercolor[2]) ^ 2;
					sqdist = pow(static_cast<float>(modecolor[0]) - static_cast<float>(newcolor[0]), 2) + pow(static_cast<float>(modecolor[1]) - static_cast<float>(newcolor[1]), 2) + pow(static_cast<float>(modecolor[2]) - static_cast<float>(newcolor[2]), 2);
					if ((closest_sqdist == -1) ||
						(sqdist < closest_sqdist)) {
						closest_sqdist = sqdist;
						newcolor_ofmodes[0] = modecolor[0];
						newcolor_ofmodes[1] = modecolor[1];
						newcolor_ofmodes[2] = modecolor[2];
					}
				}

				pT[c] = newcolor_ofmodes;

				if (debug_match) {
					Mat imgCC = Mat::zeros(100, 100, CV_8UC3);
					imgCC.setTo(newcolor);
					display_mat(&imgCC, "newcolor");
				}

				if (timing_loop2) {
					t = (double)getTickCount() - t_last;
					t_last = (double)getTickCount();
					cout << "NVS::Sharpen() execution time for color matching at(" << c << ", " << r << ") = " << t*1000. / getTickFrequency() << " ms" << endl;
				}
			}
		}
		
		if (timing) {
			t = (double)getTickCount() - t_loop;
			t_loop = (double)getTickCount();
			cout << "NVS::Sharpen() execution time for updating colors through matching for all pixels in iteration " << (iter + 1) << " of " << NUM_NVS_SHARPEN_ETEXTURE_ITERATIONS  << " iterations " << t*1000. / getTickFrequency() << " ms" << endl;
		}

		if (debug_iter) display_mat(imgT, "imgT");
	}
	
	if (timing) {
		t = (double)getTickCount() - t_all;
		cout << "NVS::Sharpen() total " << t*1000. / getTickFrequency() << " ms" << endl;
		//cin.ignore();
	}
}

float NVS::Etexture(Matrix<int, Dynamic, 3> patch1, Matrix<int, Dynamic, 3> patch2, int center_pixel_idx) {
	patch1 -= patch2;
	patch1 = patch1.cwiseProduct(patch1);
	float val = patch1.sum() - patch1.row(center_pixel_idx).sum(); // ignore center pixels in computation - only consider neighbors in patch
	//val = log(val); //-log(val);
	return val;
}