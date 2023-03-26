#include "Scene.h"

Scene::Scene() {
	depth_downscale_ = 1.;
	bv_min_ = Point3d(0., 0., 0.);
	bv_max_ = Point3d(0., 0., 0.);
}

Scene::~Scene() {
	for (std::map<int, Sensor*>::iterator it = sensors_.begin(); it != sensors_.end(); ++it) {
		int id = (*it).first;
		delete sensors_[id];
	}
	sensors_.erase(sensors_.begin(), sensors_.end());

	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int id = (*it).first;
		delete cameras_[id];
	}
	cameras_.erase(cameras_.begin(), cameras_.end());
}

void Scene::Init(std::string name) {
	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();// mask, priority values, fix images

	name_ = name;

	xml_document<> doc;
	xml_node<> *root_node, *chunk_node, *curr_node;
	// Read the xml file into a vector
	ifstream theFile(GLOBAL_FILEPATH_DATA + name_ + "\\" + GLOBAL_FOLDER_AGISOFT + "\\doc.xml"); // doc.xml is the default name of the xml doc created by Agisoft
	vector<char> buffer((istreambuf_iterator<char>(theFile)), istreambuf_iterator<char>());
	buffer.push_back('\0');
	// Parse the buffer using the xml file parsing library into doc 
	doc.parse<0>(&buffer[0]);
	// Find our root node
	root_node = doc.first_node("document");
	chunk_node = root_node->first_node("chunk");
	//name_ = chunk_node->first_attribute("label")->value();
	//replace(name_.begin(), name_.end(), ' ', '_'); // replace all ' ' with '_' to avoid filename issues when use this name_ to construct filenames

	std::string s;
	
	// import AgisoftToWorld transformation matrix for the chunk
	AgisoftToWorld_.setIdentity(); // default if fail to find it
	xml_node<> *chunk_transform_node;
	chunk_transform_node = chunk_node->first_node("transform");
	if (chunk_transform_node != 0) {
		s = chunk_transform_node->value();
		if (!s.empty()) {
			WorldToAgisoft_ = ParseString_Matrixd(s, 4, 4); // 4x4 matrix specifying chunk location in the world coordinate system
			AgisoftToWorld_ = WorldToAgisoft_.inverse();
			agisoft_to_world_scale_ = 1. / ScaleFromExtrinsics(AgisoftToWorld_);
		}
	}
	else {
		WorldToAgisoft_.setIdentity();
		AgisoftToWorld_.setIdentity();
		agisoft_to_world_scale_ = 1.;
		cerr << "No Agisoft chunk transform found - please check to ensure scale has been added to the Agisoft scene before continuing" << endl;
	}

	// import sensors
	xml_node<> *sensors_node;
	sensors_node = chunk_node->first_node("sensors");
	for (xml_node<> * sensor_node = sensors_node->first_node("sensor"); sensor_node; sensor_node = sensor_node->next_sibling()) {
		Sensor *sensor = new Sensor();
		sensor->Init(sensor_node);
		sensors_[sensor->id_] = sensor;
	}

	// import cameras
	xml_node<> *cameras_node;
	cameras_node = chunk_node->first_node("cameras");
	for (xml_node<> * camera_node = cameras_node->first_node("camera"); camera_node; camera_node = camera_node->next_sibling()) {
		Camera *cam = new Camera();
		cam->Init(name_, camera_node, AgisoftToWorld_, WorldToAgisoft_);
		cam->InitSensor(sensors_[cam->sensor_id_]);
		cameras_[cam->id_] = cam;
	}

	CullCameras();

	// set up depth map nodes
	xml_node<> *depth_maps_node, *frame_node;
	depth_maps_node = chunk_node->first_node("depth_maps");
	frame_node = depth_maps_node->first_node("frame");

	// find downsampling factor for depth maps
	curr_node = frame_node->first_node("meta");
	if (curr_node != 0) {
		curr_node = curr_node->first_node("property");
		if (curr_node != 0) {
			s = curr_node->first_attribute("name")->value();
			if (s == "depth/depth_downscale") {
				s = curr_node->first_attribute("value")->value();
				depth_downscale_ = convert_string_to_int(s);
			}
		}
	}

	// import depth maps for cameras
	for (xml_node<> * depthmap_node = frame_node->first_node("depth_map"); depthmap_node; depthmap_node = depthmap_node->next_sibling()) {
		if (strcmp(depthmap_node->name(), "meta") != 0) {
			s = depthmap_node->first_attribute("camera_id")->value();
			int cid = convert_string_to_int(s);
			if (cameras_.find(cid) == cameras_.end()) continue;

			cameras_[cid]->InitDepthMap(name_, depthmap_node, agisoft_to_world_scale_, depth_downscale_);
		}
	}

	// downsample to speed process
	if (GLOBAL_DOWNSAMPLE_FACTOR != 1) {
		for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
			int cid = (*it).first;
			if ((!(*it).second->enabled_) ||
				(!(*it).second->posed_) ||
				(!(*it).second->has_depth_map_))
				continue;
			(*it).second->DownsampleAll(GLOBAL_DOWNSAMPLE_FACTOR);
		}
	}

	// undistort images
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		(*it).second->UndistortPhotos();
	}

	// compute world space point projections
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		if (((*it).second->enabled_) &&
			 ((*it).second->posed_) &&
			 ((*it).second->has_depth_map_))
			(*it).second->InitWorldSpaceProjection();
	}
	
	// create dilated masks for use as approximate masks to accomodate error in camera pose when testing reprojection against masks
	int morph_type = MORPH_RECT; // MORPH_ELLIPSE
	int morph_size = GLOBAL_MASK_DILATION; // get rid of small regional markings
	Mat element = getStructuringElement(morph_type,
		Size(2 * morph_size + 1, 2 * morph_size + 1),
		Point(morph_size, morph_size));
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;
		if ((!(*it).second->enabled_) ||
			(!(*it).second->posed_))
			continue;

		Mat mask_dilated = cv::Mat::zeros((*it).second->imgMask_.rows, (*it).second->imgMask_.cols, CV_8UC1);
		if (GLOBAL_MASK_DILATION > 0) dilate((*it).second->imgMask_, mask_dilated, element);
		else (*it).second->imgMask_.copyTo(mask_dilated);
		Matrix<uchar, Dynamic, Dynamic> mask1((*it).second->imgMask_.rows, (*it).second->imgMask_.cols);
		EigenOpenCV::cv2eigen(mask_dilated, mask1);
		mask1.resize(mask1.rows()*mask1.cols(), 1);
		Matrix<bool, Dynamic, 1> mask2 = mask1.cast<bool>();
		masks_dilated_[cid] = mask2;
	}
	//FilterCamerasByPoseAccuracy(); // downside of using this function is that if Agisoft didn't do a great job on depth computations because it was a difficult piece, it will make it appear as if pose estimation was bad when it wasn't, and may unnecessarily eliminate too many cameras to effectively continue
	if (!GLOBAL_LOAD_COMPUTED_DISPARITY_MAPS) {
		CleanDepths(); // clean depth maps of untrusted values (that cause points to project outside the image mask for any other image)
		/*
		map<int, Matrix<float, Dynamic, Dynamic>> depth_maps_tmp; // record current depth maps because: the crowd wisdom depths may supercede current depths if they're closer, regardless of whether they are valid depths given potential camera pose error that makes them  valid from one camera but fail validity from another.  If supercede them and then zeroed out during second CleanDepths() run, will need to reinstate the original depth values, so record them here temporarily
		for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
			if ((!(*it).second->enabled_) ||
				(!(*it).second->posed_) ||
				(!(*it).second->has_depth_map_))
				continue;
			int cid = (*it).first;
			depth_maps_tmp[cid] = (*it).second->dm_->depth_map_;
		}
		
		UpdateDepthsByCrowdWisdom();
		CleanDepths(); // clean depth maps of untrusted values (that cause points to project outside the image mask for any other image); run a second time after UpdateDepthsByCrowdWisdom() because that may result in depths outside mask bounds due to misalignment of camera poses
		// the crowd wisdom depths may supercede current depths if they're closer, regardless of whether they are valid depths given potential camera pose error that makes them  valid from one camera but fail validity from another.  If supercede them and then zeroed out during second CleanDepths() run, will need to reinstate the original depth values, so record them here temporarily
		
		for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
			if ((!(*it).second->enabled_) ||
				(!(*it).second->posed_) ||
				(!(*it).second->has_depth_map_))
				continue;

			int cid = (*it).first;
			int h = (*it).second->height_;
			int w = (*it).second->width_;
			float *pDn = cameras_[cid]->dm_->depth_map_.data(); // don't use (*it) because want it to effect changes outside of this loop
			float *pDo = depth_maps_tmp[cid].data();
			for (int c = 0; c < w; c++) {
				for (int r = 0; r < h; r++) {
					if (*pDn == 0.)
						*pDn = *pDo;
					pDn++;
					pDo++;
				}
			}
		}
		*/

		UpdateCamerasWSPoints();

		if (debug) {
			for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
				if ((!(*it).second->enabled_) ||
					(!(*it).second->posed_) ||
					(!(*it).second->has_depth_map_))
					continue;

				cout << "Displaying depth image for cid " << (*it).first << endl;
				(*it).second->dm_->DisplayDepthImage();
			}
		}
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Scene::Init() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// deletes cameras without extrinsic matrices
void Scene::CullCameras() {
	// Delete cameras without extrinsic matrices
	std::vector<int> del_cams;
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		if (!(*it).second->posed_) del_cams.push_back((*it).first);
	}
	for (std::vector<int>::iterator it = del_cams.begin(); it != del_cams.end(); ++it) {
		cameras_.erase((*it));
	}
}

// gets an ordered list of num closest cameras to view of I0 (closest first)
// I0_pos is the position of camera I0 in world space; I0_view_dir is the viewing direction of camera I0 in world space
// arg num must be a positive number; if it is greater than or equal to the number of cameras, an ordered list of all available cameras will be returned
// exclude_cam_ids holds a list of camera IDs that should be excluded from the list of closest camera IDs
// if cull_backfacing, ensures camera is looking in the same general direction as the given view_dir; otherwise does not
std::vector<int> Scene::GetClosestCams(Point3d I0_pos, Point3d I0_view_dir, std::vector<int> exclude_cam_ids, int num, bool cull_backfacing) {
	assert(num >= 0);
	std::vector<int> cams = OrderCams(I0_pos, I0_view_dir, cull_backfacing);
	if ((num < cams.size()) &&
		(num != 0))
		cams.erase(cams.begin() + num, cams.end());

	for (std::vector<int>::iterator it = exclude_cam_ids.begin(); it != exclude_cam_ids.end(); ++it) {
		std::vector<int>::iterator pos;
		pos = std::find(cams.begin(), cams.end(), (*it));
		if (pos == cams.end()) continue;
		cams.erase(pos);
	}

	return cams;
}

// returns an ordered list of camera IDs from member cameras according to the closest view to I0 (by view direction, not position)
// closest camera is first in the list, farthest last
// I0_pos is the position of camera I0 in world space; I0_view_dir is the viewing direction of camera I0 in world space
// if cull_backfacing, ensures camera is looking in the same general direction as the given view_dir; otherwise does not
std::vector<int> Scene::OrderCams(Point3d I0_pos, Point3d I0_view_dir, bool cull_backfacing) {
	bool debug = false;

	if (debug) cout << "I0_pos " << I0_pos << endl;

	std::vector<std::pair<int, double>> cam_dist; // vector of pairs of: camera ID, distance from I0
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		std::pair<int, double> c;
		if ((!(*it).second->enabled_) ||
			(!(*it).second->posed_) ||
			(!(*it).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information
		c.first = (*it).second->id_;
		double vdd = I0_view_dir.ddot((*it).second->view_dir_);
		c.second = vdd;
		//c.second = vecdist(I0_pos, (*it).second->pos_);
		if ((!cull_backfacing) ||
			(vdd > 0))  // ensure camera is looking in the same general direction as the given view_dir
			cam_dist.push_back(c);
	}
	std::sort(cam_dist.begin(), cam_dist.end(), pairCompare);
	std::reverse(cam_dist.begin(), cam_dist.end()); // since using dot product of views, put in order of largest dot product to smallest
	
	std::vector<int> cams_ordered;
	int cidi, cidj;
	bool pass;
	for (std::vector<std::pair<int, double>>::iterator iti = cam_dist.begin(); iti != cam_dist.end(); ++iti) {
		cidi = (*iti).first;
		pass = true;
		for (std::vector<std::pair<int, double>>::iterator itj = cam_dist.begin(); itj != cam_dist.end(); ++itj) {
			cidj = (*itj).first;
			if (cidi == cidj) break; // break out of inner loop because have examined all nodes it2 closer to I0 than node it1; if Ij is closer to I0 than Ii is...

			
			//distij = vecdist(cameras_[cidi]->pos_, cameras_[cidj]->pos_);
			//if (distij < (*iti).second) { // ...and if Ii is closer to Ij than Ii is to I0, remove Ii from the list
			//	pass = false;
			//	break;
			//}
			
		}
		if (pass) cams_ordered.push_back(cidi);
	}

	if (debug) {
		cout << "Ordered list of camera IDs:" << endl;
		for (std::vector<std::pair<int, double>>::iterator iti = cam_dist.begin(); iti != cam_dist.end(); ++iti) {
			cout << (*iti).first << endl;
		}
	}

	return cams_ordered;
}

// debug printing
void Scene::Print() {
	for (std::map<int, Sensor*>::iterator it = sensors_.begin(); it != sensors_.end(); ++it) {
		(*it).second->Print();
	}

	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		(*it).second->Print();
	}
}

// determines point cloud bounding box volume across all camera views
void Scene::UpdatePointCloudBoundingVolume() {
	bool debug = false;

	bv_min_ = Point3d(0., 0., 0.);
	bv_max_ = Point3d(0., 0., 0.);
	bool first = true;
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;
		if ((!cameras_[cid]->posed_) ||
			(!cameras_[cid]->enabled_) ||
			(!cameras_[cid]->has_depth_map_)) continue;

		cameras_[cid]->UpdatePointCloudBoundingVolume();

		if (first) {
			bv_min_.x = cameras_[cid]->bv_min_.x;
			bv_min_.y = cameras_[cid]->bv_min_.y;
			bv_min_.z = cameras_[cid]->bv_min_.z;
			bv_max_.x = cameras_[cid]->bv_max_.x;
			bv_max_.y = cameras_[cid]->bv_max_.y;
			bv_max_.z = cameras_[cid]->bv_max_.z;
		}
		else {
			if (cameras_[cid]->bv_min_.x < bv_min_.x) bv_min_.x = cameras_[cid]->bv_min_.x;
			if (cameras_[cid]->bv_min_.y < bv_min_.y) bv_min_.y = cameras_[cid]->bv_min_.y;
			if (cameras_[cid]->bv_min_.z < bv_min_.z) bv_min_.z = cameras_[cid]->bv_min_.z;
			if (cameras_[cid]->bv_max_.x > bv_max_.x) bv_max_.x = cameras_[cid]->bv_max_.x;
			if (cameras_[cid]->bv_max_.y > bv_max_.y) bv_max_.y = cameras_[cid]->bv_max_.y;
			if (cameras_[cid]->bv_max_.z > bv_max_.z) bv_max_.z = cameras_[cid]->bv_max_.z;
		}
		first = false;
	}

	if (debug) {
		cout << "Bounding volume minimum " << bv_min_ << ", maximum " << bv_max_ << endl;
		cin.ignore();
	}
}

void Scene::TestMinMaxDepths() {
	bool debug = true;

	float established_min_depth, established_max_depth;
	float min_depth, max_depth;
	float dval;

	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;

		established_min_depth = min_depths_[cid]; // in camera space
		established_max_depth = max_depths_[cid]; // in camera space

		bool first = true;
		uchar* pM; // pointer to mask image values
		for (int r = 0; r < cameras_[cid]->dm_->depth_map_.rows(); r++) {
			pM = cameras_[cid]->imgMask_.ptr<uchar>(r);
			for (int c = 0; c < cameras_[cid]->dm_->depth_map_.cols(); c++) {
				dval = cameras_[cid]->dm_->depth_map_(r, c);
				if (dval == 0.) continue; // no depth information
				if (pM[c] == 0) continue; // masked out

				// retrieve position and normalize homogeneous coordinates
				int idx = PixIndexFwdRM(Point(c, r), cameras_[cid]->width_);

				if (first) {
					min_depth = dval;
					max_depth = dval;
				}
				else {
					if (dval < min_depth) min_depth = dval;
					if (dval > max_depth) max_depth = dval;
				}

				if ((dval < established_min_depth) ||
					(dval > established_max_depth)) {
					cout << "Camera " << cid << " has min/max outside of bounds." << endl;
					cout << "Camera space min, max already established: " << established_min_depth << ", " << established_max_depth << endl;
					cout << "Current depth is: " << dval << endl;
					cout << "Current SS position (r, c) is: (" << r << ", " << c << ")" << endl;
					int idx = r*cameras_[cid]->width_ + c;
					cout << "Index for that SS position is: " << idx << endl;
					cout << "Iws_ coordinate at that position is " << endl << cameras_[cid]->Iws_.col(idx) << endl;
					Matrix<float, 4, 1> CSpos = cameras_[cid]->RT_.cast<float>() * cameras_[cid]->Iws_.col(idx);
					float hmin = CSpos(3, 0);
					CSpos = CSpos.array() / hmin;
					cout << "CS coordinate at that position is " << endl << CSpos << endl;
					cin.ignore();
				}

				first = false;
			}
		}
	
		if ((min_depth < established_min_depth) ||
			(max_depth > established_max_depth)) {
			cout << "Camera " << cid << " has min/max outside of bounds." << endl;
			cout << "Established min, max: " << established_min_depth << ", " << established_max_depth << endl;
			cout << "Just identified min, max: " << min_depth << ", " << max_depth << endl;
			cin.ignore();
		}
		else if (debug) {
			cout << "Camera " << cid << endl;
			cout << "Established min, max: " << established_min_depth << ", " << established_max_depth << endl;
			cout << "Just identified min, max: " << min_depth << ", " << max_depth << endl;
			cin.ignore();
		}
	}

	cout << "completed test" << endl;
}

// update Iws_ for all cameras based on current depth data
void Scene::UpdateCamerasWSPoints() {
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;
		if ((!(*it).second->has_depth_map_) ||
			(!(*it).second->enabled_) ||
			(!(*it).second->posed_)) continue;

		(*it).second->InitWorldSpaceProjection();
	}
}

// computes min and max depths for all cameras' points projected into each camera's CS as a reference camera in turn
// updates min_depths_ and max_depths_
void Scene::UpdateCSMinMaxDepths() {
	bool debug = false;
	bool load_from_file = false;

	cout << "Scene::UpdateCSMinMaxDepths()" << endl;

	if ((debug) &&
		(load_from_file)) {
		std::string mat_name = name_ + "_CSMinMaxDepths";
		int max_cid = 0;
		for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
			int cid = (*it).first;
			if (cid > max_cid) max_cid = cid;
		}
		Matrix<float, Dynamic, 2> mmds(max_cid+1, 2);
		LoadEigenMatrix(mat_name, TYPE_FLOAT, mmds);

		for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
			int cid = (*it).first;
			if ((!(*it).second->has_depth_map_) ||
				(!(*it).second->enabled_) ||
				(!(*it).second->posed_)) continue;
			min_depths_[cid] = mmds(cid, 0);
			max_depths_[cid] = mmds(cid, 1);
		}
		return;
	}

	/*
	// set up point qualification matrix for each camera
	std::map<int, Matrix<bool, 1, Dynamic>> qual_pts_by_cam;
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;
		if ((!(*it).second->has_depth_map_) ||
			(!(*it).second->enabled_) ||
			(!(*it).second->posed_)) continue;
		
		uchar *pM;
		int height = (*it).second->imgMask_.rows;
		int width = (*it).second->imgMask_.cols;
		Matrix<bool, 1, Dynamic> q(1, height*width);
		q.setZero();
		for (int r = 0; r < height; r++) {
			pM = (*it).second->imgMask_.ptr<uchar>(r);
			for (int c = 0; c < width; c++) {
				if ((pM[c] == 0) ||
					((*it).second->dm_->depth_map_(r, c) == 0.)) continue;
				int idx = PixIndexFwdRM(Point(c, r), width);
				q(0, idx) = true;
			}
		}
		qual_pts_by_cam[cid] = q;
	}
	*/

	UpdateCamerasWSPoints(); // ensure Iws_ points are up to date for the camera before using them

	// treating each camera in turn as the reference camera
	for (std::map<int, Camera*>::iterator it_ref = cameras_.begin(); it_ref != cameras_.end(); ++it_ref) {
		int cid_ref = (*it_ref).first;
		if ((!(*it_ref).second->has_depth_map_) ||
			(!(*it_ref).second->enabled_) ||
			(!(*it_ref).second->posed_)) continue;

		// determine minimum and maximum depths in the camera space of the reference camera
		// project WS points from each camera into the reference camera's CS, then find min and max depth; doing it this way ensures we get the tightest bounds on depth in CS
		float min_depth = 0.;
		float max_depth = 0.;
		float curr_depth;
		bool first_cam_for_depths = true;
		for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
			int cid = (*it).first;
			if ((cid == cid_ref) ||
				(!(*it).second->has_depth_map_) ||
				(!(*it).second->enabled_) ||
				(!(*it).second->posed_)) continue;

			Matrix<float, 4, Dynamic> Ics = cameras_[cid_ref]->RT_.cast<float>() * (*it).second->Iws_; // project this camera's WS points into the reference camera's CS
			Ics.row(2) = Ics.row(2).cwiseQuotient(Ics.row(3));

			if (debug) {
				Matrix<float, 4, Dynamic> Iws = (*it).second->Iws_;
				Matrix4f RT = cameras_[cid_ref]->RT_.cast<float>();
				DebugPrintMatrix(&Iws, "Iws");
				DebugPrintMatrix(&RT, "RT");
				DebugPrintMatrix(&Ics, "Ics");
			}

			/*
			int height = (*it).second->imgMask_.rows;
			int width = (*it).second->imgMask_.cols;
			float all_max = Ics.row(2).maxCoeff();
			Matrix<float, 1, Dynamic> max_vals_for_disquals(1, height*width);
			max_vals_for_disquals.setConstant(all_max);
			max_vals_for_disquals = max_vals_for_disquals.cwiseProduct(qual_pts_by_cam[cid].cast<float>());
			Matrix<float, 1, Dynamic> testmat(1, height*width);
			testmat = Ics.row(2) + max_vals_for_disquals;
			float curr_min_depth = testmat.minCoeff();
			testmat = Ics.row(2) - max_vals_for_disquals;
			float curr_max_depth = testmat.maxCoeff();
			max_vals_for_disquals.resize(1, 0);
			testmat.resize(1, 0);
			if (curr_min_depth < min_depth) min_depth = curr_min_depth;
			if (curr_max_depth > max_depth) max_depth = curr_max_depth;
			*/
			
			uchar *pM;
			int height = (*it).second->imgMask_.rows;
			int width = (*it).second->imgMask_.cols;
			for (int r = 0; r < height; r++) {
				pM = (*it).second->imgMask_.ptr<uchar>(r);
				for (int c = 0; c < width; c++) {
					if ((pM[c] == 0) ||
						((*it).second->dm_->depth_map_(r, c) == 0.)) continue;
					int idx = PixIndexFwdRM(Point(c, r), width);
					curr_depth = Ics(2, idx);
					if (curr_depth <= 0.) continue; // exclude behind the camera
					if (first_cam_for_depths) {
						min_depth = curr_depth;
						max_depth = curr_depth;
						first_cam_for_depths = false;
					}
					else {
						if (curr_depth < min_depth) min_depth = curr_depth;
						if (curr_depth > max_depth) max_depth = curr_depth;
					}
				}
			}
			
		}

		if (debug) cout << "camera " << cid_ref << " has min_depth " << min_depth << " and max_depth " << max_depth << endl;

		min_depths_[cid_ref] = min_depth;
		max_depths_[cid_ref] = max_depth;
	}

	if (debug) {
		std::string mat_name = name_ + "_CSMinMaxDepths";
		int max_cid = 0;
		for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
			int cid = (*it).first;
			if (cid > max_cid) max_cid = cid;
		}
		Matrix<float, Dynamic, 2> mmds(max_cid+1, 2);
		for (std::map<int, float>::iterator it = min_depths_.begin(); it != min_depths_.end(); ++it) {
			int cid = (*it).first;
			float d = (*it).second;
			mmds(cid, 0) = d;
		}
		for (std::map<int, float>::iterator it = max_depths_.begin(); it != max_depths_.end(); ++it) {
			int cid = (*it).first;
			float d = (*it).second;
			mmds(cid, 1) = d;
		}
		SaveEigenMatrix(mat_name, TYPE_FLOAT, mmds);
	}
}

// note that this function uses the same WS point computation approach as Camera::InverseProjectSStoWS() it performs a similar function to StereoData::UpdateDisps(), which also computes disparity bounds and quantization, but the latter uses ojw's approach while this uses ours.  Also, this computes disparity information for every camera in turn, using all other cameras, while StereoData::UpdateDisps() computes only a single identified reference camera's disparity information and only uses cameras qualified as being used in its use_cids_ data structure.  The reason for this separate function here is to compute bounds and quantizations to use when saving and loading rendering assets (quantizing disparities minimizes their file space footprint compared to recording floating point values), while StereoData::UpdateDisps() is used when performing stereo reconstruction for a particular reference camera using a particular set of use_cid_ cameras.
void Scene::UpdateDisparities() {
	bool debug = false;

	UpdateCSMinMaxDepths();

	// treating each camera in turn as the reference camera
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid_ref = (*it).first;

		// extend range by GLOBAL_EXTEND_DEPTH_RANGE front and back
		float ext = (max_depths_[cid_ref] - min_depths_[cid_ref]) * GLOBAL_EXTEND_DEPTH_RANGE;
		float min_depth_ext = min_depths_[cid_ref] - ext; // this is in camera space for cid_out
		if (min_depth_ext <= 0.) min_depth_ext = GLOBAL_MIN_CS_DEPTH;
		float max_depth_ext = max_depths_[cid_ref] + ext; // this is in camera space for cid_out

		Matrix<float, 3, 2> depth_extremes_SSout; // 2 world space points (0,0,min_depth,1) and (0,0,max_depth,1)
		depth_extremes_SSout.setZero();
		int xpos = 0;
		int ypos = 0;
		depth_extremes_SSout(0, 0) = xpos * min_depth_ext;
		depth_extremes_SSout(0, 1) = xpos * max_depth_ext;
		depth_extremes_SSout(1, 0) = ypos * min_depth_ext;
		depth_extremes_SSout(1, 1) = ypos * max_depth_ext;
		depth_extremes_SSout(2, 0) = min_depth_ext;
		depth_extremes_SSout(2, 1) = max_depth_ext;

		// transform screen space to camera space - transform u,v to x,y, then add rows for z (equal to each depth value) and w (equal to 1.0)
		Matrix<float, 2, 3> Kinv_uvonly;
		Kinv_uvonly.row(0) = cameras_[cid_ref]->calib_.Kinv_.row(0).cast<float>();
		Kinv_uvonly.row(1) = cameras_[cid_ref]->calib_.Kinv_.row(1).cast<float>();
		Matrix<float, 2, Dynamic> Ics_xyonly = Kinv_uvonly * depth_extremes_SSout; // Ics is homogeneous 4xn matrix of camera space point
		Matrix<float, 4, 2> depth_extremes_CSout;
		depth_extremes_CSout.setOnes(); // Iws at this point is still in camera space until we multiply it below by inverse extrinsics
		depth_extremes_CSout.row(0) = Ics_xyonly.row(0);
		depth_extremes_CSout.row(1) = Ics_xyonly.row(1);
		depth_extremes_CSout(2, 0) = min_depth_ext;
		depth_extremes_CSout(2, 1) = max_depth_ext;

		Matrix<float, 4, 2> depth_extremes_WSout;
		depth_extremes_WSout = cameras_[cid_ref]->RTinv_.cast<float>() * depth_extremes_CSout;

		float disp_vals = 0; // to hold the number of disparity levels; to be set by projecting points (0,0,min_depth,1) and (0,0,max_depth,1) into each reference camera, finding the pixel distance between projected near and far points, and ensuring the minimum spacing between disparity samples is 0.5 pixels in screen space

		for (std::map<int, Camera*>::iterator it2 = cameras_.begin(); it2 != cameras_.end(); ++it2) {
			int cid = (*it2).first;
			if (cid == cid_ref) continue;

			Matrix<float, 3, 2> depth_extremes_SS = (*it2).second->P_.cast<float>() * depth_extremes_WSout; // project world space points (0,0,min_depth,1) and (0,0,max_depth,1) into screen space of this camera
			// divide the screen space points through by the homogeneous coordinates to normalize them
			float h0 = depth_extremes_SS(2, 0);
			depth_extremes_SS.col(0) = depth_extremes_SS.col(0) / h0;
			float h1 = depth_extremes_SS(2, 1);
			depth_extremes_SS.col(1) = depth_extremes_SS.col(1) / h1;
			// find the maximum pixel distance among x and y dimensions for the two projected pixels
			float xdist = abs(depth_extremes_SS(0, 0) - depth_extremes_SS(0, 1));
			float ydist = abs(depth_extremes_SS(1, 0) - depth_extremes_SS(1, 1));
			disp_vals = max(disp_vals, max(xdist, ydist));
		}

		num_disps_[cid_ref] = ceil(disp_vals * 2.); // ensure minimum spacing is 0.5 pixels, so multiply minimum 1 pixel spacing by 2 to get minimum 0.5 pixel spacing and round up to ensure integer number of disparity levels is over the threshold

		// Calculate disparities - disps ends up as range from 1/min_depth to 1/max_depth with disp_vals number of steps, evenly spaced
		disps_[cid_ref] = ArrayXd(num_disps_[cid_ref]); // set up disparities from 0 through (num_disps - 1)
		for (int i = 0; i < num_disps_[cid_ref]; i++) {
			disps_[cid_ref](i) = (float)i;
		}
		// taking num_disps_ to be 10, min depth to be 2m and max to be 8m, take it through the calcs...
		disps_[cid_ref] *= (1. - (min_depth_ext / max_depth_ext)) / (float)(num_disps_[cid_ref] - 1); // disps: 0, 3/36, 6/36, 9/36, ... , 27/36
		disps_[cid_ref] = (1. - disps_[cid_ref]) / min_depth_ext; // disps: 1/2, 5/12, 3/8, ... , 1/8

		// order disparities from foreground to background => descending
		std::sort(disps_[cid_ref].data(), disps_[cid_ref].data() + disps_[cid_ref].size(), std::greater<float>()); // sorts values in descending order, but are already in that order at this point...uncomment if want to make doubly-sure
		//igl::sort(X, dim, mode, Y, IX); // igl method for sorting values in descending order

		max_disps_[cid_ref] = disps_[cid_ref](0);
		min_disps_[cid_ref] = disps_[cid_ref](num_disps_[cid_ref] - 1);
		disp_steps_[cid_ref] = disps_[cid_ref](0) - disps_[cid_ref](1); // max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp

		if (debug) {
			cout << endl << endl << "Scene::UpdateDisparities() calculations for reference camera with ID " << cid_ref << endl;
			cout << "Min depth: " << min_depths_[cid_ref] << endl;
			cout << "Max depth: " << max_depths_[cid_ref] << endl;
			cout << "Min depth extended: " << min_depth_ext << endl;
			cout << "Max depth extended: " << max_depth_ext << endl;
			cout << "Max disparity (closest): " << max_disps_[cid_ref] << endl;
			cout << "Min disparity (farthest): " << min_disps_[cid_ref] << endl;
			cout << "Number of disparities: " << num_disps_[cid_ref] << endl;
			cout << "Disparity step: " << disp_steps_[cid_ref] << endl;

			cin.ignore();
		}
	}
}

// note: all matrices are indexed in row-major pixel order (across a row, then down to the next one)
// saves compressed data needed for rendering as a binary file with the following data, in order:
// 1. 50 character (byte) asset name that is binary-zero-terminated (char \0)
// 2. short for # rows, short for # cols
// 3. bounding volume as floats of in order of: xmin, ymin, zmin, xmax, ymax, zmax
// 4. 4x3 inverse projection matrix Pinv of 12 floats
// 5. int number of used pixels, int number of packing data integers
// 6. packing data; start with a short containing the number of contiguous used pixels' depths in the current raster row to follow (0 if starts with empty pixels), followed by the number of contiguous unused pixels next in the current raster row, followed by the number of contiguous used pixels' depths in the current raster row to follow, etc. repeated to the end of the raster row.  This is repeated for each raster row in a contiguous block of data.
// 7. sparse pixel depths, each as a half-float; they are in raster-scan order, skipping any unused pixels
// 8. sparse BGR color values, each as a triple of unsigned characters; they are in raster-scan order, skipping any unused pixels
void Scene::SaveRenderingDataRLE() {
	UpdateDisparities(); // ensure disparity data is up-to-date; must save disparity information in asset bundle for export since are saving quantized disparities that must later be converted to depth values

	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;
		if ((!(*it).second->posed_) ||
			(!(*it).second->has_depth_map_)) continue;
		(*it).second->SaveRenderingDataRLE(name_, min_disps_[cid], max_disps_[cid], disp_steps_[cid]);
	}
}

// note: all matrices are indexed in row-major pixel order (across a row, then down to the next one)
// loads compressed data needed for rendering as a binary file with the following data, in order:
// 1. 50 character (byte) asset name that is binary-zero-terminated (char \0)
// 2. short for # rows, short for # cols
// 3. bounding volume as floats of in order of: xmin, ymin, zmin, xmax, ymax, zmax
// 4. 4x3 inverse projection matrix Pinv of 12 floats
// 5. int number of used pixels, int number of packing data integers
// 6. packing data; start with a short containing the number of contiguous used pixels' depths in the current raster row to follow (0 if starts with empty pixels), followed by the number of contiguous unused pixels next in the current raster row, followed by the number of contiguous used pixels' depths in the current raster row to follow, etc. repeated to the end of the raster row.  This is repeated for each raster row in a contiguous block of data.
// 7. sparse pixel depths, each as a half-float; they are in raster-scan order, skipping any unused pixels
// 8. sparse BGR color values, each as a triple of unsigned characters; they are in raster-scan order, skipping any unused pixels
void Scene::LoadRenderingDataRLE() {
	bool debug = false;

	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;
		if ((!(*it).second->posed_) ||
			(!(*it).second->has_depth_map_)) continue;
		float min_disp, max_disp, disp_step;
		(*it).second->LoadRenderingDataRLE(name_, min_disp, max_disp, disp_step);
		min_disps_[cid] = min_disp;
		max_disps_[cid] = max_disp;
		disp_steps_[cid] = disp_step;
	}
}

// tests world space coordinates from each camera against masks from all other input cameras.  any world space point that, when reprojected into another camera space, does not fall on a pixel with a true value in that camera space is considered to have a rejected depth value.  the corresponding pixel in the corresponding depth map is assigned a depth of 0 to signify that we have no valid information for that pixel
// masking is only effective if a projected point is within a mask's screen space.  Since an camera may frame a scene such that the object of interest is not fully contained, we can't simply call projected points outside of the screen space "masked-out."  Yet we must be able to eliminate gross depth errors that result in projected points being outside many screen spaces and, therefore, valid for being considered "masked-in."  To address this issue, we make an assumption ("x" decimal percentage) regarding the maximum amount of an object of interest that may fall outside of any camera's screen space, assuming also that the object of interest is connected and roughly evenly distributed.  Then, if a projected point is (height / x) pixels above or below the screen space or (width / x) pixels to the left or right of the screen space, the current depth for the pixel in its originating camera is considered invalid.  In Globals.h, we define the related value, GLOBAL_FRAMING_MIN_OBJ_PERC, the minimum decimal percentage of an object of interest that is assumed to appear within frame for any camera.  As we go through projected pixels, zero out the depth values in the generating camera for pixels that fail this test.
void Scene::CleanDepths() {
	bool debug = true;
	bool debug_display = false;
	
	for (std::map<int, Camera*>::iterator it1 = cameras_.begin(); it1 != cameras_.end(); ++it1) {
		int cid1 = (*it1).first;
		if ((!(*it1).second->posed_) ||
			(!(*it1).second->enabled_) ||
			(!(*it1).second->has_depth_map_)) continue;

		if (debug) cout << "Scene::CleanDepths() for image " << cid1 << endl;
		
		// collect coordinates of known, masked-in pixels
		Matrix<double, Dynamic, 4> WC_tmp(cameras_[cid1]->height_*cameras_[cid1]->width_, 4);
		WC_tmp.col(2).setOnes();
		double *pX = WC_tmp.col(0).data();
		double *pY = WC_tmp.col(1).data();
		double *pDisp = WC_tmp.col(3).data();
		Matrix<bool, Dynamic, 1> known_mask(cameras_[cid1]->height_*cameras_[cid1]->width_, 1);
		bool *pM = known_mask.data();
		float *pDM = cameras_[cid1]->dm_->depth_map_.data();
		int num_known = 0;
		float z;
		for (int c = 0; c < cameras_[cid1]->imgT_.cols; c++) {
			for (int r = 0; r < cameras_[cid1]->imgT_.rows; r++) {
				z = *pDM++;
				if ((z != 0) &&
					(cameras_[cid1]->imgMask_.at<uchar>(r, c) > 0)) {
					*pX++ = c;
					*pY++ = r;
					*pDisp++ = 1. / static_cast<double>(z);
					num_known++;
					*pM++ = true;
				}
				else *pM++ = false;
			}
		}
		Matrix<double, Dynamic, 4> WC(num_known, 4);
		WC.block(0, 0, num_known, 4) = WC_tmp.block(0, 0, num_known, 4);

		for (std::map<int, Camera*>::iterator it2 = cameras_.begin(); it2 != cameras_.end(); ++it2) {
			int cid2 = (*it2).first;
			if ((!(*it2).second->posed_) || // the second camera doesn't need to have a depth map, but does need to be posed, enabled, and have a mask
				(!(*it2).second->enabled_) ||
				(!(*it1).second->has_depth_map_) ||
				(cid1 == cid2)) continue;

			CleanDepths_Pair(cid1, &WC, &known_mask, cid2);
		}
		
		if (debug_display) { // display the source camera's texture image with a change: any masked-in pixel with a 0 depth value is highlighted in red
			Mat img = cv::Mat::zeros(cameras_[cid1]->height_, cameras_[cid1]->width_, CV_8UC3);
			cameras_[cid1]->imgT_.copyTo(img);
			
			Vec3b *pT;
			int num_unknown = 0;
			for (int r = 0; r < img.rows; r++) {
				pT = img.ptr<Vec3b>(r);
				for (int c = 0; c < img.cols; c++) {
					//idx = PixIndexFwdCM(Point(c, r), img.rows);
					if ((cameras_[cid1]->dm_->depth_map_(r, c) == 0) &&
						(cameras_[cid1]->imgMask_.at<uchar>(r, c) > 0)) {
						pT[c] = Vec3b(0, 0, 255);
						num_unknown++;
					}
				}
			}

			cout << "num_unknown " << num_unknown << endl;
			display_mat(&img, "CleanDepths_Pair masked-in 0 depth in red", cameras_[cid1]->orientation_);
		}
	}
}

// given source and destination camera IDs, updates the source camera's depth map so that any of the source camera's world space points, when reprojected into the destination camera's screen space, that falls on a masked out pixel has its corresponding depth value in the source camera's depth map set to zero to signify an unknown depth (since the depth we had for that pixel was found to be incorrect)
// WC_known is data structure with SS coordinates of known pixels with column 0 containing X coords, col 1 containing Y coord, col 2 containing constant 1., and col 3 containing disparity (not depth) value in camera space
// mask known has a row for every pixel in the image, with true values where the pixel is "known"
// masking is only effective if a projected point is within a mask's screen space.  Since an camera may frame a scene such that the object of interest is not fully contained, we can't simply call projected points outside of the screen space "masked-out."  Yet we must be able to eliminate gross depth errors that result in projected points being outside many screen spaces and, therefore, valid for being considered "masked-in."  To address this issue, we make an assumption ("x" decimal percentage) regarding the maximum amount of an object of interest that may fall outside of any camera's screen space, assuming also that the object of interest is connected and roughly evenly distributed.  Then, if a projected point is (height / x) pixels above or below the screen space or (width / x) pixels to the left or right of the screen space, the current depth for the pixel in its originating camera is considered invalid.  In Globals.h, we define the related value, GLOBAL_FRAMING_MIN_OBJ_PERC, the minimum decimal percentage of an object of interest that is assumed to appear within frame for any camera.  As we go through projected pixels, zero out the depth values in the generating camera for pixels that fail this test.
void Scene::CleanDepths_Pair(int cid_src, Matrix<double, Dynamic, 4> *WC_known, Matrix<bool, Dynamic, 1> *known_mask, int cid_dest) {
	assert(cameras_[cid_src]->posed_ && cameras_[cid_src]->has_depth_map_ && cameras_[cid_dest]->posed_ && (cid_src != cid_dest));
	
	bool debug = false;
	bool debug_tmp = false;
	bool debug_tmp2 = false;

	if (debug) cout << "Scene::CleanDepths_Pair() for image " << cid_src << " against image " << cid_dest << endl;
	
	// reproject known screen space coordinates of cid_src to screen space of cid_dest and normalize by homogeneous coordinates
	Matrix<float, 1, 4> ext;
	ext << 0., 0., 0., 1.;
	Matrix4f Psrc_ext;
	Psrc_ext << cameras_[cid_src]->P_.cast<float>(), ext;
	Matrix4f Psrc_ext_inv = Psrc_ext.inverse();
	Matrix<float, 3, 4> Psrc2dest = cameras_[cid_dest]->P_.cast<float>() * Psrc_ext_inv;
	Matrix<double, Dynamic, 3> T = (*WC_known) * Psrc2dest.transpose().cast<double>();
	Matrix<double, Dynamic, 1> H = T.col(2).array().inverse(); // determine homogeneous coordinates to divide by
	T.col(0) = T.col(0).cwiseProduct(H); // divide by homogeneous coordinates
	T.col(1) = T.col(1).cwiseProduct(H); // divide by homogeneous coordinates
	
	int h_src = cameras_[cid_src]->height_;
	int w_src = cameras_[cid_src]->width_;
	int h_dest = cameras_[cid_dest]->height_;
	int w_dest = cameras_[cid_dest]->width_;

	// interpolate depths from floating point coordinates in destination screen space
	Matrix<double, Dynamic, 1> X = T.col(0);
	Matrix<double, Dynamic, 1> Y = T.col(1);
	Matrix<bool, Dynamic, 1> inbound_mask(T.rows(), 1);
	inbound_mask.setConstant(true);
	Interpolation::InterpolateAgainstMask<double>(w_dest, h_dest, &X, &Y, &masks_dilated_[cid_dest], &inbound_mask, cameras_[cid_dest]->closeup_xmin_, cameras_[cid_dest]->closeup_xmax_, cameras_[cid_dest]->closeup_ymin_, cameras_[cid_dest]->closeup_ymax_);

	if (debug_tmp) {
		cout << "Scene::CleanDepths_Pair() before for src " << cid_src << " with non-zero depth count of " << cameras_[cid_src]->dm_->depth_map_.count() << " and dest " << cid_dest << endl;
		DisplayImages::DisplayGrayscaleImage(&cameras_[cid_src]->dm_->depth_map_, h_src, w_src, cameras_[cid_src]->orientation_);
		cout << "Scene::CleanDepths_Pair() masks_dilated_[ " << cid_dest << "]" << endl;
		DisplayImages::DisplayGrayscaleImage(&masks_dilated_[cid_dest], h_dest, w_dest, cameras_[cid_dest]->orientation_);
		cout << "Scene::CleanDepths_Pair() inbound mask with count " << inbound_mask.count() << endl;
		DisplayImages::DisplayGrayscaleImageTruncated(&inbound_mask, known_mask, h_src, w_src, cameras_[cid_src]->orientation_);
	}

	if (debug_tmp2) {
		if (inbound_mask.count() != inbound_mask.rows())
			cout << "Scene::CleanDepths_Pair() for cid_src " << cid_src << ", cid_dest " << cid_dest << " inbound_mask has " << inbound_mask.count() << " of " << inbound_mask.rows() << endl;
		else cout << "Scene::CleanDepths_Pair() for cid_src " << cid_src << ", cid_dest " << cid_dest << " inbound_mask for all pixels checked" << endl;
	}

	Matrix<bool, Dynamic, 1> inbound_mask_full(h_src*w_src, 1);
	inbound_mask_full.setZero();
	EigenMatlab::AssignByTruncatedBooleans(&inbound_mask_full, known_mask, &inbound_mask);
	float val_zero = 0.;
	EigenMatlab::AssignByBooleansNot(&cameras_[cid_src]->dm_->depth_map_, &inbound_mask_full, val_zero);
	
	if (debug_tmp) {
		cout << "inbound_mask rows " << inbound_mask.rows() << ", count " << inbound_mask.count() << endl;
		cout << "inbound_mask_full rows " << inbound_mask_full.rows() << ", count " << inbound_mask_full.count() << endl;
		cout << "Scene::CleanDepths_Pair() after for src " << cid_src << " with non-zero depth count of " << cameras_[cid_src]->dm_->depth_map_.count() << endl;
		DisplayImages::DisplayGrayscaleImage(&cameras_[cid_src]->dm_->depth_map_, h_src, w_src, cameras_[cid_src]->orientation_);
	}
	
	if (debug) { // display the source camera's texture image with a change: any masked-in pixel with a 0 depth value is highlighted in red
		Mat img = cv::Mat::zeros(cameras_[cid_src]->height_, cameras_[cid_src]->width_, CV_8UC3);
		cameras_[cid_src]->imgT_.copyTo(img);

		int idx;
		Vec3b *pT;
		int num_unknown = 0;
		for (int r = 0; r < img.rows; r++) {
			pT = img.ptr<Vec3b>(r);
			for (int c = 0; c < img.cols; c++) {
				idx = PixIndexFwdCM(Point(c, r), img.rows);
				if ((cameras_[cid_src]->dm_->depth_map_(idx, 0) == 0) &&
					(cameras_[cid_src]->imgMask_.at<uchar>(r, c) > 0)) {
					pT[c] = Vec3b(0, 0, 255);
					num_unknown++;
				}
			}
		}

		cout << "num_unknown " << num_unknown << endl;
		display_mat(&img, "CleanDepths_Pair masked-in 0 depth in red", cameras_[cid_src]->orientation_);
	}
}

// camera poses may not be correct, so test input camera poses against reference camera: projections of reference camera pixels into each input camera's screen space should land inside masked pixels within an acceptable error factor, represented by applying a morphological dilation to the input camera mask with element size given by GLOBAL_MASK_DILATION
// uses const float GLOBAL_RATIO_PASS_DILATED_MASKS = 0.75; // if this decimal percentage of reprojected unknown pixels from the reference camera fall within masked-in pixels of a dilated mask of an input camera in that camera's screen space, the input camera is considered to have passed the pose accuracy test
// any failing cameras and their data are given a false flag for posed_
// if a majority of the input cameras fail the test, the reference camera is considered to have a bad pose (majority rules for pose estimation)...otherwise, the cameras with false in valid_cam_poses are assumed to have bad poses; if exactly half pass, then assume only the group of cameras that include the lowest cid pass so we don't have two groups that do not match up well when rendering
// note: this function will not display proper results for cameras of different screen sizes than the first, though that is handled appropriately elsewhere in the 
void Scene::FilterCamerasByPoseAccuracy() {
	bool debug = true;
	bool debug_display = false;

	if (debug) cout << "Scene::FilterCamerasByPoseAccuracy()" << endl;
	
	for (std::map<int, Camera*>::iterator it1 = cameras_.begin(); it1 != cameras_.end(); ++it1) {
		int cid_out = (*it1).first;
		if ((!(*it1).second->posed_) ||
			(!(*it1).second->enabled_) ||
			(!(*it1).second->has_depth_map_)) continue;

		int height = (*it1).second->imgT_.rows;
		int width = (*it1).second->imgT_.cols;

		Matrix<bool, Dynamic, Dynamic> kd = (*it1).second->dm_->depth_map_.array() != 0.;
		kd.resize(kd.rows()*kd.cols(), 1);
		Matrix<bool, Dynamic, 1> known_depths = kd;

		bool *pKD = known_depths.data();
		int num_known = known_depths.count();
		Matrix<float, Dynamic, 4> WCkd(num_known, 4);
		float *pWCkd_x = WCkd.data();
		float *pWCkd_y = WCkd.data() + num_known;
		float *pWCkd_z = WCkd.data() + 3 * num_known;
		for (int i = 0; i < height*width; i++) {
			if (!*pKD++) continue;
			Point pt = PixIndexBwdCM(i, height);
			*pWCkd_x++ = pt.x;
			*pWCkd_y++ = pt.y;
			*pWCkd_z++ = 1 / cameras_[cid_out]->dm_->depth_map_(pt.y, pt.x);
		}
		WCkd.col(2).setOnes();

		Matrix<double, 1, 4> ext;
		ext << 0., 0., 0., 1.;
		Matrix4d Pout_ext;
		Pout_ext << cameras_[cid_out]->P_, ext;
		Matrix4d Pout_ext_inv = Pout_ext.inverse();

		map<int, bool> valid_cam_poses;
		
		for (std::map<int, Camera*>::iterator it2 = cameras_.begin(); it2 != cameras_.end(); ++it2) {
			int cid = (*it2).first;
			if ((!(*it2).second->posed_) || // the second camera doesn't need to have a depth map, but does need to be posed, enabled, and have a mask
				(!(*it2).second->enabled_) ||
				(cid == cid_out)) continue;

			Matrix<double, 3, 4> Pout2in = cameras_[cid]->P_ * Pout_ext_inv;
			
			Matrix<double, Dynamic, 3> T2 = WCkd.cast<double>() * Pout2in.transpose();
			Matrix<double, Dynamic, 1> N2 = T2.col(2).array().inverse();
			T2.col(0) = T2.col(0).cwiseProduct(N2);
			T2.col(1) = T2.col(1).cwiseProduct(N2);

			Matrix<int, Dynamic, 1> X_round2 = T2.col(0).cast<int>();
			Matrix<int, Dynamic, 1> Y_round2 = T2.col(1).cast<int>();
			int *pX = X_round2.data();
			int *pY = Y_round2.data();
			int x, y, k;
			int h = cameras_[cid]->imgT_.rows;
			int w = cameras_[cid]->imgT_.cols;
			int num_pixels_fail = 0;
			for (int i = 0; i < T2.rows(); i++) {
				x = *pX++;
				y = *pY++;
				if ((x >= 0) &&
					(y >= 0) &&
					(x < w) &&
					(y < h)) {
					k = h * x + y; // col major index of current position
					if (!masks_dilated_[cid](k, 0))
						num_pixels_fail++;
				}
			}

			if (debug_display) {
				cout << "Scene::FilterCamerasByPoseAccuracy() cid " << cid << endl;

				cout << "Number of points projected: " << T2.rows() << endl;
				Mat img = cameras_[cid]->MaskedImgT();

				if (GLOBAL_MASK_DILATION > 0) {
					int morph_type = MORPH_RECT; // MORPH_ELLIPSE
					int morph_size = GLOBAL_MASK_DILATION; // get rid of small regional markings
					Mat element = getStructuringElement(morph_type,
						Size(2 * morph_size + 1, 2 * morph_size + 1),
						Point(morph_size, morph_size));
					dilate(img, img, element);
				}

				pX = X_round2.data();
				pY = Y_round2.data();
				for (int i = 0; i < T2.rows(); i++) {
					x = *pX++;
					y = *pY++;
					if ((x >= 0) &&
						(y >= 0) &&
						(x < w) &&
						(y < h)) {
						Vec3b pix = img.at<Vec3b>(y, x);
						pix[2] = 255;
						img.at<Vec3b>(y, x) = pix;
					}
				}
				display_mat(&img, "test", cameras_[cid]->orientation_);
			}

			// test whether this camera passes the hurdle
			float ratio_fail = (float)num_pixels_fail / (float)num_known;
			if (ratio_fail >= (1 - GLOBAL_RATIO_PASS_DILATED_MASKS))
				valid_cam_poses[cid] = false;
			else valid_cam_poses[cid] = true;
		}
		
		// if a majority of the input cameras fail the test, the reference camera is considered to have a bad pose (majority rules for pose estimation)...otherwise, the cameras with false in valid_cam_poses are assumed to have bad poses; if exactly half pass, then assume only the group of cameras that include the lowest cid pass so we don't have two groups that do not match up well when rendering
		int num_cams = valid_cam_poses.size();
		int num_cams_valid = 0;
		int first_cid = -1;
		for (std::map<int, bool>::iterator it = valid_cam_poses.begin(); it != valid_cam_poses.end(); ++it) {
			if ((*it).first == cid_out) continue;
			if (first_cid == -1) first_cid = (*it).first;
			if ((*it).second) num_cams_valid++;
		}
		float ratio = (float)num_cams_valid / (float)num_cams;
		bool ref_passes;
		if (ratio > 0.5) ref_passes = true;
		else if (ratio < 0.5) ref_passes = false;
		else ref_passes = valid_cam_poses[first_cid]; // ratio == 0.5

		if (ref_passes) { // current reference camera is good, so prune others by accuracy
			for (map<int, bool>::iterator it = valid_cam_poses.begin(); it != valid_cam_poses.end(); ++it) {
				int cid = (*it).first;
				bool pass = (*it).second;
				if (!pass) {
					cameras_[cid]->posed_ = false;
					if (debug) cout << "Scene::FilterCamerasByPoseAccuracy() cid " << cid << " fails projection of known pixels of reference camera against its dilated mask - pose assumed inaccurate" << endl;
				}
			}
		}
		else { // current reference camera is inaccurate
			if (debug) cout << "Scene::FilterCamerasByPoseAccuracy() cid " << cid_out << " fails projection as reference camera against other dilated masks - pose assumed inaccurate" << endl;
			cameras_[cid_out]->posed_ = false;
		}
	}
}

// generates all segmentation label images
void Scene::GenerateUnknownSegmentations() {
	for (std::map<int, Camera*>::iterator it1 = cameras_.begin(); it1 != cameras_.end(); ++it1) {
		int cid1 = (*it1).first;
		if ((!(*it1).second->posed_) ||
			(!(*it1).second->enabled_) ||
			(!(*it1).second->has_depth_map_)) continue;

		Mat img = cv::Mat::zeros(cameras_[cid1]->height_, cameras_[cid1]->width_, CV_8UC1);

		uchar *pT;
		for (int r = 0; r < img.rows; r++) {
			pT = img.ptr<uchar>(r);
			for (int c = 0; c < img.cols; c++) {
				//idx = PixIndexFwdCM(Point(c, r), img.rows);
				if ((cameras_[cid1]->dm_->depth_map_(r, c) == 0) &&
					(cameras_[cid1]->imgMask_.at<uchar>(r, c) > 0)) {
					pT[c] = 255;
				}
			}
		}
		GenerateUnknownSegmentations_img(cid1, &img);
	}
}

// img is of type CV_8UC1 with thresholded 0, 255 values only
void Scene::GenerateUnknownSegmentations_img(int cid, Mat *img)
{
	bool debug = false;

	cvb::CvBlobs blobs; // typedef std::map<CvLabel,CvBlob *> CvBlobs;
	IplImage threshImg = IplImage(*img); // converts header info without copying underlying data
	//IplImage *threshImg = cvCreateImage(cvSize(img->cols, img->rows), 8, 1);

	IplImage *labelImg = cvCreateImage(cvSize(img->cols, img->rows), IPL_DEPTH_LABEL, 1);//Image Variable for blobs

	//Finding the blobs
	unsigned int result = cvLabel(&threshImg, labelImg, blobs);
	
	if (debug) {
		IplImage *frame = cvCreateImage(cvSize(img->cols, img->rows), 8, 3);
		//Rendering the blobs
		cvRenderBlobs(labelImg, blobs, frame, frame);
		//Showing the images
		cvNamedWindow("blobs", 1);
		cvShowImage("blobs", frame);
		cvWaitKey(0);
		cvDestroyWindow("blobs");
		cvReleaseImage(&frame);
	}

	/*
	// create unknown_segs_ matrices
	for (CvBlobs::iterator it = blobs.begin(); it != blobs.end(); ++it) {
		unsigned int label = (*it).first;
		unknown_segs_[cid][label].resize(cameras_[cid]->imgT_.rows*cameras_[cid]->imgT_.cols, 1);
		unknown_segs_[cid][label].setZero();
	}

	RecordUnknownSegmentationLabels(cid, labelImg);
	*/
	//Filtering the blobs
	//cvFilterByArea(blobs, 60, 500);

	cvReleaseImage(&labelImg);
	blobs.erase(blobs.begin(), blobs.end());
}

void Scene::RecordUnknownSegmentationLabels(int cid, IplImage* labelImg) // copies non-zero pixels from a blob img_label of depth IPL_DEPTH_LABEL (32 bits) to img_dest of depth IPL_DEPTH_32S.  Both must be single-channel images of the same height and width.  During copy, multiplies values by filter_color.  Blob labels are indexed starting at 1 so that an zero-value pixel in the single-channel blob label image is not a part of any blob/patch and a -1 means it's bordering a blob/patch and cannot take on a value.  For this value, since each filter color will have its own set of blob labels, use the label for the blob that was used to create it multiplied by the filter color+1.
{
	unsigned int *data_L;
	int stepLbl;
	int rows, cols;
	unsigned int label;
	int idx;

	data_L = reinterpret_cast<unsigned int *>(labelImg->imageData);
	stepLbl = labelImg->widthStep / ((float)labelImg->depth / 8.0);
	rows = labelImg->height;
	cols = labelImg->width;
	for (int row = 0; row<rows; row++, data_L += stepLbl)
	{
		for (int col = 0; col<cols; col++)
		{
			label = data_L[col];
			
			if (label == -1) label = 0; // labels of -1 are assigned in single-pixel borders around blobs in order to signify no other label is allowed there, whereas 0 is simply than non other is assigned.  These are used but known pixels.
			if (label != 0) {
				idx = PixIndexFwdCM(Point(col, row), rows);
				unknown_segs_[cid][label](idx, 0) = label;
			}
			
		}
	}
}

// masks also important in case expanded a mask because a pixel has no valid quantized disparity labels
void Scene::UpdateDepthMapsFromStereoData(map<int, MatrixXf> depth_maps) {
	bool debug = false;
	for (std::map<int, MatrixXf>::iterator it = depth_maps.begin(); it != depth_maps.end(); ++it) {
		int cid = (*it).first;
		cameras_[cid]->dm_->depth_map_ = (*it).second;
		if (debug) DisplayImages::DisplayGrayscaleImage(&cameras_[cid]->dm_->depth_map_, cameras_[cid]->dm_->height_, cameras_[cid]->dm_->width_, cameras_[cid]->orientation_);
	}
}

// masks also important in case expanded a mask because a pixel has no valid quantized disparity labels
void Scene::UpdateDepthMapsAndMasksFromStereoData(map<int, MatrixXf> depth_maps, map<int, Matrix<bool, Dynamic, 1>> masks, map<int, int> heights, map<int, int> widths) {
	for (std::map<int, MatrixXf>::iterator it = depth_maps.begin(); it != depth_maps.end(); ++it) {
		int cid = (*it).first;
		cameras_[cid]->dm_->depth_map_ = (*it).second;
	}
	for (std::map<int, Matrix<bool, Dynamic, 1>>::iterator it = masks.begin(); it != masks.end(); ++it) {
		int cid = (*it).first;
		Matrix<bool, Dynamic, Dynamic> mask = (*it).second;
		mask.resize(heights[cid], widths[cid]);
		Mat maskCV = Mat::zeros(heights[cid], widths[cid], CV_8UC1);
		EigenOpenCV::eigen2cv(mask, maskCV);
		maskCV = maskCV * 255;
		maskCV.copyTo(cameras_[cid]->imgMask_);
	}
}

void Scene::DebugEpipolarGeometry() {
	for (std::map<int, Camera*>::iterator it1 = cameras_.begin(); it1 != cameras_.end(); ++it1) {
		int cid1 = (*it1).first;
		if ((!(*it1).second->posed_) ||
			(!(*it1).second->enabled_)) continue;
		
		int idx;
		for (int r = 0; r < cameras_[cid1]->imgT_.rows; r++) {
			for (int c = 0; c < cameras_[cid1]->imgT_.cols; c++) {
				idx = PixIndexFwdCM(Point(c, r), cameras_[cid1]->imgT_.rows);
				if ((cameras_[cid1]->dm_->depth_map_(idx, 0) == 0) &&
					(cameras_[cid1]->imgMask_.at<uchar>(r, c) > 0)) { // unknown pixel criteria
					
					// point to project into epiline in other cameras
					Point3d pt(c, r, 1);
					//std::vector<Point2d> points;
					//points.push_back(pt);

					// draw the point in question on image cid1 and display it
					cv::Scalar color(0, 0, 255);
					cv::Mat outImg1(cameras_[cid1]->imgT_.rows, cameras_[cid1]->imgT_.cols, CV_8UC3);
					cameras_[cid1]->imgT_.copyTo(outImg1);
					cv::circle(outImg1, Point(c, r), 3, color, -1, CV_AA);
					display_mat(&outImg1, "epipolar line", cameras_[cid1]->orientation_);

					// for each other camera
					for (std::map<int, Camera*>::iterator it2 = cameras_.begin(); it2 != cameras_.end(); ++it2) {
						int cid2 = (*it2).first;
						if ((!(*it2).second->posed_) || // the second camera doesn't need to have a depth map, but does need to be posed, enabled, and have a mask
							(!(*it2).second->enabled_) ||
							(cid1 == cid2)) continue;

						// calculate fundamental matrix F from cid1 to cid2: F = e' x (P')(P+)x where e' is the epipole in the second image and is defined by e' = P'C where C is the camera center in cid1; P' is the projection matrix of cid2; P+ is the pseudo-inverse of the projection matrix in cid1, which equals its inverse if the matrix is invertible; F is 3x3; x is the point in cid1 to project; so we end up taking the cross product of two vectors to find the epiline
						Matrix<double, 4, 1> C; // homogeneous point of camera center in cid1
						C(0, 0) = cameras_[cid1]->pos_.x;
						C(1, 0) = cameras_[cid1]->pos_.y;
						C(2, 0) = cameras_[cid1]->pos_.z;
						C(3, 0) = 1.; // homogeneous coordinate
						Matrix<double, 3, 1> e_prime = cameras_[cid2]->P_ * C; // epipole e' of C projected into cid2

						Matrix<double, 3, 1> x;
						x(0, 0) = c;
						x(1, 0) = r;
						x(2, 0) = 1;
						Matrix<double, 3, 1> PPx = cameras_[cid2]->P_ * cameras_[cid1]->Pinv_ * x;

						Point3d PPx_pt(PPx(0, 0), PPx(1, 0), PPx(2, 0));
						Point3d e_prime_pt(e_prime(0, 0), e_prime(1, 0), e_prime(2, 0));
						Point3d epiline = e_prime_pt.cross(PPx_pt);

						// draw the epiline on image cid2 and display it
						cv::Mat outImg2(cameras_[cid2]->imgT_.rows, cameras_[cid2]->imgT_.cols, CV_8UC3);
						cameras_[cid2]->imgT_.copyTo(outImg2);

						cv::line(outImg2,
							cv::Point(0, -epiline.z / epiline.y),
							cv::Point(cameras_[cid2]->imgT_.cols, -(epiline.z + epiline.x * cameras_[cid2]->imgT_.cols) / epiline.y),
							color);
						display_mat(&outImg2, "epipolar line", cameras_[cid2]->orientation_);
					}
				}
			}
		}
	}
}
/*
// returns true if a pixel color could be retrieved, false otherwise
// updates color with the value if one is retrieved
// color may not be retrieved if a valid simulated triangle exists at that position given where pixels are not masked and they don't vary in depth more than allowed for a single triangle
bool Scene::SimulateRegularMeshPixelColor(int cid, double xpos, double ypos, Vec3b &color) {
	int x = static_cast<int>(xpos);
	int y = static_cast<int>(ypos);
	if ((x < 0) ||
		(y < 0) ||
		(x >= (cameras_[cid]->width_-1)) ||
		(y >= (cameras_[cid]->height_-1))) return false;
	double u = xpos - x;
	double v = ypos - y;

	// try first triangle
	if ((cameras_[cid]->imgMask_.at<uchar>(y, x)) &&
		(cameras_[cid]->imgMask_.at<uchar>(y + 1, x + 1)) &&
		(cameras_[cid]->imgMask_.at<uchar>(y + 1, x))) { // pixels must be masked in
		// pixels must be within a tight depth range
		float depth_ul = cameras_[cid]->dm_->depth_map_(y, x);
		float depth_lr = cameras_[cid]->dm_->depth_map_(y + 1, x + 1);
		float depth_ll = cameras_[cid]->dm_->depth_map_(y + 1, x);
		if ((abs(depth_ul - depth_lr) <= GLOBAL_TRIANGLE_DEPTH_DIFF_MAX) &&
			(abs(depth_ul - depth_ll) <= GLOBAL_TRIANGLE_DEPTH_DIFF_MAX) &&
			(abs(depth_ - depth_lr) <= GLOBAL_TRIANGLE_DEPTH_DIFF_MAX))
	}

	// try second triangle
}
*/

void Scene::ExportDepthMapsEXR(string filepath) {
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		if ((!(*it).second->enabled_) ||
			(!(*it).second->posed_) ||
			(!(*it).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

		(*it).second->dm_->Export(filepath);
	}
}

void Scene::SyncDepthMaps() {
	for (std::map<int, Camera*>::iterator it1 = cameras_.begin(); it1 != cameras_.end(); ++it1) {
		if ((!(*it1).second->enabled_) ||
			(!(*it1).second->posed_) ||
			(!(*it1).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

		int cid1 = (*it1).first;

		for (std::map<int, Camera*>::iterator it2 = cameras_.begin(); it2 != cameras_.end(); ++it2) {
			if ((!(*it2).second->enabled_) ||
				(!(*it2).second->posed_) ||
				(!(*it2).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

			int cid2 = (*it2).first;
			if (cid2 <= cid1) continue;

			Matrix<bool, Dynamic, 1> change_map((*it2).second->dm_->depth_map_.rows(), (*it2).second->dm_->depth_map_.cols());
			(*it1).second->ReprojectMeshDepths((*it2).second->view_dir_, &(*it2).second->P_, &(*it2).second->RT_, &(*it2).second->dm_->depth_map_, &change_map);
		}
	}
}

// updates depth map data for camera cid_ref using depth data from all other cameras
// smallest cid_ref camera space depth wins races
// 0 depths automatically overwritten
// due to rounding among pixel coordinates on reprojection, may slightly violate masks, despite CleanDepths() already having been run; we unfortunately don't yet have valid ranges available to snap to, and they would only be calculated for "unknown" pixels anyway, and here we are updating what will become "known" pixels
void Scene::UpdateDepthsByCrowdWisdom() {
	bool debug = false;
	double t;
	bool timing = true;
	if (timing) t = (double)getTickCount();

	cout << "Scene::UpdateDepthsByCrowdWisdom()" << endl;

	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;
		if ((!(*it).second->posed_) ||
			(!(*it).second->enabled_) ||
			(!(*it).second->has_depth_map_)) continue;

		int num_nonzero_depths = (*it).second->dm_->depth_map_.count();
		if (num_nonzero_depths == 0)
			continue; // must have some depth information

		int h = (*it).second->height_;
		int w = (*it).second->width_;

		Matrix<float, 4, Dynamic> Iws;
		InverseProjectSStoWS(w, h, &(*it).second->imgMask_, &(*it).second->dm_->depth_map_, &(*it).second->calib_.Kinv_, &(*it).second->RTinv_, &Iws);

		for (std::map<int, Camera*>::iterator it_ref = cameras_.begin(); it_ref != cameras_.end(); ++it_ref) {
			int cid_ref = (*it_ref).first;
			if (cid_ref == cid) continue; // also do for cid_ref
			if ((!(*it_ref).second->posed_) ||
				(!(*it_ref).second->enabled_) ||
				(!(*it_ref).second->has_depth_map_)) continue;

			int h_ref = (*it_ref).second->height_;
			int w_ref = (*it_ref).second->width_;

			// project points from cid screen space to cid_ref screen space to get destination pixels
			Matrix<float, 4, Dynamic> Ics_ref = (*it_ref).second->RT_.cast<float>() * Iws;
			Ics_ref.row(2) = Ics_ref.row(2).cwiseQuotient(Ics_ref.row(3)); // divide by homogeneous coordinates

			//if (debug) DebugPrintMatrix(&Ics_ref, "Ics_ref");

			// project points from cid screen space to cid_ref camera space to get cid_ref camera space depth information
			Matrix<float, 3, Dynamic> Iss_ref = (*it_ref).second->P_.cast<float>() * Iws;
			Matrix<float, 1, Dynamic> N = Iss_ref.row(2).array().inverse(); // determine homogeneous coordinates to divide by
			Iss_ref.row(0) = Iss_ref.row(0).cwiseProduct(N); // divide by homogeneous coordinates
			Iss_ref.row(1) = Iss_ref.row(1).cwiseProduct(N); // divide by homogeneous coordinates

			//if (debug) DebugPrintMatrix(&Iss_ref, "Iss_ref");

			float depth_new, depth_curr;
			for (int idx_known_in = 0; idx_known_in < Iss_ref.cols(); idx_known_in++) {
				int x = floor(Iss_ref(0, idx_known_in));
				int y = floor(Iss_ref(1, idx_known_in));
				if ((x < 0) ||
					(y < 0) ||
					(x >= (w_ref - 1)) ||
					(y >= (h_ref - 1))) // projected location must be within valid screen space of output image
					continue;
				depth_new = Ics_ref(2, idx_known_in); // new disparity information is taken from cid_ref camera space

				if (depth_new != 0) {
					// update appropriate output pixels that surround floating point projected location
					for (int i = 0; i <= 1; i++) {
						for (int j = 0; j <= 1; j++) {
							if ((*it_ref).second->imgMask_.at<uchar>(y + j, x + i) == 0) continue; // landed on masked-out pixel, so skip it
							depth_curr = (*it_ref).second->dm_->depth_map_(y + j, x + i);
							if ((depth_curr == 0) ||
								(depth_new < depth_curr)) {
								cameras_[cid_ref]->dm_->depth_map_(y + j, x + i) = depth_new;
							}
						}
					}
				}
			}
		}
	}

	if (debug) {
		for (std::map<int, Camera*>::iterator it_ref = cameras_.begin(); it_ref != cameras_.end(); ++it_ref) {
			int cid_ref = (*it_ref).first;
			if ((!(*it_ref).second->posed_) ||
				(!(*it_ref).second->enabled_) ||
				(!(*it_ref).second->has_depth_map_)) continue;

			cout << "Scene::UpdateDepthsByCrowdWisdom() result for cid_ref " << cid_ref << endl;
			cameras_[cid_ref]->dm_->DisplayDepthImage();
		}
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Scene::UpdateDepthsByCrowdWisdom() running time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// returns 3 by (ss_w*ss_h) data structure with homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h assuming row-major order of indices
Matrix<float, 3, Dynamic> Scene::ConstructSSCoordsCM(int ss_w, int ss_h, Mat* imgMask, Matrix<float, Dynamic, Dynamic> *depth_map) {
	bool debug = false;

	Matrix<float, 3, Dynamic> I(3, ss_w*ss_h); // 3xn matrix of homogeneous screen space points where n is the number of pixels in imgT_
	I.row(2).setConstant(1.);

	float *pD = depth_map->data();
	int idx = 0;
	for (int c = 0; c < ss_w; c++) {
		for (int r = 0; r < ss_h; r++) {
			if ((imgMask->at<uchar>(r, c) == 0) ||
				(*pD == 0.)) {
				pD++;
				continue;
			}
			//idx = PixIndexFwdCM(Point(c, r), ss_h);
			I(0, idx) = (float)c;
			I(1, idx) = (float)r;
			idx++;
			pD++;
		}
	}

	if (debug) DebugPrintMatrix(&I, "I");

	Matrix<float, 3, Dynamic> Icull(3, idx);
	Icull = I.block(0, 0, 3, idx);

	if (debug) DebugPrintMatrix(&Icull, "Icull");

	return Icull;
}

// inverse projects screen space points (screen space dimensions ss_width x ss_height) with depths given by imgD from screen space to world space using Kinv and RTinv, updating a 4xn matrix of type float of the corresponding points in world space
// imgD is a 2D depth image matrix of size ss_height x ss_width (n points) whose depth values are in units that match Kinv
// Kinv is a 3x3 inverse calibration matrix of type CV_64F
// RTinv is a 4x4 inverse RT matrix of type CV_64F
// Iws must be a 4x(ss_width*ss_height) matrix
// updates Iws with homogeneous world space points as (x,y,z,w)
// expects everything in column-major
void Scene::InverseProjectSStoWS(int ss_width, int ss_height, Mat* imgMask, Matrix<float, Dynamic, Dynamic> *depth_map, Matrix3d *Kinv, Matrix4d *RTinv, Matrix<float, 4, Dynamic> *Iws) {
	bool debug = false;

	bool timing = false; double t;
	if (timing) t = (double)getTickCount();
	
	assert(depth_map->rows() == ss_height && depth_map->cols() == ss_width);

	// scale u,v,w by the desired depth amount to get homogeneous coordinates that reflect the depth after transformation
	Matrix<float, 3, Dynamic> I = ConstructSSCoordsCM(ss_width, ss_height, imgMask, depth_map);

	// cull depth map according to mask
	Matrix<float, 1, Dynamic> dm(1, I.cols()); // Iws and related matrices are row-major interpretations of their 2D counterparts
	float *pD = depth_map->data();
	float *pDt = dm.data();
	for (int c = 0; c < ss_width; c++) {
		for (int r = 0; r < ss_height; r++) {
			if ((imgMask->at<uchar>(r, c) == 0) ||
				(*pD == 0.)) {
				pD++;
				continue;
			}
			*pDt++ = *pD++;
		}
	}

	if (debug) DebugPrintMatrix(&I, "I");
	if (debug) DebugPrintMatrix(&dm, "dm");
	// use depth values from depth_map
	I.row(0) = I.row(0).cwiseProduct(dm);
	I.row(1) = I.row(1).cwiseProduct(dm);
	I.row(2) = I.row(2).cwiseProduct(dm);
	if (debug) DebugPrintMatrix(&I, "I");

	// transform screen space to camera space - transform u,v to x,y, then add rows for z (equal to each depth value) and w (equal to 1.0)
	Matrix<float, 2, 3> Kinv_uvonly;
	Kinv_uvonly.row(0) = Kinv->row(0).cast<float>();
	Kinv_uvonly.row(1) = Kinv->row(1).cast<float>();
	if (debug) DebugPrintMatrix(&Kinv_uvonly, "Kinv_uvonly");
	Matrix<float, 2, Dynamic> Ics_xyonly = Kinv_uvonly * I; // Ics is homogeneous 4xn matrix of camera space points
	I.resize(3, 0);
	if (debug) DebugPrintMatrix(&Ics_xyonly, "Ics_xyonly");
	Iws->resize(4, Ics_xyonly.cols());
	Iws->row(3).setOnes(); // Iws at this point is still in camera space until we multiply it below by inverse extrinsics
	Iws->row(0) = Ics_xyonly.row(0);
	Iws->row(1) = Ics_xyonly.row(1);
	Ics_xyonly.resize(2, 0);
	if (debug) DebugPrintMatrix(Iws, "Iws");

	// in camera space, set z to depth value and w to 1 (already scaled x,y in homogeneous screen space)
	// use depth values from dm
	Iws->row(2) = dm;
	dm.resize(1, 0);
	if (debug) DebugPrintMatrix(Iws, "Iws");

	// transform camera space positions to world space
	(*Iws) = (*RTinv).cast<float>() * (*Iws); // Iws is homogeneous 4xn matrix of world space points; RTinv includes transformation from Agisoft space to world space
	Matrix<float, 1, Dynamic> H = Iws->row(3).array().inverse();

	if (debug) DebugPrintMatrix(Iws, "Iws");

	// normalize by homogeneous value
	Iws->row(0) = Iws->row(0).cwiseProduct(H);
	Iws->row(1) = Iws->row(1).cwiseProduct(H);
	Iws->row(2) = Iws->row(2).cwiseProduct(H);
	Iws->row(3).setOnes();

	if (debug) DebugPrintMatrix(Iws, "Iws");

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Scene::InverseProjectSStoWS() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// exports each camera's image after masking it
void Scene::ExportMaskedCameraImages() {
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		if ((!(*it).second->enabled_) ||
			(!(*it).second->posed_) ||
			(!(*it).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

		int cid = (*it).first;

		cameras_[cid]->SaveMaskedImage(name_);
	}
}


// in order to determine the proper dilation element size for masks when constraining depth values, view projections of known pixels from reference image in each other image after dilation to discern overlap visually
// inputs and outputs must be set first
void Scene::DebugViewPoseAccuracy(int cid_ref) {
	// project known pixels into each other image and change destination pixels' red component to 255 to see accuracy of projection matrices
	int h_ref = cameras_[cid_ref]->height_;
	int w_ref = cameras_[cid_ref]->width_;
	Matrix<bool, Dynamic, Dynamic> known = cameras_[cid_ref]->dm_->depth_map_.array() > 0.;
	int num_known = known.count();
	Matrix<float, Dynamic, 4> WCkd(num_known, 4);
	float *pWCkd_x = WCkd.data();
	float *pWCkd_y = WCkd.data() + num_known;
	float *pWCkd_z = WCkd.data() + 3 * num_known;
	float depth;
	for (int i = 0; i < (h_ref * w_ref); i++) {
		Point pt = PixIndexBwdCM(i, h_ref);
		depth = cameras_[cid_ref]->dm_->depth_map_(pt.y, pt.x);
		if (depth <= 0.) continue;
		*pWCkd_x++ = pt.x;
		*pWCkd_y++ = pt.y;
		*pWCkd_z++ = 1 / depth;
	}
	WCkd.col(2).setOnes();

	Matrix<float, 1, 4> ext;
	ext << 0., 0., 0., 1.;
	Matrix4f Pout_ext;
	Pout_ext << cameras_[cid_ref]->P_.cast<float>(), ext;
	Matrix4f Pout_ext_inv = Pout_ext.inverse();

	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		int cid = (*it).first;
		if (cid == cid_ref) continue;
		if ((!(*it).second->enabled_) ||
			(!(*it).second->posed_) ||
			(!(*it).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information
		
		Matrix<float, 3, 4> Pout2in = cameras_[cid]->P_.cast<float>() * Pout_ext_inv;

		Matrix<float, Dynamic, 3> T2 = WCkd * Pout2in.transpose();
		Matrix<float, Dynamic, 1> N2 = T2.col(2).array().inverse();
		T2.col(0) = T2.col(0).cwiseProduct(N2);
		T2.col(1) = T2.col(1).cwiseProduct(N2);

		int h = (*it).second->height_;
		int w = (*it).second->width_;
		Mat img = Mat::zeros(h, w, CV_8UC3);
		cameras_[cid]->imgT_.copyTo(img);

		if (GLOBAL_MASK_DILATION > 0) {
			int morph_type = MORPH_RECT; // MORPH_ELLIPSE
			int morph_size = GLOBAL_MASK_DILATION; // get rid of small regional markings
			Mat element = getStructuringElement(morph_type,
				Size(2 * morph_size + 1, 2 * morph_size + 1),
				Point(morph_size, morph_size));
			dilate(img, img, element);
		}

		Matrix<int, Dynamic, 1> X_round2 = T2.col(0).cast<int>();
		Matrix<int, Dynamic, 1> Y_round2 = T2.col(1).cast<int>();
		int *pX = X_round2.data();
		int *pY = Y_round2.data();
		int x, y, k;
		for (int i = 0; i < T2.rows(); i++) {
			x = *pX++;
			y = *pY++;
			if ((x >= 0) &&
				(y >= 0) &&
				(x < w) &&
				(y < h)) {
				k = h * x + y; // col major index of current position
				Vec3b pix = img.at<Vec3b>(y, x);
				pix[2] = 255;
				img.at<Vec3b>(y, x) = pix;
			}
		}
		display_mat(&img, "test", cameras_[cid]->orientation_);
	}
}

// exports camera ID, WS position, WS view direction, and WS up direction; also exports bounding volume info
void Scene::ExportSceneInfo() {
	UpdatePointCloudBoundingVolume();

	std::string fn = GLOBAL_FILEPATH_DATA + name_ + "\\camerainfo.txt";
	ofstream myfile;
	myfile.open(fn);

	myfile << "bounding volume minimum " << bv_min_.x << " " << bv_min_.y << " " << bv_min_.z << endl;
	myfile << "bounding volume maximum " << bv_max_.x << " " << bv_max_.y << " " << bv_max_.z << endl;
	myfile << endl;

	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		if ((!(*it).second->enabled_) ||
			(!(*it).second->posed_) ||
			(!(*it).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

		int cid = (*it).first;

		Point3d pos = Camera::GetCameraPositionWS(&(*it).second->RTinv_);
		Point3d view_dir = Camera::GetCameraViewDirectionWS(&(*it).second->RTinv_);
		Point3d up_dir = Camera::GetCameraUpDirectionWS(&(*it).second->RTinv_);

		myfile << "id " << cid << endl;
		myfile << "file " << (*it).second->fn_ << endl;
		myfile << "position " << pos.x << " " << pos.y << " " << pos.z << endl;
		myfile << "viewdir " << view_dir.x << " " << view_dir.y << " " << view_dir.z << endl;
		myfile << "updir " << up_dir.x << " " << up_dir.y << " " << up_dir.z << endl;
		myfile << endl;
	}
	myfile.close();
}

void Scene::CleanFacesAgainstMasks(int cid_mesh) {
	// build world space points and map between them and faces
	Matrix<double, 4, Dynamic> Iws;
	Iws.row(3).setOnes();
	map<int, Vec3i> fid_map; // map of face ID => Iws indices for each of the three vertices of the face
	int iws_idx = 0;
	for (map<int, Vec3i>::iterator it = cameras_[cid_mesh]->mesh_faces_.begin(); it != cameras_[cid_mesh]->mesh_faces_.end(); ++it) {
		int fid = (*it).first;
		Vec3i vs = (*it).second;
		Vec3i iws_indices;
		for (int i = 0; i < 3; i++) {
			int vid = vs[i];
			Point3d v = cameras_[cid_mesh]->mesh_vertices_[vid];
			Iws(0, iws_idx) = v.x;
			Iws(1, iws_idx) = v.y;
			Iws(2, iws_idx) = v.z;
			iws_indices[i] = iws_idx;
			iws_idx++;
		}
		fid_map[fid] = iws_indices;
	}

	// validate mesh against inverse masks
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		if ((!(*it).second->enabled_) ||
			(!(*it).second->posed_) ||
			(!(*it).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information
		
		int cid = (*it).first;

		// reproject world space coordinates to other camera's screen space
		Matrix<double, 3, Dynamic> I_dest = cameras_[cid]->P_ * Iws; // note the matrix multiplication property: Ainv * A = A * Ainv
		Matrix<double, 1, Dynamic> H = I_dest.row(2).array().inverse();
		I_dest.row(0) = I_dest.row(0).cwiseProduct(H);
		I_dest.row(1) = I_dest.row(1).cwiseProduct(H);

		// validate each face
		for (map<int, Vec3i>::iterator itf = fid_map.begin(); itf != fid_map.end(); ++itf) {
			int fid = (*itf).first;
			Vec3i iws_indices = (*itf).second;
			map<int, Point> vs;
			for (int i = 0; i < 2; i++) {
				Point v;
				v.x = round(Iws(0, iws_indices[i]));
				v.y = round(Iws(1, iws_indices[i]));
				// crop vertices to image bounds; can't reject faces with vertices outside the image since an image could be a close-up cropped shot, so instead validate as best we can for in-image portion of the projection against the inverse mask
				v.x = max(v.x, 0); v.x = min(v.x, cameras_[cid]->width_);
				v.y = max(v.y, 0); v.y = min(v.y, cameras_[cid]->height_);

				vs[i] = v;
			}

			// find labels of 3 vertices in cid - if labels are the same, then face passes
			unsigned int l1, l2, l3;
			l1 = cameras_[cid]->seg_(vs[0].y, vs[0].x);
			l2 = cameras_[cid]->seg_(vs[1].y, vs[1].x);
			l3 = cameras_[cid]->seg_(vs[2].y, vs[2].x);
			if ((l1 == l2) &&
				(l2 == l3))
				continue;

			// create line iterator for each of 3 edges of face and grab pixels from mask - if any pixel is less than GLOBAL_MIN_MASKSEG_LINEVAL, the face (and its corresponding normal) should be removed from the list for cid_mesh
			// grabs pixels along the line (pt1, pt2)
			// from 8-bit 3-channel image to the buffer
			bool fails = false;
			int vi_last = 2;
			int vi = 0;
			while ((!fails) &&
				(vi < 2)) {
				LineIterator itl(cameras_[cid]->imgMask_, vs[vi_last], vs[vi], 8);
				for (int i = 0; i < itl.count; i++, ++itl) {
					int val = cameras_[cid]->imgMask_.at<uchar>(itl.pos());
					if (val < GLOBAL_MIN_MASKSEG_LINEVAL) { // face fails
						fails = true;
						break;
					}
				}

				vi_last = vi;
				vi++;
			}
			if (fails) {
				cameras_[cid_mesh]->mesh_faces_.erase(fid);
				cameras_[cid_mesh]->mesh_normals_.erase(fid);
			}
		}
	}
}