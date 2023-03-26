#include "Scene.h"

Scene::Scene() {
	depth_downscale_ = 1.;
}

Scene::~Scene() {
}

void Scene::Init(std::string agisoft_filename) {
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	int found = agisoft_filename.find(".");
	if (found != std::string::npos) name_ = agisoft_filename.substr(0,found-1);
	else name_ = agisoft_filename;

	xml_document<> doc;
	xml_node<> *root_node, *chunk_node, *curr_node;
	// Read the xml file into a vector
	ifstream theFile(GLOBAL_FILEPATH_INPUT + agisoft_filename);
	vector<char> buffer((istreambuf_iterator<char>(theFile)), istreambuf_iterator<char>());
	buffer.push_back('\0');
	// Parse the buffer using the xml file parsing library into doc 
	doc.parse<0>(&buffer[0]);
	// Find our root node
	root_node = doc.first_node("document");
	chunk_node = root_node->first_node("chunk");

	std::string s;
	
	// import AgisoftToWorld transformation matrix for the chunk
	AgisoftToWorld_ = cv::Mat::eye(4, 4, CV_64F); // default if fail to find it
	xml_node<> *chunk_transform_node;
	chunk_transform_node = chunk_node->first_node("transform");
	if (chunk_transform_node != 0) {
		s = chunk_transform_node->value();
		if (!s.empty()) {
			AgisoftToWorldinv_ = ParseString_Matrix64F(s, 4, 4); // 4x4 matrix specifying chunk location in the world coordinate system
			AgisoftToWorld_ = AgisoftToWorldinv_.inv();
			agisoft_to_world_scale_ = 1 / ScaleFromExtrinsics(&AgisoftToWorld_);
		}
	}

	// import sensors
	xml_node<> *sensors_node, *sensor_node;
	sensors_node = chunk_node->first_node("sensors");
	for (xml_node<> * sensor_node = sensors_node->first_node("sensor"); sensor_node; sensor_node = sensor_node->next_sibling()) {
		Sensor *sensor = new Sensor();
		sensor->Init(sensor_node);
		sensors_[sensor->id_] = sensor;
	}

	// import cameras
	xml_node<> *cameras_node, *camera_node;
	cameras_node = chunk_node->first_node("cameras");
	for (xml_node<> * camera_node = cameras_node->first_node("camera"); camera_node; camera_node = camera_node->next_sibling()) {
		Camera *cam = new Camera();
		cam->Init(camera_node, &AgisoftToWorld_, &AgisoftToWorldinv_);
		cam->InitSensor(sensors_[cam->sensor_id_]);
		cameras_[cam->id_] = cam;
	}

	CullCameras();

	// set up depth map nodes
	xml_node<> *depth_maps_node, *frame_node, *depth_map_node;
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

			cameras_[cid]->InitDepthMap(depthmap_node, agisoft_to_world_scale_, depth_downscale_);
		}
	}

	// undistort images
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		(*it).second->UndistortPhotos();
	}

	// compute world space point projections
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		(*it).second->InitWorldSpaceProjection();
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

// saves Iws_ for each camera
void Scene::SaveCamerasWSPointProjections() {
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		(*it).second->Save_Iws(name_);
	}
}

// determines point cloud bounding box volume across all camera views
void Scene::DeterminePointCloudBoundingVolume(Point3d &bv_min, Point3d &bv_max) {
	bool debug = true;

	bv_min = Point3d(0., 0., 0.);
	bv_max = Point3d(0., 0., 0.);
	bool first = true;
	Point3d cam_bv_min, cam_bv_max;
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		(*it).second->DeterminePointCloudBoundingVolume(cam_bv_min, cam_bv_max);

		if (first) {
			bv_min.x = cam_bv_min.x;
			bv_min.y = cam_bv_min.y;
			bv_min.z = cam_bv_min.z;
			bv_max.x = cam_bv_max.x;
			bv_max.y = cam_bv_max.y;
			bv_max.z = cam_bv_max.z;
		}
		else {
			if (cam_bv_min.x < bv_min.x) bv_min.x = cam_bv_min.x;
			if (cam_bv_min.y < bv_min.y) bv_min.y = cam_bv_min.y;
			if (cam_bv_min.z < bv_min.z) bv_min.z = cam_bv_min.z;
			if (cam_bv_max.x > bv_max.x) bv_max.x = cam_bv_max.x;
			if (cam_bv_max.y > bv_max.y) bv_max.y = cam_bv_max.y;
			if (cam_bv_max.z > bv_max.z) bv_max.z = cam_bv_max.z;
		}
		first = false;
	}

	if (debug) cout << "Bounding volume minimum " << bv_min << ", maximum " << bv_max << endl;
}

// gets an ordered list of num closest cameras to view of I0 (closest first)
// I0_pos is the position of camera I0 in world space; I0_view_dir is the viewing direction of camera I0 in world space
// arg num must be a positive number; if it is greater than or equal to the number of cameras, an ordered list of all available cameras will be returned
// exclude_cam_ids holds a list of camera IDs that should be excluded from the list of closest camera IDs
std::vector<int> Scene::GetClosestCams(Point3d I0_pos, Point3d I0_view_dir, std::vector<int> exclude_cam_ids, int num) {
	assert(num >= 0, "Scene::GetClosestCams() num must be a positive number");
	std::vector<int> cams = OrderCams(I0_pos, I0_view_dir);
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

// returns an ordered list of camera IDs from member cameras according to the closest view to I0
// closest camera is first in the list, farthest last
// I0_pos is the position of camera I0 in world space; I0_view_dir is the viewing direction of camera I0 in world space
std::vector<int> Scene::OrderCams(Point3d I0_pos, Point3d I0_view_dir) {
	bool debug = false;

	if (debug) cout << "I0_pos " << I0_pos << endl;

	std::vector<std::pair<int, double>> cam_dist; // vector of pairs of: camera ID, distance from I0
	for (std::map<int, Camera*>::iterator it = cameras_.begin(); it != cameras_.end(); ++it) {
		std::pair<int, double> c;
		c.first = (*it).second->id_;
		c.second = vecdist(I0_pos, (*it).second->pos_);
		double vdd = I0_view_dir.ddot((*it).second->view_dir_); // ensure camera is looking in the same general direction as the given view_dir
		if (vdd > 0) cam_dist.push_back(c);
	}
	std::sort(cam_dist.begin(), cam_dist.end(), pairCompare);
	
	std::vector<int> cams_ordered;
	int cidi, cidj;
	double distij;
	bool pass;
	for (std::vector<std::pair<int, double>>::iterator iti = cam_dist.begin(); iti != cam_dist.end(); ++iti) {
		cidi = (*iti).first;
		pass = true;
		for (std::vector<std::pair<int, double>>::iterator itj = cam_dist.begin(); itj != cam_dist.end(); ++itj) {
			cidj = (*itj).first;
			if (cidi == cidj) break; // break out of inner loop because have examined all nodes it2 closer to I0 than node it1; if Ij is closer to I0 than Ii is...

			/*
			distij = vecdist(cameras_[cidi]->pos_, cameras_[cidj]->pos_);
			if (distij < (*iti).second) { // ...and if Ii is closer to Ij than Ii is to I0, remove Ii from the list
				pass = false;
				break;
			}
			*/
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