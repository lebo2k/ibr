#include "Sensor.h"

Sensor::Sensor() {
	id_ = -1;
	label_ = "";
	width_ = 0;
	height_ = 0;
	pixel_width_ = 0.;
	pixel_height_ = 0.;
	focal_length_ = 0.;
}

Sensor::~Sensor() {
}

// set member values according to data in node argument from Agisoft doc.xml file
void Sensor::Init(xml_node<> *sensor_node) {
	assert(strcmp(sensor_node->name(), "sensor") == 0, "Sensor::Init() wrong arg node type passed");

	std::string s;
	xml_node<> *curr_node;

	s = sensor_node->first_attribute("id")->value();
	if (!s.empty()) id_ = convert_string_to_int(s);

	label_ = sensor_node->first_attribute("label")->value();

	for (xml_node<> * curr_node = sensor_node->first_node("resolution"); curr_node; curr_node = curr_node->next_sibling()) {
		if (strcmp(curr_node->name(), "resolution") == 0) {
			s = curr_node->first_attribute("width")->value();
			if (!s.empty()) width_ = convert_string_to_int(s);
			s = curr_node->first_attribute("height")->value();
			if (!s.empty()) height_ = convert_string_to_int(s);
		}
		else if (strcmp(curr_node->name(), "property") == 0) {
			double val = 0.;
			s = curr_node->first_attribute("value")->value();
			if (!s.empty()) val = convert_string_to_double(s);
			else continue;

			if (strcmp(curr_node->first_attribute("name")->value(), "pixel_width") == 0)
				pixel_width_ = val; // mm
			else if (strcmp(curr_node->first_attribute("name")->value(), "pixel_height") == 0)
				pixel_height_ = val; // mm
			else if(strcmp(curr_node->first_attribute("name")->value(), "focal_length") == 0)
				focal_length_ = val; // mm
		}
		else if (strcmp(curr_node->name(), "calibration") == 0)
			calib_.Init(curr_node); // not all sensors are necessarily calibrated
	}
}

// debug printing
void Sensor::Print() {
	cout << "Sensor" << endl;
	cout << "ID " << id_ << endl;
	cout << "Label " << label_ << endl;
	cout << "Width, height " << width_ << ", " << height_ << " pixels" << endl;
	cout << "Pixel width, pixel height " << pixel_width_ << ", " << pixel_height_ << " mm" << endl;
	cout << "Focal length " << focal_length_ << " mm" << endl;
	calib_.Print();
	cout << endl;
}
