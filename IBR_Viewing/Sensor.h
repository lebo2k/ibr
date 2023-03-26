#ifndef Sensor_H
#define Sensor_H

#include "Globals.h"
#include "Calibration.h"

/*
	note that fx = f * sx and fy = f * sy where pixel width = 1 / sx and pixel height = 1 / sy, BUT the pixel width and height given here are ideal numbers assuming the camera imaging sensor is exactly the size it should be (e.g. 30mm x 24mm).  fx and fy in the calibration object are calculated a different way, not going through sx and sy, and give more accurate results that reflect the actual sensor size from manufacture
	also note that camera calibration units are given in mm while world space units are in m.  This is ok because camera calibration mm units are canceled out during construction of K, leaving pixel units only, so they don't have to match world space units.
*/

class Sensor {

private:

public:

	int id_;
	std::string label_;
	int width_, height_; // resolution in pixels
	double pixel_width_, pixel_height_; // pixel width and height in mm
	double focal_length_; // in mm
	Calibration calib_;

	// Constructors / destructor
	Sensor();
	~Sensor();

	void Init(xml_node<> *sensor_node);

	/*
	// Accessors
	inline int id() { return id_; }
	inline int width() { return width_; }
	inline int height() { return height_; }
	inline double pix_width() { return pix_width_; }
	inline double pix_height() { return pix_height_; }
	inline double focal_length() { return focal_length_; }
	*/

	void Print(); // debug printing
};

#endif