#include <iostream>
#include <string>
#include <vector>

#include "opencv.hpp"
#include "cvblob.h"

using namespace cv;
using namespace cvb;
using namespace std;

static const double PI = 3.1415926536;

struct Options
{
	int threshold;
};

struct Star
{
	double x, y;
	double r;

	Star(double x, double y, double r)
	{
		this->x = x;
		this->y = y;
		this->r = r;
	}
};

typedef vector<Star> Stars;

void die(const string & msg)
{
	cerr << "Error: " << msg << endl;
	exit(1);
}


void getStars(const vector<string> & fn, vector<Stars *> & stars, const Options & opt)
{
	stars.clear();
	for (vector<string>::const_iterator it = fn.begin(); it != fn.end(); ++it)
	{
		cout << "  * " << *it << ": ";

		// load the image
		Mat srcimg = imread(*it, CV_LOAD_IMAGE_GRAYSCALE);
		Mat image;
		threshold(srcimg, image, opt.threshold, 255, THRESH_BINARY);

		// find the blobs
		CvBlobs blobs;
		IplImage img = image;
		IplImage * label = cvCreateImage(cvGetSize(&img), IPL_DEPTH_LABEL, 1);
		cvLabel(&img, label, blobs);
		cvReleaseImage(&label);

		// traverse the blobs
		Stars * st = new vector<Star>();
		for (CvBlobs::const_iterator it = blobs.begin(); it != blobs.end(); ++it)
		{
			const CvBlob * b = it->second;
			st->push_back(Star(b->centroid.x, b->centroid.y, sqrt(b->area / PI)));
		}

		cout << st->size() << " stars" << endl;

		stars.push_back(st);
	}
}

int main(int argc, char ** argv)
{
	char ** end =  argv + argc;
	++argv; // skip the name of the executable

	// some default options
	Options opt;
	opt.threshold = 48;

	// get the options
	while (argv < end)
	{
		// end of options
		if (**argv != '-')
			break;

		string opt = *argv++;

		// no options will follow
		if (opt == "--")
			break;

		die("unknown option " + opt);
	}

	// get the list of images
	vector<string> imgNames;
	while (argv < end)
		imgNames.push_back(*argv++);

	// perform some sanity checks
	if (imgNames.size() < 2)
		die("no point in aligning less than two images");

	// find stars on each image
	cout << "Finding stars..." << endl;
	vector<Stars *> stars;
	getStars(imgNames, stars, opt); // allocates stars

	// free the memory
	for (vector<Stars *>::iterator it = stars.begin(); it != stars.end(); ++it)
	{
		delete *it;
	}

	return 0;
}
