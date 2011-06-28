#include <iostream>
#include <string>
#include <vector>

#include "opencv.hpp"
#include "cvblob.h"

using namespace cv;
using namespace std;

struct Options
{
	int threshold;

	// StarDetector options
	int maxSize;
	int responseThreshold;
	int lineThresholdProjected;
	int lineThresholdBinarized;
	int suppressNonmaxSize;
};

void die(const string & msg)
{
	cerr << "Error: " << msg << endl;
	exit(1);
}

void getStars(const vector<string> & fn, vector<vector<KeyPoint> *> & stars, const Options & opt)
{
	namedWindow("Preview", CV_WINDOW_AUTOSIZE);
	stars.clear();
	for (vector<string>::const_iterator it = fn.begin(); it != fn.end(); ++it)
	{
		cout << "  * " << *it << ": ";

		// load the image
		Mat srcimg = imread(*it, CV_LOAD_IMAGE_GRAYSCALE);
		Mat image;
		threshold(srcimg, image, opt.threshold, 255, THRESH_TOZERO);

		// find the stars
		vector<KeyPoint> * pts = new vector<KeyPoint>();
		StarDetector(
			opt.maxSize, opt.responseThreshold, opt.lineThresholdProjected,
			opt.lineThresholdBinarized, opt.suppressNonmaxSize
		)(image, *pts);

		// push the vector
		stars.push_back(pts);
		cout << pts->size() << " stars." << endl;

		// show the image
		//drawKeypoints(image, *pts, image, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);
		imshow("Preview", image);
		waitKey(1); // gui event loop
	}
}

int main(int argc, char ** argv)
{
	char ** end =  argv + argc;
	++argv; // skip the name of the executable

	// some default options
	Options opt;
	opt.maxSize = 32;
	opt.responseThreshold = 10;
	opt.lineThresholdProjected = 8;
	opt.lineThresholdBinarized = 8;
	opt.suppressNonmaxSize = 0;
	opt.threshold = 32;

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
	vector<vector<KeyPoint> *> stars;
	getStars(imgNames, stars, opt); // allocates stars

	// free the memory
	for (vector<vector<KeyPoint> *>::iterator it = stars.begin(); it != stars.end(); ++it)
	{
		delete *it;
	}

	return 0;
}
