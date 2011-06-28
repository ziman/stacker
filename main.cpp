#include <iostream>
#include <string>
#include <vector>

#include "cv.h"
#include "cvaux.hpp"
#include "cv.hpp"
#include "cxcore.hpp"
#include "cxtypes.h"
#include "cxmat.hpp"
#include "highgui.h"

using namespace cv;
using namespace std;

struct StarOptions
{
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

void getStars(const vector<string> & fn, vector<vector<KeyPoint> *> & stars, const StarOptions & opt)
{
	namedWindow("Preview", CV_WINDOW_AUTOSIZE);
	stars.clear();
	for (vector<string>::const_iterator it = fn.begin(); it != fn.end(); ++it)
	{
		cout << "  * " << *it << ": ";

		// load the image
		Mat image = imread(*it, CV_LOAD_IMAGE_GRAYSCALE);

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
		drawKeypoints(image, *pts, image, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);
		imshow("Preview", image);
		waitKey(1); // gui event loop
	}
}

int main(int argc, char ** argv)
{
	char ** end =  argv + argc;
	++argv; // skip the name of the executable

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

	StarOptions sopt =
	{
		/* .maxSize = */ 32,
		/* .responseThreshold = */ 20,
		/* .lineThresholdProjected = */ 6,
		/* .lineThresholdBinarized = */ 5,
		/* .suppressNonmaxSize = */ 0
	};

	vector<vector<KeyPoint> *> stars;
	getStars(imgNames, stars, sopt); // allocates stars

	return 0;
}
