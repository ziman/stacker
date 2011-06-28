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

bool operator<(const Star & l, const Star & r)
{
	if (l.r < r.r)
		return true;
	if (l.r > r.r)
		return false;

	return (l.x < r.x);
}

typedef vector<Star> Stars;

void die(const string & msg)
{
	cerr << "Error: " << msg << endl;
	exit(1);
}

inline double sqr(double x)
{
	return x*x;
}

inline double min3(double x, double y, double z)
{
	return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

#define D(m,n) dist[N*m + n]
#define C(m,n) cost[NN*m + n]
void getTransform(Stars & xs, Stars & ys)
{
	int M = xs.size();
	int N = ys.size();

	// distance table
	double distSum = 0;
	double * dist = new double[M*N];
	for (int m = 0; m < M; ++m)
		for (int n = 0; n < N; ++n)
		{
			D(m,n) = sqrt(
				  sqr(xs[m].x - ys[n].x)
				+ sqr(xs[m].y - ys[n].y)
			);
			distSum += D(m,n);
		}
	double avgDist = distSum / (M*N);

	// calculate the cost table
	int MM = M+1, NN = N+1;
	double * cost = new double[MM*NN];
	for (int m = 0; m <= M; ++m) C(m,0) = 0;
	for (int n = 0; n <= N; ++n) C(0,n) = 0;
	for (int m = 1; m <= M; ++m)
	{
		for (int n = 1; n <= N; ++n)
		{
			C(m,n) = min3(
				C(m-1, n) + 2*avgDist,
				C(m, n-1) + 2*avgDist,
				C(m-1, n-1) + D(m-1, n-1)
			);
			
			cout << C(m,n) << " ";
		}
		cout << endl;
	}
	
	// gather results
	int bestm = M, bestn = N;
	for (int m = 3; m <= M; ++m) // start from 3 -> not interested in solutions with less than two em's
		if (C(m,N) < C(bestm,bestn)) { bestm = m; bestn = N; }
	for (int n = 3; n <= N; ++n)
		if (C(M,n) < C(bestm,bestn)) { bestm = M; bestn = n; }
	
	// correlation check
	if (bestm == 0 || bestn == 0)
	{
		cerr << "No correlation!" << endl;
		return;
	}
	else
	{
		cout << "Best correlation: " << C(bestm,bestn) << endl;
	}
	
	// trace back the optimal solution
	int ms[3], ns[3]; // circular buffers
	int m = bestm; int n = bestn;
	int rnd = 0;
	while (m > 0 && n > 0)
	{
		if (C(m,n) == C(m-1,n))
			--m; // skip m
		else if (C(m,n) == C(m,n-1))
			--n; // skip n
		else
		{
			// match! -> add to circular buffer
			ms[rnd] = m-1;
			ns[rnd] = n-1;
			
			rnd = (rnd + 1) % 3;
			--m;
			--n;
		}
	}
	// in the circular buffer: first three matches
	
	cout << "We have this aligment:" << endl;
	for (int i = 0; i < 3; ++i)
	{
		cout << "  " << ms[i] << " - " << ns[i];
		cout << ", distance " << D(ms[i], ns[i]) << endl;
	}
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

	// sort each vector by star size
	for (vector<Stars *>::iterator it = stars.begin(); it != stars.end(); ++it)
	{
		sort((*it)->rbegin(), (*it)->rend());
	}
	
	// align the stars
	getTransform(*stars[0], *stars[1]);

	// free the memory
	for (vector<Stars *>::iterator it = stars.begin(); it != stars.end(); ++it)
	{
		delete *it;
	}

	return 0;
}
