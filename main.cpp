#include <iostream>
#include <string>
#include <vector>
#include <list>

#include "opencv.hpp"

using namespace cv;
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

struct Blob
{
	double x, y;
	double S;
};

Blob operator+(const Blob & l, const Blob & r)
{
	Blob q;
	q.S = l.S + r.S;
	q.x = (l.S*l.x + r.S*r.x) / q.S;
	q.y = (l.S*l.y + r.S*r.y) / q.S;
}

Blob & operator+=(Blob & blob, const Blob & x)
{
	double S = blob.S + x.S;
	blob.x = (blob.S*blob.x + x.S*x.x) / S;
	blob.y = (blob.S*blob.y + x.S*x.y) / S;
	blob.S = S;
}

bool operator<(const Star & l, const Star & r)
{
	if (l.r < r.r)
		return true;
	if (l.r > r.r)
		return false;

	return (l.x < r.x);
}

typedef vector<Star> Stars;
typedef vector<Blob> Blobs;

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

void getTransform(Stars & xs, Stars & ys)
{
}

struct ScanItem
{
	Blob blob;
	int l, r;

	ScanItem(int _l, int _r, const Blob & _b)
		: l(_l), r(_r), blob(_b)
	{}
};

void findBlobs(const Mat & mat, Blobs & blobs)
{
	/*
	namedWindow("foo");
	imshow("foo", mat);
	waitKey(1000);
	*/

	//cout << "depth: " << mat.depth() << ", type: " << mat.type() << ", chans: " << mat.channels() << endl;
	list<ScanItem> scan, newscan;
	for (int y = 0; y < mat.rows; ++y)
	{
		const uint8_t * row = mat.ptr<uint8_t>(y);
		list<ScanItem>::iterator it = scan.begin();
		int l = 0;
		for (int x = 0; x < mat.cols; ++x)
		{
			// skip blanks
			while (it != scan.end() && it->r < x)
			{
				blobs.push_back(it->blob);
				++it;
			}

			// find the end of the white segment
			while (x < mat.cols && row[x])
				++x;

			// if white segment found
			if (row[l])
			{
				//cout << "rowscan at " << l << ".." << x << "," << y << endl;
				Blob cur = {(x+l-1)/2.0, y, x-l};
				while (it != scan.end() && it->l < x)
				{
					cur += it->blob;
					++it;
				}
				newscan.push_back(ScanItem(l,x-1,cur));
			}

			l = ++x;
		}
		scan = newscan;
		newscan.clear();
	}

	for (list<ScanItem>::const_iterator it = scan.begin(); it != scan.end(); ++it)

	{
		blobs.push_back(it->blob);
	}
}

void getStars(const vector<string> & fn, vector<Stars *> & stars, const Options & opt)
{
	stars.clear();
	for (vector<string>::const_iterator it = fn.begin(); it != fn.end(); ++it)
	{
		cout << "  * " << *it << ": ";

		// load the image
		Mat image = imread(*it, CV_LOAD_IMAGE_GRAYSCALE);
		Mat srcimg;
		resize(image, srcimg, Size(0,0), 0.25, 0.25);
		threshold(srcimg, image, opt.threshold, 255, THRESH_BINARY);

		// find the blobs
		Blobs blobs;
		findBlobs(image, blobs);

		// traverse the blobs
		Stars * st = new vector<Star>();
		for (Blobs::const_iterator it = blobs.begin(); it != blobs.end(); ++it)
		{
			st->push_back(Star(it->x, it->y, sqrt(it->S / PI)));
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
