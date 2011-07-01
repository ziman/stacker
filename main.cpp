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

inline double dist(const Star & x, const Star & y)
{
	return sqrt(sqr(x.x - y.x) + sqr(x.y - y.y));
}

inline double min3(double x, double y, double z)
{
	return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

struct Triangle
{
	const Star * t, * u, * v;
	double a, b, c;
	double cfer;

	Triangle(const Star & p, const Star & q, const Star & r)
	{
		double m = dist(p,q);
		double n = dist(q,r);
		double o = dist(r,p);

		if (m < n)
		{
			if (o < m)
				{ a = o; b = m; c = n; t = &r; u = &p; v = &q; }
			else
				if (o < n)
					{ a = m; b = o; c = n; t = &q; u = &p; v = &r; }
				else
					{ a = m; b = n; c = o; t = &p; u = &q; v = &r; }
		}
		else
		{
			if (o < n)
				{ a = o; b = n; c = m; t = &p; u = &r; v = &q; }
			else
				if (o < m)
					{ a = n; b = o; c = m; t = &q; u = &r; v = &p; }
				else
					{ a = n; b = m; c = o; t = &r; u = &q; v = &p; }
		}

		cfer = m+n+o;
	}
};

double tdist(const Triangle & a, const Triangle & b)
{
	return sqr(a.a - b.a) + sqr(a.b - b.b) + sqr(a.c - b.c);
}

vector<Triangle> triangles(const Stars & stars)
{
	static const double MIN_CFER = 300;
	int N = stars.size();
	vector<Triangle> result;

	for (int i = 0; i < N; ++i)
		for (int j = i+1; j < N; ++j)
			for (int k = j+1; k < N; ++k)
			{
				Triangle t(stars[i], stars[j], stars[k]);
				if (t.cfer > MIN_CFER)
					result.push_back(t);
			}

	return result;
}

ostream & operator<<(ostream & out, const Triangle & t)
{
	return out << "[" << t.a << ", " << t.b << ", " << t.c << "]";
}

Mat getTransform(Stars & xs, Stars & ys)
{
	vector<Triangle> xt = triangles(xs);
	vector<Triangle> yt = triangles(ys);

	vector<double> xdist;
	int p = -1, q = -1;
	double maxdist = 0;
	double mindist = 1.0e10;
	for (int i = 0; i < xt.size(); ++i)
	{
		for (int j = 0; j < yt.size(); ++j)
		{
			double dist = tdist(xt[i], yt[j]);
			if (dist < mindist)
			{
				mindist = dist;
				p = i; q = j;
			}
		}
	}

	if (p == -1)
		die("No unique triangle found!");

	Point2f xp[3], yp[3];
	xp[0] = Point2f(xt[p].t->x, xt[p].t->y);
	xp[1] = Point2f(xt[p].u->x, xt[p].u->y);
	xp[2] = Point2f(xt[p].v->x, xt[p].v->y);

	yp[0] = Point2f(yt[q].t->x, yt[q].t->y);
	yp[1] = Point2f(yt[q].u->x, yt[q].u->y);
	yp[2] = Point2f(yt[q].v->x, yt[q].v->y);

	Mat trans = getAffineTransform(xp, yp);

	cout << "Most similar triangles: " << p << ":" << q
		<< ", dist = " << tdist(xt[p], yt[q]) << endl;
	cout << xt[p] << endl << yt[q] << endl;
	cout << "Transform: " << endl << trans << endl;

	return trans;
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
