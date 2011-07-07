#include <iostream>
#include <string>
#include <vector>
#include <list>

#include "opencv.hpp"

using namespace cv;
using namespace std;
using namespace flann;

static const double PI = 3.1415926536;

struct Options
{
	int threshold;
	double subsample; // 0..1 -> size on load
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
	
	Star(const Star &s)
		: x(s.x), y(s.y), r(s.r)
	{
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

Point2f controlPoint(const Point2f & u, const Point2f & v)
{
	double dx = v.x - u.x;
	double dy = v.y - u.y;

	return Point2f(u.x - dy, u.y + dx);
}

struct Line
{
	Star a, b;
	double length;

	Line(const Star & a_, const Star & b_)
		: a(a_), b(b_)
	{
		length = sqrt(sqr(a.x-b.x) + sqr(a.y - b.y));
	}
	
	Line(const Line& x)
		: a(x.a), b(x.b), length(x.length)
	{
	}
};

bool operator<(const Line & l, const Line & r)
{
	return (l.length < r.length);
}

void getLines(const Stars & stars, vector<Line> & lines)
{
	for (int i = 0; i < stars.size(); ++i)
	{
		for (int j = i+1; j < stars.size(); ++j)
		{
			lines.push_back(Line(stars[i], stars[j]));
		}
	}
}

Mat getLineTransform(const Line & a, const Line & b)
{
	Point2f xp[3], yp[3];
	xp[0] = Point2f(a.a.x, a.a.y);
	xp[1] = Point2f(a.b.x, a.b.y);
	xp[2] = controlPoint(xp[0], xp[1]);

	yp[0] = Point2f(b.a.x, b.a.y);
	yp[1] = Point2f(b.b.x, b.b.y);
	yp[2] = controlPoint(yp[0], yp[1]);

	return getAffineTransform(xp, yp);
}

struct ScanItem
{
	Blob blob;
	int l, r;

	ScanItem(int _l, int _r, const Blob & _b)
		: l(_l), r(_r), blob(_b)
	{}
};

double evaluate(const Mat & trans, const Stars & xs, Index_<double> & yindex)
{
	Mat q(3, xs.size(), CV_64F);
	for (int x = 0; x < xs.size(); ++x)
	{
		q.at<double>(0, x) = xs[x].x;
		q.at<double>(1, x) = xs[x].y;
		q.at<double>(2, x) = 1;
	}
	Mat query = trans * q;
	
	Mat indices(xs.size(), 1, CV_32S), dists(xs.size(), 1, CV_32F);
	yindex.knnSearch(query.t(), indices, dists, 1, SearchParams());
	
	int cnt = 0;
	double sum = 0;
	float CLOSE_ENOUGH = 100;
	int ENOUGH_STARS = 2 * xs.size() / 3;
	for (int i = 0; i < dists.rows; ++i)
	{
		float dist = dists.at<float>(i, 0);
		if (dist < CLOSE_ENOUGH) {
			++cnt;
			sum += dist;
		}
	}
	
	if (cnt < ENOUGH_STARS)
		return 0;

	return CLOSE_ENOUGH - sum/cnt;
}

bool getTransform(const Stars & xs, const Stars & ys, Mat & bestTrans)
{
	static const double LENGTH_TOLERANCE = 5;
	
	// precompute NN search index
	Mat ymat(ys.size(), 2, CV_64F);
	for (int y = 0; y < ys.size(); ++y)
	{
		ymat.at<double>(y, 0) = ys[y].x;
		ymat.at<double>(y, 1) = ys[y].y;
	}
	Index_<double> yindex(ymat, AutotunedIndexParams());

	// find all lines
	vector<Line> xl, yl;
	getLines(xs, xl);
	getLines(ys, yl);
	
	// sort the lines
	sort(xl.rbegin(), xl.rend());
	sort(yl.begin(), yl.end());
	
	// cout << "X lines: " << xl.size() << ", Y lines: " << yl.size() << endl;

	bestTrans = Mat::eye(2, 3, CV_64F);
	double bestScore = 0;
	
	for (int i = 0; i < xl.size(); ++i)
	{
		const Line & xline = xl[i];
		double xlen = xline.length;
		
		// allow up to 5% tolerance
		if (xlen < LENGTH_TOLERANCE * 20)
			break;

		// bisect -> find estimate
		int lo = 0;
		int hi = yl.size() - 1;
		while (lo < hi)
		{
			int mid = (lo + hi) / 2;
			if (xlen < yl[mid].length)
				hi = mid;
			else
				lo = mid+1;
		}

		// find upper && lower bound
		int estimate = lo;
		int estlo = estimate, esthi = estimate;
		while (estlo >= 0 && yl[estlo].length + LENGTH_TOLERANCE >= xlen)
			--estlo;
		while (esthi < yl.size() && yl[esthi].length - LENGTH_TOLERANCE <= xlen)
			++esthi;

		hi = lo+1;
		while (lo > estlo || hi < esthi)
		{
			if (lo > estlo)
			{
				Mat trans = getLineTransform(xline, yl[lo]);
				double score = evaluate(trans, xs, yindex);
				if (score > bestScore)
				{
					bestScore = score;
					bestTrans = trans;
				}
			}
			
			if (hi < esthi)
			{
				Mat trans = getLineTransform(xline, yl[hi]);
				double score = evaluate(trans, xs, yindex);
				if (score > bestScore)
				{
					bestScore = score;
					bestTrans = trans;
				}
			}
			
			--lo;
			++hi;
		}
	}

	return (bestScore > 0);
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

void findStars(const Mat & srcimg, Stars & stars, int thresh)
{
	// threshold the image
	Mat image;
	threshold(srcimg, image, thresh, 255, THRESH_BINARY);

	// find the blobs
	Blobs blobs;
	findBlobs(image, blobs);

	// traverse the blobs
	stars.clear();
	for (Blobs::const_iterator it = blobs.begin(); it != blobs.end(); ++it)
	{
		stars.push_back(Star(it->x, it->y, sqrt(it->S / PI)));
	}
}

inline uint8_t clamp(int x)
{
	return (x < 0) ? 0 : ((x > 255) ? 255 : x);
}

#define FOREACH(st) \
	for (int y = 0; y < mat.rows; ++y) 			\
	{							\
		uint8_t * row = mat.ptr<uint8_t>(y);	\
		const uint8_t * end = row + mat.cols;		\
		while (row < end) {				\
			st;				\
			++row;					\
		}						\
	}
void normalize(Mat & mat)
{
	int sum = 0;
	FOREACH(sum += *row);

	int N = mat.rows * mat.cols;
	int avg = sum / N;
	int sqdiff = 0;
	FOREACH(sqdiff += (avg-*row) * (avg-*row));
	
	int var = sqdiff / N;
	int sigma = lround(sqrt(var));
	
	FOREACH(*row = clamp(255 * ((int) *row - (int) avg) / (16 * sigma)));
}

Mat merge(const vector<string> & fn, int a, int b, const Options & opt)
{
	if (a+1 >= b)
	{
		cout << "Loading " << fn[a] << endl;
		Mat full = imread(fn[a], CV_LOAD_IMAGE_GRAYSCALE);
		Mat subsampled;
		resize(full, subsampled, Size(0,0), opt.subsample, opt.subsample);
		normalize(subsampled);
		return subsampled;
	}

	// merge recursively
	int mid = (a + b) / 2;
	Mat l = merge(fn, a, mid, opt);
	Mat r = merge(fn, mid, b, opt);

	// align the images
	Stars lstars, rstars;
	findStars(l, lstars, opt.threshold);
	findStars(r, rstars, opt.threshold);
	Mat trans;
	bool ret = getTransform(lstars, rstars, trans);
	if (!ret) {
		cout << "Could not align images!" << endl;
		return l; // no transform could be found -> return (arbitrarily) the left child
	}

	// remap
	Mat lremap;
	warpAffine(l, lremap, trans, r.size());
	
	// merge
	return (0.5*lremap + 0.5*r);
}

int main(int argc, char ** argv)
{
	char ** end =  argv + argc;
	++argv; // skip the name of the executable

	// some default options
	Options opt;
	opt.threshold = 128;
	opt.subsample = 0.3;

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

	// stack the images
	Mat stack = merge(imgNames, 0, imgNames.size(), opt);

	namedWindow("preview");
	imshow("preview", stack);
	waitKey(0);

	return 0;
}
