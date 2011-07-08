/* Astrophoto stacker, Copyright (c) 2011, Matus Tejiscak <functor.sk@ziman>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Astrophoto Stacker nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <string>
#include <vector>
#include <list>

#include "opencv.hpp"

using namespace cv;
using namespace std;
using namespace flann;

static const double PI = 3.1415926536;

/** Commandline options. */
struct Options
{
	int threshold; ///< Threshold estimate. Not very important, used only for speedup of estimation.
	double subsample; ///< Within [0.0 .. 1.0], factor to resize loaded images to.
	float minLineLength; ///< Discard all lines shorter than this.
	float relativeLengthTolerance; ///< Maximum relative length difference between two lines to be considered identical.
	int percentStarsRequired; ///< The minimum per cent of stars matched between two images.
	float starDistCutoff; ///< Maximum px distance between two stars to be considered identical.
	int starCount; ///< Calculate with (roughly) this number of brightest stars in the images.
	string outfile; ///< Destination image file name. Leave empty to display directly.
};

/** A star in the image. */
struct Star
{
	double x, y; // position
	double r; // radius

	Star(double x, double y, double r)
	{
		this->x = x;
		this->y = y;
		this->r = r;
	}
	
	Star(const Star &s)
		: x(s.x), y(s.y), r(s.r)
	{}
};

/** A blob in the image. */
struct Blob
{
	double x, y;
	double S;
};

/// Combine two blobs.
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

typedef vector<Star> Stars;
typedef vector<Blob> Blobs;

/// Print error message and quit.
void die(const string & msg)
{
	cerr << "Error: " << msg << endl;
	exit(1);
}

inline double sqr(double x)
{
	return x*x;
}

/// Create a third point, perpendicular to UV.
Point2f controlPoint(const Point2f & u, const Point2f & v)
{
	double dx = v.x - u.x;
	double dy = v.y - u.y;

	return Point2f(u.x - dy, u.y + dx);
}

/// A line in the image.
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
	{ }
	
	/// Return this line with (a,b) swapped.
	Line swap() const
	{
		return Line(b,a);
	}
};

bool operator<(const Line & l, const Line & r)
{
	return (l.length < r.length);
}

/// Create a line from each pair of stars.
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

/// Calculate affine matrix transform between two lines.
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

/// Scan item for blob search.
struct ScanItem
{
	Blob blob;
	int l, r; // leftmost & rightmost x-index of pixels of the blob on the previous scanline

	ScanItem(int _l, int _r, const Blob & _b)
		: l(_l), r(_r), blob(_b)
	{}
};

/// Return a number representing the fitness of the given transformation. Higher is better.
double evaluate(const Mat & trans, const Stars & xs, Index_<double> & yindex, const Options & opt)
{
	// create the query matrix containing all stars from xs
	Mat q(xs.size(), 3, CV_64F);
	for (int x = 0; x < xs.size(); ++x)
	{
		q.at<double>(x,0) = xs[x].x;
		q.at<double>(x,1) = xs[x].y;
		q.at<double>(x,2) = 1;
	}
	
	// transform the query using the transformation in question
	Mat query = q * trans.t();
	
	// for each transformed star, find its nearest neighbor
	Mat indices(xs.size(), 1, CV_32S), dists(xs.size(), 1, CV_32F);
	yindex.knnSearch(query, indices, dists, 1, SearchParams());
	
	// evaluate the nearest-neighbor assignment
	int cnt = 0;
	double sum = 0;
	for (int i = 0; i < dists.rows; ++i)
	{
		float dist = dists.at<float>(i, 0);
		if (dist < opt.starDistCutoff) {
			// if the counterparts are close enough, register 'em
			++cnt;
			sum += dist;
		}
	}
	
	// if not enough stars, quit
	if (cnt < opt.percentStarsRequired * xs.size() / 100)
	{
		// cout << "Not enough stars: " << cnt << endl;
		return 0;
	}

	return opt.starDistCutoff - sum/cnt;
}

/// Get the best transformation that transforms xs closest to ys.
bool getTransform(const Stars & xs, Index_<double> & yindex, const vector<Line> & yl, Mat & bestTrans, const Options & opt)
{
	// find all lines
	vector<Line> xl;
	getLines(xs, xl);
	
	// we want at least one line
	if (xl.size() < 1 || yl.size() < 1)
	{
		cout << "No lines." << endl;
		return false;
	}
	
	// sort the lines
	sort(xl.rbegin(), xl.rend());
	
	// cout << "X stars: " << xs.size() << ", Y stars: " << ys.size() << endl;
	// cout << "X lines: " << xl.size() << ", Y lines: " << yl.size() << endl;

	bestTrans = Mat::eye(2, 3, CV_64F);
	double bestScore = 0;
	int bestOfs = 0;
	
	// for each source line
	for (int i = 0; i < xl.size(); ++i)
	{
		// consider its length
		const Line & xline = xl[i];
		double xlen = xline.length;
		
		if (xlen < opt.minLineLength)
			break; // too short to be precise enough and only shorter ones will follow
		
		// cout << "Line length: " << xlen << endl;

		// bisect -> find the nearest counterpart in yl
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

		// find upper && lower bound within the tolerance
		int estimate = lo;
		int estlo = estimate, esthi = estimate;
		double tolerance = xlen * opt.relativeLengthTolerance;
		while (estlo >= 0 && yl[estlo].length + tolerance >= xlen)
			--estlo;
		while (esthi < yl.size() && yl[esthi].length - tolerance <= xlen)
			++esthi;

		// traverse all lines within the tolerance
		for (int i = estlo+1; i < esthi; ++i)
		{
			Mat trans = getLineTransform(xline, yl[i]);
			double score = evaluate(trans, xs, yindex, opt);
			if (score > bestScore)
			{
				bestScore = score;
				bestTrans = trans;
				bestOfs = i - estimate;
			}
			
			trans = getLineTransform(xline, yl[i].swap());
			score = evaluate(trans, xs, yindex, opt);
			if (score > bestScore)
			{
				bestScore = score;
				bestTrans = trans;
				bestOfs = i - estimate;
			}
		}
	}

	if (bestScore > 0)
		cout << "OK (score " << opt.starDistCutoff-bestScore << " at offset " << bestOfs << ")" << endl;
	else
		cout << "FAIL, skipping" << endl;
	return (bestScore > 0);
};

/// Find all blobs in a thresholded image.
void findBlobs(const Mat & mat, Blobs & blobs, int limit)
{
	//cout << "depth: " << mat.depth() << ", type: " << mat.type() << ", chans: " << mat.channels() << endl;
	
	// traverse all scanlines
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
				
				if (blobs.size() > limit)
				{
					// this sample will be rejected anyway -> save CO2
					return;
				}
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
				
				if (newscan.size() > limit)
				{
					// this threshold will be rejected anyway
					return;
				}
			}

			l = ++x;
		}
		scan = newscan;
		newscan.clear();
	}

	// flush all unfinished blobs
	for (list<ScanItem>::const_iterator it = scan.begin(); it != scan.end(); ++it)

	{
		blobs.push_back(it->blob);
	}
}

/// Find all stars in the given image.
void findStars(const Mat & srcimg, Stars & stars, int thresh, int limit)
{
	// threshold the image
	Mat image;
	threshold(srcimg, image, thresh, 255, THRESH_BINARY);

	// find the blobs
	Blobs blobs;
	findBlobs(image, blobs, limit);

	// traverse the blobs
	stars.clear();
	for (Blobs::const_iterator it = blobs.begin(); it != blobs.end(); ++it)
	{
		stars.push_back(Star(it->x, it->y, sqrt(it->S / PI)));
	}
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
	FOREACH(*row = *row ? lround(31 * log2(*row)) : 0);
}

/// Find all stars using an adaptive threshold.
void findStarsThresh(const Mat & srcimg, Stars & stars, Options & opt)
{
	// threshold estimate
	int oldThresh = (opt.threshold == -1) ? 128 : opt.threshold;
	
	// binary search bounds
	int lo = 0;
	int hi = 255;
	if (oldThresh < 128)
		hi = 2*oldThresh;
	else
		lo = 2*oldThresh - 256;
	
	int thresh = oldThresh;
	while (lo+1 < hi) {
		// estimate threshold
		stars.clear();
		int cnt = 0;
		thresh = (hi+lo) / 2;
		
		// calculate the number of stars
		findStars(srcimg, stars, thresh, 2 * opt.starCount);
		cnt = stars.size();
		
		// roughly required number -> done
		if (abs(cnt - opt.starCount) < opt.starCount/5)
		{
			// cout << "thresholding by " << thresh << endl;
			opt.threshold = thresh;
			return;
		}
		// adjust the threshold
		else if (cnt < opt.starCount)
			hi = thresh;
		else
			lo = thresh;
		
		//cout << "Threshold auto-estimated at " << oldThresh << endl;
	}

	//cout << "thresholding by " << thresh << endl;
	opt.threshold = thresh;
}

Mat load(const string & fn, const Options & opt)
{
	cout << fn << " ... ";
	cout.flush();
	Mat full = imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
	Mat subsampled;
	resize(full, subsampled, Size(0,0), opt.subsample, opt.subsample);
	return subsampled;
}

Mat floatify(const Mat & img)
{
	Mat result;
	img.convertTo(result, CV_32F, 1/255.0, 0);
	return result;
}

Mat merge(const vector<string> & fn, Options & opt)
{
	// load the middle image
	int mid = fn.size() / 2;
	Mat mimg = load(fn[mid], opt);
	
	// convert to float
	Mat merged = floatify(mimg);
	
	// find its stars
	Stars mstars;
	normalize(mimg);
	findStarsThresh(mimg, mstars, opt);
	
	// precompute NN search index
	Mat ymat(mstars.size(), 2, CV_64F);
	for (int y = 0; y < mstars.size(); ++y)
	{
		ymat.at<double>(y, 0) = mstars[y].x;
		ymat.at<double>(y, 1) = mstars[y].y;
	}
	Index_<double> yindex(ymat, KDTreeIndexParams(4));

	// precompute line list
	vector<Line> ylines;
	getLines(mstars, ylines);
	sort(ylines.begin(), ylines.end());
	
	cout << "preprocessed." << endl;
	
	double n = 0;
	for (int i = 0; i < fn.size(); ++i)
	{
		if (i == mid) continue;
		
		// load the image
		Mat inorm = load(fn[i], opt);
		Mat img = floatify(inorm);
		normalize(inorm);
		
		// find stars
		Stars stars;
		findStarsThresh(inorm, stars, opt);
		
		// calculate the transformation
		Mat trans;
		bool ret = getTransform(stars, yindex, ylines, trans, opt);
		if (!ret)
			continue;
		
		// remap the original image
		Mat lremap;
		warpAffine(img, lremap, trans, mimg.size());
		
		// merge the images
		n += 1;
		merged = (1 - 1/n) * merged + (1/n) * lremap;
	}
	
	return merged;
}

/// Print usage information and quit.
void usage()
{
	cout
		<< "usage: ./align [options] image1 image2 ... imagen" << endl
		<< endl
		<< "Options:" << endl
		<< "  -s <factor>  : scale the image by the given factor (default = 0.5)" << endl
		<< "  -l <length>  : minimum line length to be taken in account (default = 100)" << endl
		<< "  -p <percent> : portion of stars required to match between images (default = 66)" << endl
		<< "  -t <factor>  : maximum relative length error between two matching lines (default = 0.01)" << endl
		<< "  -d <pixels>  : maximum distance between two matching stars (default = 10)" << endl
		<< "  -c <count>   : approximate target star count after thresholding (default = 20)" << endl
		<< "  -o <imgname> : write the result here, instead of displaying it" << endl
		;
	exit(1);
}

int main(int argc, char ** argv)
{
	char ** end =  argv + argc;
	++argv; // skip the name of the executable

	// some default options
	Options opt;
	opt.threshold = -1; // autodetect
	opt.subsample = 0.5;
	opt.minLineLength = 100;
	opt.percentStarsRequired = 66;
	opt.relativeLengthTolerance = 0.01;
	opt.starDistCutoff = 10;
	opt.starCount = 20;
	opt.outfile = "";

	// get the options
	while (argv < end)
	{
		// end of options
		if (**argv != '-')
			break;

		string o = *argv++;

		// no options will follow
		if (o == "--")
			break;
		
		if (o == "-s")
			opt.subsample = atof(*argv++);
		else if (o == "-l")
			opt.minLineLength = atof(*argv++);
		else if (o == "-p")
			opt.percentStarsRequired = atoi(*argv++);
		else if (o == "-t")
			opt.relativeLengthTolerance = atof(*argv++);
		else if (o == "-d")
			opt.starDistCutoff = atof(*argv++);
		else if (o == "-c")
			opt.starCount = atoi(*argv++);
		else if (o == "-o")
			opt.outfile = *argv++;
		else 
			usage();
	}

	// get the list of images
	vector<string> imgNames;
	while (argv < end)
		imgNames.push_back(*argv++);

	// perform some sanity checks
	if (imgNames.size() < 2)
		die("no point in aligning less than two images");

	// stack the images
	Mat stack = merge(imgNames, opt);

	if (opt.outfile == "")
	{
		// show the image
		namedWindow("preview");
		imshow("preview", stack);
		waitKey(0);
	}
	else
	{
		// save the image
		imwrite(opt.outfile, stack);
		cout << "Image saved to " << opt.outfile << endl;
	}

	return 0;
}
