#include <iostream>
#include <string>
#include <vector>

#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

void die(const string & msg)
{
	cerr << "Error: " << msg << endl;
	exit(1);
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

	return 0;
}
