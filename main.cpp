#include <iostream>
#include <string>
#include <vector>

#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
	char ** end =  argv + argc;
	while (argv < end)
	{
		// not an option
		if (**argv != '-')
			break;

		string opt = *argv++;

		// no options will follow
		if (opt == "--")
			break;
	}

	vector<Mat> images(end - argv);
	while (argv < end)
	{
		cout << "Loading " << *argv << "..." << endl;
		images.push_back(imread(*argv++, -1));
	}

	return 0;
}
