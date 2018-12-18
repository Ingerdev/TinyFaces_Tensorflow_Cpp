// FaceDetect.cpp : Defines the entry point for the application.
//
#pragma once
//#include "caf/all.hpp"

#include "app/FaceDetect.h"
#include "videorender/MainWrapper.h"
#include "app/Communicator.h"
//#include "app/fiber_test.hpp"


using namespace std;

bool check_parameter_validness(int argc, char* argv[], float& seek_file_percentage, float& faceprob_lower_limit)
{
	//we should have 1-3 parameters, no more.
	if ((argc < 2) || (argc > 4))
	{
		return false;
	};

	//validate seek parameter
	if (argc >2)
	{
		try
		{
			seek_file_percentage = std::stof(argv[2]);
			if ((seek_file_percentage < 0) || (seek_file_percentage > 1))
				return false;
		}
		catch (...)
		{
			return false;
		}
	}

	//validate face probability parameter
	if (argc > 3)
	{
		try
		{
			faceprob_lower_limit = std::stof(argv[3]);
			if ((faceprob_lower_limit < 0) || (faceprob_lower_limit > 1))
				return false;
		}
		catch (...)
		{
			return false;
		}
	}
	return true;
}

void show_usage()
{
	std::cout << "Tensorflow c++ face finder on video files\n";
	std::cout << "Usage: face_detect.exe <path to video file> [seek file percentage] [face probability lowerlimit]\n";
	std::cout << "[seek file percentage] - optional parameter [0..1] will seek file to defined position.\n";
	std::cout << "[face probability lowerlimit] - optional parameter [0..1] which defines minimum probability "
			  << "of detected face to be shown. Default value is 0.2 \n";
	std::cout << "example:\"face_detect.exe c:\\video.avi 0.5\" will open file and seek it to the middle length position\n";
	std::cout << "example:\"face_detect.exe c:\\video.avi 0 0.5\" will open file without seek and use 50% face probability limit\n";
	std::cout << "press Ctrl^C to exit program\n";
}


int main(int argc,char* argv[])
{	
	float seek_file_percentage = 0.0f;
	float faceprob_lower_limit = 0.2f;
	if (!check_parameter_validness(argc, argv, seek_file_percentage, faceprob_lower_limit))
	{
		show_usage();
		return -1;
	}

	app::Communicator comm(argv[1], seek_file_percentage,faceprob_lower_limit);
	return 0;
}

