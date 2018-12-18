#pragma warning(push, 0) 
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#pragma warning(pop) 
#include "videorender/RenderToDisplay.h"



namespace videorender
{  
	RenderToDisplay::RenderToDisplay()
	{
		std::cout << "RenderToDisplay" << "\n";
		cv::namedWindow(window_name_);
		cv::moveWindow(window_name_, 20, 20);
		
	}

	void RenderToDisplay::process_faces(const std::vector<Prediction>& faces)
	{
		predictions_ = faces;
	}


	void RenderToDisplay::process_frame(const cv::Mat & frame_image) const
	{
		//copy image and draw predictions on it
		auto image = frame_image.clone();
		for (auto& prediction : predictions_)
		{
			draw_face_rect(image, prediction);
		}
		
		cv::imshow(window_name_, image);
		
		//necessary for opencv inner loop
		cvWaitKey(1);					
	}

	void RenderToDisplay::draw_face_rect(cv::Mat& frame_image, const Prediction& prediction) const
	{	
		cv::rectangle(frame_image, 
			cv::Rect(prediction.left, prediction.top, 
					 prediction.width(),prediction.height()), cv::Scalar(255,255,255));
	}

}