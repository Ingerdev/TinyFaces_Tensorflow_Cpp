#pragma once
#pragma warning(push, 0) 
#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma warning(pop) 

namespace videorender
{
	using frame_callback_t = std::function<bool(const cv::Mat)>;
}
