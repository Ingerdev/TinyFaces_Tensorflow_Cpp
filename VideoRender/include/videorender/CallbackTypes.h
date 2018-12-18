#pragma once
#pragma warning(push, 0) 
#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma warning(pop) 
#include "Prediction.h"

namespace videorender
{
	using ImageCallback = std::function<void(cv::Mat)>;
	using FacesCallback = std::function<void(std::vector<Prediction>)>;
}
