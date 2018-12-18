#pragma once
#include "videorender/MainWrapper.h"
#include "videorender/VideoRender.h"

MainWrapper::MainWrapper(const std::string& file_name)
{
	videorender::VideoRender render;	
	render.start_render(file_name);
}
