#include "videorender/VideoRenderImpl.h"
#include <memory>
#include <iostream>

#include "videorender/VideoDecoder.h"
#include "videorender/StreamDecodersManager.h"
#include "videorender/SingleVideoStreamSelector.h"
#include "videorender/StreamVideoDecoder.h"



namespace videorender
{
	namespace internal
	{
		VideoRenderImpl::VideoRenderImpl() :
			_decoder_mgr( new StreamDecodersManager),
			_stream_video_decoder( new StreamVideoDecoder),
			_video_stream_selector(	new SingleVideoStreamSelector),
			_decoder( new VideoDecoder(*_decoder_mgr.get(), *_video_stream_selector.get()))		

		{
			_decoder_mgr.get()->registerDecoder(_stream_video_decoder.get(), AVMediaType::AVMEDIA_TYPE_VIDEO);

		}

		//workaround for issue
		//https://stackoverflow.com/questions/9954518/stdunique-ptr-with-an-incomplete-type-wont-compile
		VideoRenderImpl::~VideoRenderImpl() = default;
		

		void VideoRenderImpl::register_image_callback(frame_callback_t frame_callback)
		{
			_stream_video_decoder.get()->register_image_callback(frame_callback);
		}

		void VideoRenderImpl::start_render(std::string_view input_name,float file_seek_percentage)
		{			
			_decoder.get()->open(input_name, file_seek_percentage);
		}
	}
}