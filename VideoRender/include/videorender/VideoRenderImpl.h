#pragma once
#include <string_view>
#include "SignalTypes.h"
namespace videorender
{
	namespace internal
	{
		class StreamDecodersManager;
		class StreamVideoDecoder;
		class SingleVideoStreamSelector;
		class VideoDecoder;

		class VideoRenderImpl
		{
		public:
			VideoRenderImpl();
			~VideoRenderImpl();
			void start_render(std::string_view input_name,float file_seek_percentage);
			void register_image_callback(frame_callback_t frame_callback);

		private:
			std::unique_ptr<StreamDecodersManager> _decoder_mgr;
			std::unique_ptr<StreamVideoDecoder> _stream_video_decoder;
			std::unique_ptr<SingleVideoStreamSelector> _video_stream_selector;
			std::unique_ptr<VideoDecoder> _decoder;
		};
	}
}
