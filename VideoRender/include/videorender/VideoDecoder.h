#pragma once
#pragma warning(push, 0) 
#include <string>
#include <string_view>
#include "ffmpeg.h"
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#pragma warning(pop) 

#include "StreamDecodersManager.h"
#include "BaseStreamsSelector.h"
#include <videorender/Exceptions.h>
#include "Deleters.h"
namespace videorender
{
	namespace internal
	{
		class AVInitSingleton {
		public:
			static AVInitSingleton& Instance() {
				static AVInitSingleton S;
				return S;
			}

		private:
			AVInitSingleton() { av_register_all(); }
			~AVInitSingleton() {};
		};

		class VideoDecoder
		{
		public:
			VideoDecoder(StreamDecodersManager& manager, BaseStreamsSelector& streams_selector);
			VideoDecoder(StreamDecodersManager& manager, BaseStreamsSelector& streams_selector,
				const std::string_view filename);
			void open(const std::string_view filename,float file_seek_percentage = 0);
			~VideoDecoder() = default;

		private:
			//open file and create contexts
			void parse_video_from_file(const std::string_view filename,float file_seek_percentage);

			//create scaler and Frame structure
			void init_scaler();
			void decode_streams(float file_seek_percentage);
			void decode_video_stream(int stream_id, float file_seek_percentage);

			//init stuff (should be created once per program start)
			AVInitSingleton& _av_init;

			std::unique_ptr<AVFormatContext, AVFormatContextDeleter> _inctx;

			std::unique_ptr<SwsContext, SwsContextDeleter> _swsctx;
			std::unique_ptr<AVFrame, AVFrameDeleter> _frame;
			std::vector<uint8_t> _framebuf;

			//non-owned pointer
			AVStream* _vstrm;

			StreamDecodersManager& _stream_decoders_manager;
			BaseStreamsSelector& _streams_selector;
		};
	}
}
