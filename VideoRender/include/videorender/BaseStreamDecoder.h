#pragma once
#pragma warning(push, 0) 
#include <cstdio>
#pragma warning(pop) 

#include <videorender/Exceptions.h>
#include "ffmpeg.h"
//interface class for stream decoding
//both for audio and video
namespace videorender
{
	namespace internal
	{
		class BaseStreamDecoder
		{
		public:
			BaseStreamDecoder();
			virtual ~BaseStreamDecoder();

			virtual void decode_stream(AVFormatContext * input_stream, AVCodecContext * input_codec,
				int stream_id,float file_seek_percentage = 0) = 0;
		protected:
			static int print_error(const char* prefix, int errorCode);
		};
	}
}
