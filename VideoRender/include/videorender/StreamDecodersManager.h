#pragma once
#pragma warning(push, 0) 
#include <map>
#pragma warning(pop) 

#include "ffmpeg.h"
#include "BaseStreamDecoder.h"
namespace videorender
{
	namespace internal
	{
		class StreamDecodersManager
		{
		public:
			StreamDecodersManager();
			~StreamDecodersManager();
			void registerDecoder(BaseStreamDecoder* stream_decoder, AVMediaType decoder_type);
			void decode_stream(AVFormatContext * input_stream, AVCodecContext * input_codec, int stream_id,
				AVMediaType stream_type,float file_seek_percentage);
			void clear_decoders_list();
			void unregister_decoder(AVMediaType decoder_type);
		private:
			std::map<AVMediaType, BaseStreamDecoder*> _decoders;
		};
	}
}
