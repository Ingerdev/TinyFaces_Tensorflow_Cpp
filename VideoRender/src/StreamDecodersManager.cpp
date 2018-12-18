#pragma warning(push, 0) 
#pragma warning(pop) 
#include "videorender/StreamDecodersManager.h"
namespace videorender
{
	namespace internal
	{
		StreamDecodersManager::StreamDecodersManager()
		{
		}

		StreamDecodersManager::~StreamDecodersManager()
		{
		}
		void StreamDecodersManager::registerDecoder(BaseStreamDecoder * stream_decoder, AVMediaType decoder_type)
		{
			_decoders[decoder_type] = stream_decoder;
		}
		void StreamDecodersManager::decode_stream(AVFormatContext * input_stream, AVCodecContext * input_codec, 
			int stream_id, AVMediaType stream_type,float file_seek_percentage)
		{
			if (_decoders.find(stream_type) != _decoders.end())
				_decoders[stream_type]->decode_stream(input_stream, input_codec, stream_id, file_seek_percentage);
		}
		void StreamDecodersManager::clear_decoders_list()
		{
			_decoders.clear();
		}
		void StreamDecodersManager::unregister_decoder(AVMediaType decoder_type)
		{
			if (_decoders.find(decoder_type) != _decoders.end())
				_decoders.erase(decoder_type);
		}
	}
}