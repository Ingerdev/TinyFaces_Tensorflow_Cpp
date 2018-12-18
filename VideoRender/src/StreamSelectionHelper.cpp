#pragma warning(push, 0) 
#pragma warning(pop) 
#include "videorender/StreamsSelectionHelper.h"
#include "videorender/Exceptions.h"


namespace videorender
{
	namespace internal
	{
		const std::vector<int> StreamsSelectionHelper::GetAllStreamsOfType(const AVFormatContext * input_context, AVMediaType requested_type)
		{
			return GetStreamsOfType(input_context, requested_type, -1);
		}
		const std::vector<int> StreamsSelectionHelper::GetStreamsOfType(const AVFormatContext * input_context, AVMediaType requested_type, int requested_count)
		{
			std::vector<int> streams;
			int count = requested_count;
			for (int i = 0; i < static_cast<int>(input_context->nb_streams); i++)
			{
				if (input_context->streams[i]->codecpar->codec_type == requested_type)
				{
					streams.push_back(i);

					//if requested_count streams was found - exit loop.
					//note if requested_count less than 0 then loop wont break until enumerate every stream.
					//we use it to get all streams of requested type
					count--;
					if (count == 0)
						break;
				}
			}
			return streams;
		}

		const std::vector<int> StreamsSelectionHelper::GetBestStreamOfType(const AVFormatContext * input_context, AVMediaType requested_type)
		{
			std::vector<int> results;
			int stream = av_find_best_stream(const_cast<AVFormatContext *>(input_context), requested_type, -1, -1, NULL, 0);
			if (stream >= 0)
				results.push_back(stream);
			else
				if (stream == AVERROR_STREAM_NOT_FOUND)
					throw VideoDecoderException(VideoDecoderError::CannotFindBestStream);
				else
					if (stream == AVERROR_DECODER_NOT_FOUND)
						throw VideoDecoderException(VideoDecoderError::CannotFindProperCodec);
					else
						throw VideoDecoderException(VideoDecoderError::UnknownError);

			return results;
		}

		StreamsSelectionHelper::StreamsSelectionHelper()
		{
		}


		StreamsSelectionHelper::~StreamsSelectionHelper()
		{
		}
	}
}