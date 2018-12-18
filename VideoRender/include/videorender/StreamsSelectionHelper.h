#pragma once
#pragma warning(push, 0) 
#include <vector>
#pragma warning(pop) 

#include "ffmpeg.h"
namespace videorender
{
	namespace internal
	{

		class StreamsSelectionHelper
		{
		public:
			static const std::vector<int> GetAllStreamsOfType(const AVFormatContext* input_context, AVMediaType requested_type);
			static const std::vector<int> GetStreamsOfType(const AVFormatContext* input_context, AVMediaType requested_type, int requested_count);
			static const std::vector<int> GetBestStreamOfType(const AVFormatContext* input_context, AVMediaType requested_type);
		private:
			StreamsSelectionHelper();
			~StreamsSelectionHelper();
		};

	}
}
