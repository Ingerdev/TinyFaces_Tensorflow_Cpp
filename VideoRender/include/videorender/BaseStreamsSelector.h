#pragma once
#pragma warning(push, 0) 
#include <vector>
#pragma warning(pop) 
#include "ffmpeg.h"

namespace videorender
{
	namespace internal
	{
		class BaseStreamsSelector
		{
		public:
			virtual const std::vector<int> SelectStreams(const AVFormatContext* input_context) = 0;

			BaseStreamsSelector();
			virtual ~BaseStreamsSelector();
		};
	}
}
