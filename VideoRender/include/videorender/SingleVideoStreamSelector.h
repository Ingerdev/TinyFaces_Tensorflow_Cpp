#pragma once
#pragma warning(push, 0) 
#pragma warning(pop) 
#include "BaseStreamsSelector.h"
#include "StreamsSelectionHelper.h"
namespace videorender
{
	namespace internal
	{
		class SingleVideoStreamSelector :
			public BaseStreamsSelector
		{
		public:
			virtual const std::vector<int> SelectStreams(const AVFormatContext* input_context);

			SingleVideoStreamSelector();
			virtual ~SingleVideoStreamSelector();
		};

	}
}