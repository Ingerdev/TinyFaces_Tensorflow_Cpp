#pragma warning(push, 0) 
#pragma warning(pop) 
#include "videorender/SingleVideoStreamSelector.h"

namespace videorender
{
	namespace internal
	{

		const std::vector<int> SingleVideoStreamSelector::SelectStreams(const AVFormatContext * input_context)
		{
			return StreamsSelectionHelper::GetBestStreamOfType(input_context, AVMediaType::AVMEDIA_TYPE_VIDEO);
		}

		SingleVideoStreamSelector::SingleVideoStreamSelector()
		{

		}


		SingleVideoStreamSelector::~SingleVideoStreamSelector()
		{
		}
	}
}