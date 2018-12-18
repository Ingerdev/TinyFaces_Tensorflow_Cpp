#pragma once
#include <iostream>
#include <istream>

#include "videorender/Prediction.h"

namespace tiny_face_model
{
	namespace internal
	{
		//convert tensor to predictions
		template <typename T>
		class TensorPredictionHelper
		{
		public:
			TensorPredictionHelper::TensorPredictionHelper() :counter(0), elements_{}
			{

			}

			TensorPredictionHelper& operator<<(const char*)
			{
				//do nothing as its wrong type
				return *this;
			}

			TensorPredictionHelper& operator<<(const T &value)
			{
				elements_[counter] = value;
				counter++;

				if (counter == 5)
				{
					counter = 0;
					preds_.push_back(Prediction( elements_ ));
					counter = 0;
				}

				return *this;
			}

			std::vector<Prediction> get_predictions() const
			{
				return preds_;
			}

			//clear internal state
			void reset()
			{
				counter = 0;
				preds_.clear();
			}

		private:			
			std::array<T, 5> elements_;
			size_t counter;
			std::vector<Prediction> preds_;
		};


	}
}
