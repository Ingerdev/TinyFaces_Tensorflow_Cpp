#pragma once
#pragma warning(push, 0) 
#include <array>
#include <iostream>
#pragma warning(pop) 
namespace videorender
{
	//prediction storage
	struct Prediction
	{
	public:
		float probability;
		int left;
		int top;
		int right;
		int bottom;
		Prediction(float probability, int left, int top, int right, int bottom)
		{
			this->probability = probability;
			this->left = left;
			this->top = top;
			this->left = right;
			this->bottom = bottom;
		}		

		template <typename T>
		Prediction(std::array<T,5> values)
		{
			this->probability = values[4];
			this->left = static_cast<int>(values[0]);
			this->top = static_cast<int>(values[1]);
			this->right = static_cast<int>(values[2]);
			this->bottom = static_cast<int>(values[3]);
		}

		std::ostream& print(std::ostream& os) const
		{
			os << "[" << left << "," << top << "," << right << "," << bottom << "," << probability << "]\n";
			return os;
		}

		inline int width() const 
		{
			return right - left;
		}

		inline int height() const
		{
			return bottom - top;
		}

	};

	inline std::ostream& operator<<(std::ostream& os,const Prediction& obj) 
	{
		return obj.print(os);
	}

}