#pragma once
//helper class
#pragma warning(push, 0) 
#include <vector>
#include <algorithm>
#include <iterator>
#include <type_traits>
#include <cstdint>

#include <H5Cpp.h>
#include "tensorflow/core/framework/tensor.h"
#pragma warning(pop)

//not-sussessfull try to rewrite helper methods.
//failed at H5::PredType::* using, cant use it it compile-time.
/*
namespace tf_helpers
{
	namespace internal
	{
		namespace tf = tensorflow;
		static const H5::PredType& X = H5::PredType::NATIVE_INT32;
		
		//template<EnumType E> struct X; template<> struct X<enumA> { using type = int; };

		template <typename CPP_T, tf::DataType TF_V, const H5::PredType& H5_O >
		struct tf_map_type
		{
		public:
			using  cpp_type = CPP_T;
			static constexpr tf::DataType tf_value = TF_V;
			static constexpr const H5::PredType& h5_object = H5_O;						
		};

		template <tf::DataType> struct from_tf;
		template <typename > struct from_cpp_type;
		template <const H5::PredType& > struct from_h5;


#define MAKE_TYPEMAP_ENTRY(cpp_t, tf_v, h5_o)\
			static const auto conv_##cpp_t(h5_o); \
			template <> struct from_tf<tf_v>:tf_map_type<cpp_t, tf_v, conv_##cpp_t>{}; \
		    template <> struct from_cpp_type<cpp_t>:tf_map_type<cpp_t, tf_v, conv_##cpp_t>{}; \
			template <> struct from_h5<h5_o>:tf_map_type<cpp_t, tf_v, conv_##cpp_t>{}; 
		
		
		
		MAKE_TYPEMAP_ENTRY(int32_t, tf::DataType::DT_INT32,  H5::PredType::NATIVE_INT32);
		//template <> struct from_tf<tf::DataType::DT_INT32> :tf_map_type<int32_t,
		//	tf::DataType::DT_INT32,	> {}; 
	}

	void test()
	{
		internal::from_tf<tf::DataType::DT_INT32>::h5_object;
	}
}
*/