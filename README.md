# Tiny_Faces_In_Tensorflow_Cpp
TinyFaces C++ implementation  based on  python tensorflow version.
# Tiny Face Detector in TensorFlow

 A TensorFlow port(inference only) of Tiny Face Detector from [authors' MatConvNet codes](https://github.com/peiyunh/tiny)[1].

# Requirements

Codes are written in C++, MSVC 2017. Model stored in HDF5 file.
First, download [HDF5](https://support.hdfgroup.org/HDF5/release/obtain518.html)
Then install [OpenCV](https://github.com/opencv/opencv), [TensorFlow](https://www.tensorflow.org/).

# Usage
Visualisation not implemented, model outputs faces prediction as array. Code-only.
Note: hdf5 file is not available here as its slightly exceed limit 100 mb

# Why?

- research c++ and tf.

# How?

- original tf model was loaded to python script (link above) and dumped tensor by tensor to hdf5 file
- hdf5 deserializer used as data source in c++ program.
- model was constructed manually, line per line. Some of python commands unrolls to dozens of c++ lines.