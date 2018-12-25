# Tiny_Faces_In_Tensorflow_Cpp
TinyFaces C++ implementation  based on  python tensorflow version.  
Note that this repository is copy of [main repository](https://bitbucket.org/ingerdev/tinyfacemodel) and may not reflect latest changes. 
# Tiny Face Detector in TensorFlow

 A TensorFlow port(inference only) of Tiny Face Detector from [authors' MatConvNet codes](https://github.com/peiyunh/tiny)[1].

# Requirements

Codes are written in C++/17, CMake , /W3. Model stored in [HDF5 file](https://bitbucket.org/ingerdev/tinyfacemodel/src/master/Model/weights.h5)

First, download [HDF5](https://support.hdfgroup.org/HDF5/release/obtain518.html)
Then install [OpenCV](https://github.com/opencv/opencv), [TensorFlow](https://www.tensorflow.org/).

# Usage
Read .exe usage message. 

# Why?

- research c++ and tf.

# How?

- original tf model was loaded to python script (link above) and dumped tensor by tensor to hdf5 file
- hdf5 deserializer used as data source in c++ program.
- model was constructed manually, line per line. Some of python commands unrolls to dozens of c++ lines.
