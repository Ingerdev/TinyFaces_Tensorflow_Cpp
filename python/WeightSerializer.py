class WeightSerializerBinary():
    def __init__(self,stream_file_name,dict_file_name):
        self.file_stream = open(stream_file_name,"wb")
        self.dict_file = open(dict_file_name,"wt")
        self.details = []

    def close(self):
        def dump_to_json(object,file):
            import json
            file.write(json.dumps(object))

        self.file_stream.close()
        dump_to_json(self.details,self.dict_file)
        self.dict_file.close()

    #serialize binary protocol
    def _write_stream_binary(self,prefix,name,value):
        import sys
        self.details.append({'name':prefix+name,'position':self.file_stream.tell(),'size':sys.getsizeof(value)})
        self.file_stream.write(value)


    def write_stream(self,prefix, name,value):
        self._write_stream_binary(prefix, name,value)

import h5py
class WeightSerializerHDF5():
    def __init__(self,h5_file_name):
        self.h5file = h5py.File(h5_file_name,"w")


    def close(self):
        self.h5file.close()


    #serialize hdf5 protocol
    def _write_stream_hdf5(self,prefix,name,value):
        self.h5file.create_dataset(prefix+name,data = value)

    def write_stream(self,prefix, name,value):
        self._write_stream_hdf5(prefix, name,value)