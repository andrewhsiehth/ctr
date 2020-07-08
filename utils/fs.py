from pyarrow import fs 
from pyarrow.lib import NativeFile

import io 

# DO NOT ASK!

def open_buffered_file_reader(uri: str, buffer_size: int = io.DEFAULT_BUFFER_SIZE) -> io.BufferedReader: 
    try:
        input_file = open_input_file(uri) 
        return io.BufferedReader(input_file, buffer_size=buffer_size)  
    except: 
        input_file.close() 

def open_input_file(uri: str) -> NativeFile:  
    filesystem, path = fs.FileSystem.from_uri(uri) 
    return filesystem.open_input_file(path) 


def file_info(uri: str) -> fs.FileInfo: 
    filesystem, path = fs.FileSystem.from_uri(uri) 
    info, = filesystem.get_file_info([path]) 
    return info 
    
def open_buffered_stream_writer(uri: str, buffer_size: int = io.DEFAULT_BUFFER_SIZE) -> io.BufferedWriter: 
    try: 
        output_stream = open_output_stream(uri) 
        return io.BufferedWriter(output_stream, buffer_size=buffer_size) 
    except Exception as e:  
        output_stream.close() 
        raise e  

def open_output_stream(uri: str) -> NativeFile: 
    filesystem, path = fs.FileSystem.from_uri(uri) 
    return filesystem.open_output_stream(path) 

def write_file(buffer: io.BytesIO, uri: str, buffer_size: int = io.DEFAULT_BUFFER_SIZE) -> None: 
    with open_buffered_stream_writer(uri, buffer_size=buffer_size) as output_stream: 
        output_stream.write(buffer.getbuffer()) 




    