
import numpy as np
import tensorrt as trt

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # parse ONNX

    with open( onnx_file_path, 'rb') as model:
        


        print('Beginning ONNX file parsing')

        parser.parse(model.read())
        print(parser.parse(model.read()))
    

    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine , context

def main():
  # initialize TensorRT engine and parse ONNX modelù
    ONNX_FILE_PATH='weights/model.onnx'
    engine, context = build_engine(ONNX_FILE_PATH)


            
if __name__ == '__main__':
    main()
