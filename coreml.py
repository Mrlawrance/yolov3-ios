# Conversion script for yolov3 to Core ML.
# Needs Python 2.7 and Keras 2.1.5

import coremltools

coreml_model = coremltools.converters.keras.convert(
    'dense121.h5',
    input_names='image',
    image_input_names='image',
    image_scale=1/255.)

coreml_model.author = 'Original paper: Joseph Redmon, Ali Farhadi'
coreml_model.license = 'Public Domain'
coreml_model.short_description = "The YOLO network from the paper 'YOLOv3：An Incremental Improvement' (2018)"
coreml_model.input_description['image'] = 'Input image'

print(coreml_model)
coreml_model.save('yolo.mlmodel')
