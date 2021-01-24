# Face-Recognition-and-Tracking

<img src="/result.gif" width="350" height="400"/>

This repository shows how to implement end to end Face Recognition and Tracking. The training (face recognition) performed by applying deep learning.
The presence and location of a face were detected in an image using [facessd_mobilenet_v2_quantized_open_image_v4](https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite) and then using a [model](https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7) from [openFace project](https://cmusatyalab.github.io/openface/), we have extracted feature vectors (128-d e.g. a array that contains 128 numerical value called face embidding) that quantify the face in image. The same process was applied to every image in our training data in order to get face embeddings. After that we have trained a Support Vector Machine model on top of obtained face embeddings.
After training face recognition model was deployed along with [SSD model trained v2 pretrained model](https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite) and object tracking model [Deep Sort](https://github.com/nwojke/deep_sort) on a resource-constrained hardware (raspberry-pi Coral Edge TPU usb Accelerator). 
It is important to underline that SSD model trained v2 pretrained model and object tracking model (Deep Sort) are used perform realtime tracking after face is recognized. 

Please follow [Pyimagesearch's tutorial](https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator) to setup the rasberry pi for google coral's TPU usb Accelerator.
# Inference:

To run real time inference: python object_tracker.py. Note that --input, --confidence, --cosine_distance, --nms_thresholdare, and --output are optional. One can ajust any of those commands line to their need.

--input is the path to input video
--confidence is the minimum proba to filter weak detections
--cosine_distance is the cosine distance (deep sort param)
--nms_thresholdare is non-maximum suppression threshold (deep sort param)
--output is the path to output vide
# References

https://www.pyimagesearch.com/raspberry-pi-for-computer-vision/
https://coral.ai/docs/
https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/
https://github.com/omarabid59/TensorflowDeepSortTracking
