import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    heatmap_blob = output['Mconv7_stage2_L2']
    reshaped_heatmap = np.zeros([heatmap_blob.shape[1], input_shape[0], input_shape[1]])
    # TODO 2: Resize the heatmap back to the size of the input
    for i in range(len(heatmaps[0])):
        reshaped_heatmap[h] = cv2.resize(heatmap_blob[0][i], input_shape[0:2][::-1])
    
    return reshaped_heatmap


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    '''
    
    # TODO 1: Extract only the first blob output (text/no text classification)
    text_blob = output['model/segm_logits/add']
    #print(text_blob.shape)
    
    # TODO 2: Resize this output back to the size of the input
    text = np.zeros([text_blob.shape[1], input_shape[0], input_shape[1]])
    #print(text.shape)
    for i in range(len(text_blob[0])):
        text[i] = cv2.resize(text_blob[0][i], input_shape[0:2][::-1])
        #print(text.shape)

    return text


def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    # TODO 1: Get the argmax of the "color" output
    #print(output['color'].shape)
    ccolor = np.argmax(output['color'].flatten())
    
    # TODO 2: Get the argmax of the "type" output
    ctype = np.argmax(output['type'].flatten())
    
    return ccolor, ctype


def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
