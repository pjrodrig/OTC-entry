import tensorflow as tf
import heapq as hp
import cv2.cv2 as cv
import numpy as np

class Being:



    delta_wave = [] # 0.5 to 3 hz
    theta_wave = [] # 3 to 8 hz
    alpha_wave = [] # 8 to 12 hz
    beta_wave = [] # 12 to 38 hz
    gamma_wave = [] # 38 to 42 hz

    ### VISION ###  
    vision = []
    width = 84
    height = 84
    top = []
    red = []
    green = []
    blue = []
    gray = []
    c1 = []
    c2 = []
    c3 = []
    edge_detection = []
    ### END VISION ###
    """
    neuron types:  
        switch on/off
        dial range
        constant timed ticks


        - brightness = grayscale value - >  rgb changes equally
            evaluate at different brightness thresholds
        horizontal edge detection pix - pix to left
        vertical edge detection
        
    """
    neurons = {}
    activations = []

    """
    transmission %
    reuptake %
    """
    connections = {}
    n_count = 0

    tick = 0

    """
    types of stimulation
    """
    chemicals = {
        'a': 1,
        'b': 1,
        'c': 1,
        'd': 1,
        'e': 1,
        'f': -1,
        'g': -1,
        'h': -1
    }

    def __init__(self):
        pass

    def evaluate(self, observation):
        self.evaluate_image(observation)
        

    def init_vision(self):
        n_count += 1
        mean_neuron = {
            'type': 'mean',
            'uid': n_count
        }
        self.vision.append(mean_neuron)
        self.mean = mean_neuron
        neurons[mean_neuron['uid']] = mean_neuron
        n_count += 1
        median_neuron = {
            'type': 'median',
            'uid': n_count
        }
        self.vision.append(median_neuron)
        self.median = median_neuron
        neurons[median_neuron['uid']] = median_neuron
        for y in range(0, height):
            for x in range(0, width):
                if(y < 10):
                    n_count += 1
                    top_neuron = {
                        'type': 'vision-R-' + str(x) + '-' + str(y),
                        'uid': n_count
                    }
                    vision.append(top_neuron)
                    self.neurons[top_neuron['uid']] = top_neuron
                    self.top.append(top_neuron)

                n_count += 1
                red_neuron = {
                    'type': 'vision-red-' + str(x) + '-' + str(y),
                    'uid': n_count
                }
                n_count += 1
                green_neuron = {
                    'type': 'vision-green-' + str(x) + '-' + str(y),
                    'uid': n_count
                }
                n_count += 1
                blue_neuron = {
                    'type': 'vision-blue-' + str(x) + '-' + str(y),
                    'uid': n_count
                }
                n_count += 1
                gray_neuron = {
                    'type': 'vision-gray-' + str(x) + '-' + str(y),
                    'uid': n_count
                }
                n_count += 1
                canny1_neuron = {
                    'type': 'vision-c1-' + str(x) + '-' + str(y),
                    'uid': n_count
                }
                n_count += 1
                canny2_neuron = {
                    'type': 'vision-c2-' + str(x) + '-' + str(y),
                    'uid': n_count
                }
                n_count += 1
                canny3_neuron = {
                    'type': 'vision-c3-' + str(x) + '-' + str(y),
                    'uid': n_count
                }

                self.vision.append(red_neuron)
                self.red.append(red_neuron)
                self.vision.append(green_neuron)
                self.vision.append(blue_neuron)
                self.vision.append(gray_neuron)
                self.vision.append(canny1_neuron)
                self.vision.append(canny2_neuron)
                self.vision.append(canny3_neuron)
                self.neurons[red_neuron['uid']] = red_neuron
                self.neurons[green_neuron['uid']] = green_neuron
                self.neurons[blue_neuron['uid']] = blue_neuron
                self.neurons[gray_neuron['uid']] = gray_neuron
                self.neurons[canny1_neuron['uid']] = canny1_neuron
                self.neurons[canny2_neuron['uid']] = canny2_neuron
                self.neurons[canny3_neuron['uid']] = canny3_neuron


    def evaluate_image(self, image):
        top, mean, median, images = self.process_image(image)
        self.populate_image_neurons(top, mean, median, images)


    def process_image(self, image):
        top = image[:10]
        image = image[10:]
        grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        canny1 = cv.Canny(grayscale, 0, 100)
        canny2 = cv.Canny(grayscale, 130, 150)
        canny3 = cv.Canny(grayscale, 180, 200)
        median = np.median(grayscale)
        mean = np.mean(grayscale)
        # cv.imwrite('./img/top.jpg', top)
        # cv.imwrite('./img/image.jpg', image)
        # cv.imwrite('./img/gray.jpg', grayscale)
        # cv.imwrite('./img/canny1.jpg', canny1)
        # cv.imwrite('./img/canny2.jpg', canny2)
        # cv.imwrite('./img/canny3.jpg', canny3)
        # cv.imwrite('./img/cannyMean.jpg', cmean)
        # cv.imwrite('./img/cannyMedian.jpg', cmedian)
        tp_image = image.transpose()
        return top.flatten(), mean, median, [
            tp_image[0],
            tp_image[1],
            tp_image[2],
            grayscale,
            canny1,
            canny2,
            canny3
        ]

    def activate_vision_neurons(self, top, mean, median, images):
        for activation, neuron in zip(
            np.array([mean, median, top, images]).flatten(), 
            np.array([self.mean, self.median, self.top, self.images]).flatten()
        ):
            self.activate_vision_neuron(activation, neuron['uid'])

    def activate_vision_neuron(self, activation, uid):
        hp.heappush(activations, (activation/256, uid)

