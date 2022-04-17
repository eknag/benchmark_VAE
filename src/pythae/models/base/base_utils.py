from collections import OrderedDict
from typing import Any, Tuple
from torchvision import transforms
import dill
import torch
import io
import random
import cv2
import numpy as np
from instafilter import Instafilter
import skimage
from skimage.transform import resize

class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class CPU_Unpickler(dill.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)

class AugmentationProcessor():
    '''
    The class that collects all the image augmentation methods that might be used in VAEs.
    By default we assume the transformations are applied on torch.Tensors
    For each VAE object you will have an augmentation_processor, where you can get augmentation from it
    then you call transform(input) and the function will return the transformed images
    '''
    def __init__(self):
        SimpleAugmentation = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1)),
        ])


        LargeAugmentation = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.75,1.25)),
        ])


        SimpleVerticalFlipAugmentation = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1)),
        ])


        LargeVerticalFlipAugmentation = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.75,1.25)),
        ])


        SimpleJitterAugmentation = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.2, hue=0.2, contrast=0.5),
        ])


        LargeJitterAugmentation = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.75,1.25)),
            transforms.ColorJitter(brightness=0.2, hue=0.2, contrast=0.5),
        ])


        SimpleVerticalFlipJitterAugmentation = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.2, hue=0.2, contrast=0.5),
        ])


        LargeVerticalFlipJitterAugmentation = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.75,1.25)),
            transforms.ColorJitter(brightness=0.2, hue=0.2, contrast=0.5),
        ])

        denoise_augmentation = numpy_based_filter(denoise)
        aspect_ratio_augmentation = numpy_based_filter(change_aspect_ratio)
        ins_filter_processor = instafilter(Instafilter("Lo-Fi"))
        instafilter_augmentation = numpy_based_filter(ins_filter_processor)


        self.IMAGENET_MEAN = [0.485, 0.456, 0.406] 
        self.IMAGENET_STD = [0.229, 0.224, 0.225]

        self.augmentations = {
                'simple': SimpleAugmentation,
                'large': LargeAugmentation,
                'simple_vertical_flip': SimpleVerticalFlipAugmentation,
                'large_vertical_flip': LargeVerticalFlipAugmentation,
                'simple_jitter': SimpleJitterAugmentation,
                'large_jitter': LargeJitterAugmentation,
                'simple_vertical_flip_jitter': SimpleVerticalFlipJitterAugmentation,
                'large_vertical_flip_jitter': LargeVerticalFlipJitterAugmentation,
                'random_noise': add_random_noise,
                'denoise': denoise_augmentation,
                'ins_filter': instafilter_augmentation,
                'change_aspect_ratio': aspect_ratio_augmentation
            }

        self.augmentation_names = [ 'simple', 
                                    'large', 
                                    'simple_vertical_flip', 
                                    'large_vertical_flip', 
                                    'simple_jitter', 
                                    'large_jitter', 
                                    'simple_vertical_flip_jitter', 
                                    'large_vertical_flip_jitter', 
                                    'random_noise',
                                    'denoise', # buggy
                                    'ins_filter', # tested
                                    'change_aspect_ratio'] # tested

    def get_augmentation(self, aug_type=None, normalize=False, mean=None, std=None):
        '''
        Get a list of callable image transforms
        '''
        if aug_type == None or aug_type not in self.augmentation_names:
            aug_type = random.choice(self.augmentation_names)
        augmentation = self.augmentations[aug_type]    
        if normalize:
            mean = mean if mean is not None else self.IMAGENET_MEAN
            std = std if std is not None else self.IMAGENET_STD
            normalize_aug = transforms.Normalize(mean, std)
            augmentation.transforms.append(normalize_aug)

        return augmentation

def add_random_noise(x: torch.Tensor, sigma=0.25):
    '''
    Add gaussian noise to the input
    '''
    noise_added_to_input = torch.randn(size=x.size()).cuda() * sigma
    noisy_x = x + noise_added_to_input
    return noisy_x

def convert_tensor_to_array(x: torch.Tensor):
    '''
    Assume inputs to be tensors of size # N * channel * H * W, maximum pixel value is 1
    Output numpy of size N * H * W * channel , maximum pixel value is 255
    '''
    return x.permute(0, 2, 3, 1).cpu().numpy() * 255

def convert_array_to_tensor(x: np.array, to_gpu=True):
    '''
    Assume inputs to be numpy array of size # N * H * W * channel, maximum pixel value is 255
    Output tensor of shape N * channel * H * W, maximum pixel value is 1
    '''
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=3)
    x_tensor = torch.Tensor(x) #if not to_gpu else torch.Tensor(x, device="cuda")
    x_tensor = x_tensor.permute(0, 3, 1, 2)/255
    return x_tensor.cuda()

def numpy_based_filter(processor_function):
    '''
    Decorator function for image transformation that requires the conversion to numpy
    '''
    def func(x: torch.Tensor):
        x = convert_tensor_to_array(x)
        x = processor_function(x)
        return convert_array_to_tensor(x)
    return func

def denoise(x: np.array):
    '''
    Denoise an image
    '''
    res_list = []
    for i in range(x.shape[0]):
        current = skimage.restoration.denoise_wavelet(x[i])
        if not np.any(np.isnan(current)):
            res_list.append(current)
        else:
            res_list.append(x[i])
    return np.stack(res_list)

def instafilter(filter_processor):
    '''
    Instrgram filter, with deep neural network
    can be optimized to batch processing
    '''
    def filter_image(x: np.array):
        res_list = []
        n, h, w, c = x.shape
        for i in range(n):
            res_list.append(filter_processor(x[i]))
        return np.stack(res_list)
    return filter_image

def change_aspect_ratio(x: np.array):
    '''
    change the aspect ratio of the image, and pad the image to original size
    '''
    res_list = []
    n, h, w, c = x.shape
    for i in range(n):
        ar_changed = max(0.8, random.random())
        change_w = random.random() > 0.5
        if not change_w:
            target_h, target_w = int(h * ar_changed), w
        else:
            target_h, target_w = h, int(w * ar_changed)
        image_resized = resize(x[i], [target_h, target_w])
        top_padding = random.randint(0 ,h - target_h)
        bot_padding = h - target_h - top_padding
        left_padding = random.randint(0 ,w - target_w)
        right_padding = w - target_w - left_padding
        image_resized = cv2.copyMakeBorder(image_resized, top_padding, bot_padding,\
            left_padding, right_padding, cv2.BORDER_DEFAULT)
        res_list.append(image_resized)
    return np.stack(res_list)