import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from scipy import signal

__all__ = ['get_preprocessing_pipelines', 'Compose', 'Normalize', 'CenterCrop', 'RgbToGray', 'RandomCrop',
           'HorizontalFlip', 'AddNoise', 'NormalizeUtterance', 
           'FrameDownSample', 'speed_perturb', 'add_noise', 'add_rev']

def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- voxceleb config
    crop_size = (96, 96)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([
                                # Normalize( 0.0,255.0),
                                RandomCrop(crop_size),
                                HorizontalFlip(0.5),
                                # Normalize(mean, std)
                                ])

    preprocessing['val'] = Compose([
                                # Normalize( 0.0,255.0 ),
                                CenterCrop(crop_size),
                                # Normalize(mean, std) 
                                ])
    preprocessing['test'] = preprocessing['val']
    return preprocessing


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RgbToGray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


class NormalizeUtterance():
    """Normalize per raw audio by removing the mean and divided by the standard deviation
    """
    def __call__(self, signal):
        signal_std = 0. if np.std(signal)==0. else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std


class AddNoise(object):
    """Add SNR noise [-1, 1]
    """

    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
        
        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 **2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [np.float32, np.float64], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
            return desired_signal


def FrameDownSample(data, fps_act):
    if fps_act == 5:
        data = np.array([x for i, x in enumerate(data) if i%5==0])      
    if fps_act == 1:
        data = np.array([x for i, x in enumerate(data) if i%25==0])
    if fps_act == 2:     
        np.random.shuffle(data)
        data = data[0:2,:,:]
    return data


class FeatureAug(nn.Module):
    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x): 
        with torch.no_grad(): 
            x = self.mask_along_axis(x, dim=2)
            x = self.mask_along_axis(x, dim=1)
        return x


def add_noise(audio, noisecat, numnoise, noiselist, noisesnrrange, num_second):
    clean_db    = 10 * np.log10(np.mean(audio ** 2)+1e-4) 
    numnoise    = numnoise[noisecat]
    noiselist   = random.sample(noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
    noises = []
    for noise in noiselist:
        noiseaudio, sr = sf.read(noise)
        length = num_second * 16000 + 240
        if noiseaudio.shape[0] <= length:
            shortage = length - noiseaudio.shape[0]
            noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
        noiseaudio = noiseaudio[start_frame:start_frame + length]
        noiseaudio = np.stack([noiseaudio],axis=0)
        noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4) 
        noisesnr   = random.uniform(noisesnrrange[noisecat][0],noisesnrrange[noisecat][1])
        noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
    noise = np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True)
    return noise + audio


def add_rev(audio, num_second, rir_files):
    rir_file    = random.choice(rir_files)
    rir, sr     = sf.read(rir_file)
    rir         = np.expand_dims(rir.astype(np.float),0)
    rir         = rir / np.sqrt(np.sum(rir**2))
    return signal.convolve(audio, rir, mode='full')[:,:num_second * 16000 + 240]


def speed_perturb(audiofeat, sid, n_spk, speed_idx, speeds, sample_rate):
    """ Apply speed perturb to the data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speed_idx > 0:
        audiofeat, _ = torchaudio.sox_effects.apply_effects_tensor(
            audiofeat, sample_rate, [['speed', str(speeds[speed_idx])], ['rate',str(sample_rate)]])
        aug_sid = sid + n_spk * speed_idx
    else:
        aug_sid = sid

    return audiofeat, aug_sid