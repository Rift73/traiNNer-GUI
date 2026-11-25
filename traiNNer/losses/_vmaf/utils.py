import torch
import torch.nn.functional as F


def gaussian_kernel_1d(kernel_size, sigma=1):

    # corresponds to matlab fspecial('gaussian',hsize,sigma)
    x = torch.arange(-(kernel_size//2), kernel_size//2+1)
    gauss = torch.exp(-(x**2 / (2.0 * sigma**2)))
    gauss = gauss/gauss.sum()

    return gauss


def vmaf_pad(input, pad):
    '''Pad image using padding mode: dcb|abcdef|fed - same as in VMAF C code'''

    padding_left, padding_right, padding_top, padding_bottom = pad
    w = input.shape[-1]
    h = input.shape[-2]
    if padding_left <= 1 and padding_right <= 1 and padding_top <= 1 and padding_bottom <= 1:
        padded = F.pad(input, (padding_left, 0, padding_top, 0), mode='reflect')
        padded = F.pad(padded, (0, padding_right, 0, padding_bottom), mode='replicate')
    else:
        # pad right
        if padding_right > 0:
            padded = F.pad(input, (0, padding_right-1, 0, 0), mode='reflect')                               # make abcdef|ed
            padded = torch.cat((padded[:, :, :, :w], padded[:, :, :, w-1:w], padded[:, :, :, w:]), dim=-1)  # insert f manually so we have abcdef|fed
        else:
            padded = input
        # pad left
        padded = F.pad(padded, (padding_left, 0, 0, 0), mode='reflect')
        # pad bottom
        if padding_bottom > 0:
            padded = F.pad(padded, (0, 0, 0, padding_bottom-1), mode='reflect')
            padded = torch.cat((padded[:, :, :h, :], padded[:, :, h-1:h, :], padded[:, :, h:, :]), dim=-2)
        # pad top
        padded = F.pad(padded, (0, 0, padding_top, 0), mode='reflect')

    return padded


def fast_gaussian_blur(x, weight, stride=1):
    '''Fast gaussian blur using separable filter

    Args:
        x: input image of shape (b,1,h,w)
        weight: 1d gaussian kernel of shape (1,1,1,kernel_size)
    '''
    if x.is_cuda and torch.version.cuda is not None:
        from .filter_cuda import GaussianBlur
        return GaussianBlur.apply(x, weight, stride)
    return F.conv2d(F.conv2d(x, weight.view(1, 1, 1, -1), stride=(1, stride)), weight.view(1, 1, -1, 1), stride=(stride, 1))
