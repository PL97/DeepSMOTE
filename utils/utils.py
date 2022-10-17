

class UnNormalize(object):
    '''
    unnormalize the image according to the input mean and std
    defult are imagenet mean and std
    '''
    def __init__(self, fake=False, norm_type="others", grayscale=False):
        if grayscale:
            mean = [0.5]
            std = [0.5]
        else:
            if norm_type == "imagenet":
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
            else:
                mean = (0.5, 0.5, 0.5)
                std = (0.5, 0.5, 0.5)
        self.mean = mean
        self.std = std
        self.fake = fake

    def __call__(self, tensor, cwh=True, batch=False):
        if self.fake:
            return tensor
            
        for idx, (m, s) in enumerate(zip(self.mean, self.std)):
            if cwh:
                if batch:
                    tensor[:, idx, :, :] = tensor[:, idx, :, :]*s + m
                else:
                    tensor[idx, :, :] = tensor[idx, :, :]*s + m
            else:
                tensor[:, :, idx] = tensor[:, :, idx]*s + m
            # The normalize code -> t.sub_(m).div_(s)
        return tensor