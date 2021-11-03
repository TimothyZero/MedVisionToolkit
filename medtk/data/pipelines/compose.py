import collections
import random
import traceback

from .viewer import Display

d = Display()


class OneOf(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.latitude = 1.0
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                raise NotImplementedError
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    @property
    def canBackward(self):
        return True

    def setLatitude(self, val):
        self.latitude = val
        for t in self.transforms:
            if hasattr(t, 'setLatitude'):
                t.setLatitude(self.latitude)

    def getLatitude(self):
        return self.latitude

    def __call__(self, result, forward=True):
        if forward:
            t = random.choice(self.transforms)
        else:
            name = result[0]['history'][-1]
            t = [tt for tt in self.transforms if name in tt.name][0]
        result = t(result, forward)
        if result is None:
            return None
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '(['
        for t in self.transforms:
            format_string += '\n'
            format_string += '        {0},'.format(t)
        format_string += '\n        ])'
        return format_string

    def __contains__(self, item: str):
        return item in self.__repr__()


class ForwardCompose(object):
    def __init__(self, transforms: (list, tuple)):
        assert isinstance(transforms, collections.abc.Sequence)
        self.latitude = 1.0
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                raise NotImplementedError
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def setLatitude(self, val):
        self.latitude = val
        for t in self.transforms:
            if hasattr(t, 'setLatitude'):
                t.setLatitude(self.latitude)

    def getLatitude(self):
        return self.latitude

    def __call__(self, result):
        for t in self.transforms:
            try:
                result = t(result)
            except Exception as e:
                print(self.__class__.__name__, 'Error')
                print(t)
                d(result)
                traceback.print_exc()
                raise e
            if result is None:
                return None
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0},'.format(t)
        format_string += '\n)'
        return format_string

    def __contains__(self, item: str):
        return item in self.__repr__()


class BackwardCompose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                raise NotImplementedError
            elif callable(transform):
                if hasattr(transform, 'canBackward'):
                    self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')
        # it should support backward
        only_forward = [t for t in self.transforms if not t.canBackward]
        assert not only_forward, \
            f"All transforms used in BackwardCompose must support backward! But these are not!\n{only_forward}"

    def __call__(self, result):
        for t in self.transforms:
            try:
                result = t(result, forward=False)
            except Exception as e:
                print(self.__class__.__name__, 'Error')
                print(t)
                d(result)
                traceback.print_exc()
                # raise e
            if result is None:
                return None
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0},'.format(t)
        format_string += '\n)'
        return format_string

    def __contains__(self, item: str):
        return item in self.__repr__()