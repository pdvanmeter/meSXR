"""
This module allows for the definition of new data types which are
used throughout the module.
"""
class AttrDict(dict):
    """
    A dictionary which also allows access of entries via object attributes. In some
    cases this can simplify code, especially when the dictionary itself contains other
    dictionary objects.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self