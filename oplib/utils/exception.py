class ArrayShapeError(Exception):
    def __init__(self, x, msg=None):
        if msg is None:
            msg = f"The array dimension must be below {x}-dimensions."
        self.x = x
        self.msg = msg
        super().__init__(self.msg)


class ValueLevelError(Exception):
    def __init__(self, msg=None):
        self.msg = msg
        super().__init__(self.msg)


class DataError(Exception):
    def __init__(self, msg=None):
        self.msg = msg
        super().__init__(self.msg)
