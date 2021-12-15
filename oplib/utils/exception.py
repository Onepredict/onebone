class ArrayShapeError(Exception):
    def __init__(self, msg=None):
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
