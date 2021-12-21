class ArrayShapeError(Exception):
    def __init__(self, msg=None):
        self.msg = msg
        super().__init__(self.msg)
