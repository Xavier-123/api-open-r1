

class CustomError(Exception):
    """自定义错误类型"""
    def __init__(self, message=""):
        super().__init__()
        self.message = message
        self.code = -1

    def __str__(self):
        return self.message

class BinaryDecodingError(CustomError):
    """二进制文件解码错误"""

    def __init__(self, message=""):
        super().__init__()
        self.message = f"Binary decoding error! {message}"
        self.code = 1


class FileConversionError(CustomError):
    """文件转化错误"""

    def __init__(self, message=""):
        super().__init__()
        self.message = f"File conversion error! {message}"
        self.code = 2

class DataDistillationError(CustomError):
    """数据蒸馏错误"""

    def __init__(self, message=""):
        super().__init__()
        self.message = f"Data distillation error! {message}"
        self.code = 2