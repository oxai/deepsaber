from base.models.base_model import BaseModel


class TestModel(BaseModel):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def name(self):
        return "TestModel"