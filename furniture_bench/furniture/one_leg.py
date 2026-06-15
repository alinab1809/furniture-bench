from furniture_bench.furniture.square_table import SquareTable


class OneLeg(SquareTable):
    def __init__(self, env_idx=0):
        super().__init__(env_idx)
        self.should_be_assembled = [(0, 4)]
