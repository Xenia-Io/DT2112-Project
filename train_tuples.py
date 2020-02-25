class Trainer:
    def __init__(self, mask1, spec1, mask2, spec2):
        self.mask1 = mask1
        self.spec1 = spec1
        self.mask2 = mask2
        self.spec2 = spec2

    def __str__(self):
        print(f"mask1: {self.mask1} mask2: {self.mask2} spec1: {self.spec1} spec1: {self.spec1}")
