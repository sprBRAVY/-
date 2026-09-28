class LFSR:
    def __init__(self, seed=0xACE1):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def step(self):
        # Полином: X^32+X^28+X^22+X^21+X^20+X^19+X^18+X^16+X^10+X^8+X^7+X^3+1
        # Индексы для битовых сдвигов (0-31)
        bit = ((self.state >> 31) ^ (self.state >> 27) ^ (self.state >> 21) ^
               (self.state >> 20) ^ (self.state >> 19) ^ (self.state >> 18) ^
               (self.state >> 17) ^ (self.state >> 15) ^ (self.state >> 9) ^
               (self.state >> 7) ^ (self.state >> 6) ^ (self.state >> 2)) & 1

        self.state = ((self.state << 1) | bit) & 0xFFFFFFFF
        return self.state

    def random(self):
        # Генерируем 32 новых бита для случайного числа типа float [0, 1)
        for _ in range(32):
            self.step()
        return self.state / 0xFFFFFFFF