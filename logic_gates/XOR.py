from AND import AND
from NAND import NAND
from OR import OR

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)

print(XOR(0, 0) == 0)
print(XOR(1, 0) == 1)
print(XOR(0, 1) == 1)
print(XOR(1, 1) == 0)
