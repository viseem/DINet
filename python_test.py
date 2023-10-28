# 写一个feibonaqie数列
# 1 1 2 3 5 8 13 21 34 55 89 144
# 用 python 实现
# 1. 用递归实现
# 2. 用循环实现
# 3. 用递归实现，但是要用到缓存，比如用一个字典来保存已经计算过的值，下次如果再次计算，就直接从字典中取值，不要再次计算了

# 1. 用递归实现
def feibonaqie(n):
    if n == 1 or n == 2:
        return 1
    else:
        return feibonaqie(n-1) + feibonaqie(n-2)
    
print(feibonaqie(10))

# 2. 用循环实现
def feibonaqie2(n):
    a, b = 1, 1
    for i in range(n-1):
        a, b = b, a+b
    return a

print(feibonaqie2(10))

# 3. 用递归实现，但是要用到缓存，比如用一个字典来保存已经计算过的值，下次如果再次计算，就直接从字典中取值，不要再次计算了
def feibonaqie3(n):
    if n == 1 or n == 2:
        return 1
    else:
        return feibonaqie3(n-1) + feibonaqie3(n-2)
    
def feibonaqie4(n):
    cache = {}
    if n == 1 or n == 2:
        return 1
    else:
        if n in cache:
            return cache[n]
        else:
            cache[n] = feibonaqie4(n-1) + feibonaqie4(n-2)
            return cache[n]
