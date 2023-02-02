def salam(a):
    if a:
        return False, False
    return True
a = False
cal = salam(a)
print(cal[0])