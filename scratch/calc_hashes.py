def fnv1a_24(s):
    hash = 2166136261
    for char in s:
        hash ^= ord(char)
        hash &= 0xFFFFFFFF
        hash *= 16777619
        hash &= 0xFFFFFFFF
    return hash & 0xFFFFFF

labels = ["add", "sub", "mul", "div", "lt", "gt", "le", "ge", "eq", "ne", "exit", "print_str", "print_i32", "streq", "array_new", "array_set", "array_get", "era", "num", "str", "pair", "dup", "branch", "call", "strlen"]
for l in labels:
    print(f"{l}: {fnv1a_24(l):06x}")
