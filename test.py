x = {'a':1}
y = {'a':2}
for key in x.keys():
    y[key] = x[key]
x['a'] = 3
print(x)
print(y)