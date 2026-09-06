# 判断一个整数是奇数还是偶数

while True:
    try:
        num = int(input("请输入一个整数："))
        break
    except ValueError:
        print("输入无效，请输入整数！")

# 能被 2 整除就是偶数，否则是奇数
if num % 2 == 0:
    print(num, "是偶数")
else:
    print(num, "是奇数")
