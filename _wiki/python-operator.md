---
layout: wiki
title: Python语法
categories: Python语法
description: Python 常见的重载运算符
keywords: Python
---

## python中常见的重载运算符
**什么是运算符重载:** 让自定义的类生成的对象(实例)能够使用运算符进行操作.

**作用:**
+ 让自定义的实例或者是对象能够内建对象一样进行运算符操作.
+ 让程序简洁易读.
+ 对自定义对象将运算符赋予新的规则
### 算术运算符的重载
|方法名|运算符和表达式|说明|
|----|----|----|
|`__add__(self,rhs)`|`self+rhs`|加法|
|`__sub__(self,rhs)`|`self-rhs`|减法|
|`__mul__(self,rhs)`|`self*rhs`|乘法|
|`__truediv__(self,rhs)`|`self/rhs`|除法|
|`__floordiv__(self,rhs)`|`self//rhs`|floor除法|
|`__mod__(self,rhs)`|`self%rhs`|取模(取余数)|
|`__pow__(self,rhs)`|`self**rhs`|幂运算|
一般二元运算符的重载方法的格式如下所示:
```python
def __xx__(self,other):
    pass
```
### 反向运算符的重载
当运算符的左侧为内建类型的时候,右侧为自定义类型进行算术运算的时候会出现`TypeError`错误,这是因为无法修改内建类型的代码,所以此时需要使用反向运算符的重载.
|方法名|运算符和表达式|说明|
|----|----|----|
|`__radd__(self,lhs)`|`lhs+self`|加法|
|`__rsub__(self,lhs)`|`lhs-self`|减法|
|`__rmul__(self,lhs)`|`lhs*self`|乘法|
|`__rtruediv__(self,lhs)`|`lhs/self`|除法|
|`__rfloordiv__(self,lhs)`|`lhs//self`|floor除法|
|`__rmod__(self,lhs)`|`lhs%self`|取模(取余数)|
|`__rpow__(self,lhs)`|`lhs**self`|幂运算|
### 复合算术运算符的重载
|方法名|运算符和表达式|说明|
|----|----|----|
|`__iadd__(self,rhs)`|`self+=rhs`|加法|
|`__isub__(self,rhs)`|`self-=rhs`|减法|
|`__imul__(self,rhs)`|`self*=rhs`|乘法|
|`__itruediv__(self,rhs)`|`self/=rhs`|除法|
|`__ifloordiv__(self,rhs)`|`self//=rhs`|floor除法|
|`__imod__(self,rhs)`|`self%=rhs`|取模(取余数)|
|`__ipow__(self,rhs)`|`self**=rhs`|幂运算|
### 比较运算符的重载
|方法名|运算符和表达式|说明|
|----|----|----|
|`__lt__(self,rhs)`|`self+=rhs`|小于|
|`__le__(self,rhs)`|`self-=rhs`|小于等于|
|`__gt__(self,rhs)`|`self*=rhs`|大于|
|`__ge__(self,rhs)`|`self/=rhs`|大于等于|
|`__eq__(self,rhs)`|`self//=rhs`|等于|
|`__ne__(self,rhs)`|`self%=rhs`|不等于|
### 位运算符重载
|方法名|运算符和表达式|说明|
|----|----|----|
|`__and__(self,rhs)`|`self & rhs`|位与|
|`__or__(self,rhs)`|`self | rhs`|位或|
|`__xor__(self,rhs)`|`self ^ rhs`|位异或|
|`__lshift__(self,rhs)`|`self << rhs`|左移|
|`__rshift__(self,rhs)`|`self >> rhs`|右移|
### 反向位运算符重载
|方法名|运算符和表达式|说明|
|----|----|----|
|`__rand__(self,lhs)`|`lhs & self`|位与|
|`__ror__(self,lhs)`|`lhs | self`|位或|
|`__rxor__(self,lhs)`|`lhs ^ self`|位异或|
|`__rrshift__(self,lhs)`|`lhs << self`|左移|
|`__rfloordiv__(self,lhs)`|`lhs >> self`|右移|
### 一元运算符的重载
|方法名|运算符和表达式|说明|
|----|----|----|
|`__neg__(self)`|`- self`|负号|
|`__pos__(self)`|`+ self`|正号|
|`__invert__(self)`|`~ self`|取反|
用以以下的形式
```python
class Test:
    def __xxx__(self):
        pass
```
### in/not in 运算符重载
格式形式如下所示
```python
class Test:
    def __contains__(self,element):
        pass
```
当重载了`__contains__()`之后,`in`和`not in`运算符是都可以使用的.
### 索引和切片运算符重载方法
|方法名|运算符和表达式|说明|
|----|----|----|
|`__getitem__(self)`|`x = self[i]`|索引/切片取值|
|`__setitem__(self)`|`self[i] = v`|索引/切片赋值|
|`__delitem__(self)`|`del self`|`del`语句删除索引/切片|
可以自定义的类型的对象能够支持索引和切片操作。
