---
layout: wiki
title: Python语法
categories: Python语法
description: Python 常用语法使用方法
keywords: Python
---

## with 语句使用

### with 语句的原理
+ 上下文管理协议(Context Management Protocol)：包含有方法`__enter__`和方法`__exit__`，支持该协议的对象要实现这两个方法。
+ 上下文管理器(Context Manager)：支持上下文刮玻璃协议的对象，这种对象实现了`__enter__()`和`__exit__()`方法。上下文管理器定义执行with语句的时候要建立的运行时上下文，负责执行with语句块上下文中的进入与退出的操作。通常使用with语句调用上下文管理器，也可以通过直接调用其中的方法来使用。

with语句的结构如下所示
```python
with experssion as var:
    block sentence
```
其中`expresion`可以是任意表达式, `as var`是可选的，其一般的执行过程如下所示
1. 执行`expression`,生成上下文管理器`context_manager`;
2. 获取上下文管理的`__exit__()`方法,并保存起来用于之后的调用;
3. 调用上下文管理器的`__enter__()`方法,如果使用了`as`子句,则将`__enter__()`方法的返回值赋值给`as`子句中的`var`;
4. 执行`block sentence`中的表达式;
5. 不管是否执行过程中发生了异常,执行上下文管理器的`__exit__()`方法,`__exit__()`方法负责执行“清理”工作,例如释放资源等等。如果执行过程中没有发生异常，或者是语句体中执行行了`break/continue/return`,则以`None`作为参数调用`__exit__()`;如果执行过程中出现异常事件,则使用`sys.exc_info`得到的异常信息作为参数调用`__exit__(exc_type,exc_value,exc_traceback)`;
6. 出现异常的时候,如果`__exit__(type,value,tracback)`返回`False`,则会重新抛出异常,让`with`之外的语句逻辑来处理异常,这也是通用的做法;如果返回`True`,则忽略异常事件,不再对异常进行处理。

### 自定义上下文管理器
Python的with语句是提供一个有效的机制,让代码更加简练，同时在异常产生时候，清理工作更简单。
```python
class DBManager(object):
    def __init__(self):
      pass
    def __enter__(self):
      print('__enter__ processed !')
      return self
    def __exit__(self):
      print('__exit__ processed!')
      return True
def build_instance():
    # 工厂方法
    return DBManager()
with build_instance() as dbmanager:
    print('with sentence')
```
**注意：with后面必须跟上一个上下文管理器,如果使用了as,则是把上下文管理器的`__enter__()`方法的返回值赋值给target,target可以是单个变量,或者由"()"括起来的元组(不能是仅仅由","分隔的变量列表,必须加上"()")**

实际上就是,当我们使用`__enter__()`方法被调用,并且将返回值赋值给`as`后面的变量,并且在退出`with`的时候自动执行`__exit__()`方法。

### 总结
一般自定义上下文管理器来对软件系统的资源进行管理,例如有数据库连接、或者是共享资源的访问控制等等。

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
