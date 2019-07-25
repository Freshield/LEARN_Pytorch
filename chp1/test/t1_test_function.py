#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: t1_test_function.py
@Time: 2019-07-25 15:04
@Last_update: 2019-07-25 15:04
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
from types import FunctionType

def foo(input1, input2, input3=3):
    print(input1)
    print(input2)
    print(input3)

def func_loader(func, input1, input2):
    func(input1, input2)

func_loader(lambda y_true, y_pred: foo(y_true, y_pred, 4) , 1, 2)

foo_code = compile('def foo(): return 1', "<string>", "exec")
print(foo_code.co_consts)
foo_func = FunctionType(foo_code.co_consts[0], globals(), "foo")

print(foo_func)