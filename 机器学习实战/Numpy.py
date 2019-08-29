# -*- encoding: utf-8 -*-
# Author: MingCrash
import numpy as np
from numpy import mat,eye

list = np.random.rand(2,3,4,5)
print(type(list),np.shape(list)) #输出的是数组

matrix = mat(np.random.rand(2,3))
print(type(matrix),np.shape(matrix)) #输出的是矩阵对象

matrix_1 = matrix.I #输出的是矩阵的逆
print(np.shape(matrix_1))

unit_matrix =  matrix * matrix_1  #输出的是单位矩阵
print(np.shape(unit_matrix),'\n',unit_matrix,'\n')
print(unit_matrix - eye(2))   #eye(3) 生成3X3的单位矩阵


