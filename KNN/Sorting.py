#-*- encoding: utf-8 -*-
import datetime
import numpy as np

#冒泡算法
def BubbleSorting(testlist):
    list=testlist
    start = datetime.datetime.now()

    for x in range(0,len(list)-1):
        for y in range(0,len(list)-1):
            if not list[y]<list[y+1]:
                tmp=list[y]
                list[y]=list[y+1]
                list[y+1]=tmp
    print(str(list)+'——————>>>BubbleSort '+str(datetime.datetime.now()-start))
    return list

#鸡尾酒算法(定向冒泡算法)
def CocktailSort(testlist):
    list = testlist
    left = 0
    right=len(list)-1
    start = datetime.datetime.now()

    while left<right:
        for x in range(left,right):
            if list[x]>list[x+1]:
                tmp=list[x]
                list[x]=list[x+1]
                list[x+1]=tmp
        right-=1

    print(str(list)+'——————>>>CocktailSort '+str(datetime.datetime.now()-start))

#选择排序
class SelectionSort():
    def startSelectionSort(self,list):
        start = datetime.datetime.now()
        for cur in range(0,len(list)):
            min=cur
            for j in range(cur,len(list)):
                if list[j]<list[min]:
                    min=j
            self.swap(list,min,cur)
        print(str(list) + '——————>>>SelectionSort ' + str(datetime.datetime.now() - start))

    def swap(self,list,i,j):
        tmp=list[i]
        list[i]=list[j]
        list[j]=tmp

#插入排序

#快速排序
class QuickSort:
    def startQuickSort(self,list):
        sortlist=list
        start=datetime.datetime.now()
        self.quickSortRange(sortlist,0,len(sortlist)-1)
        print(str(sortlist) + '——————>>>QuickSort ' + str(datetime.datetime.now() - start))

    def quickSortRange(self,list,left,right):
        if left<=right:                                            #递归结束条件就是 置换区间为零
            pivotPosition=self.partition(list,left,right)          #划分-算法核心，以基数为准，前区间大于数与后区间小于数互换，然后基数归位，返回归位位置
            self.quickSortRange(list,left,pivotPosition-1)         #递归运算前分区
            self.quickSortRange(list,pivotPosition+1,right)        #递归运算后分区
        return

    def partition(self,list,left,right):
        pivote=list[right]                                         #取基数原则：取划分区间最高位的数
        l=left
        r=right                                                    #连基数一起扫描
        while l<=r:                                                #两个指标碰上，就定位这是基数位置
            while list[l]<=pivote and l<r:                         #设置l<r防止两者互相超越
                l+=1
            while list[r]>=pivote and l<r:
                r-=1
            self.swap(list,l,r)
            if l==r:                                               #扫描结束
                self.swap(list,l,right)                            #基数归位
                return l

    def swap(self,list,i,j):
        tmp=list[i]
        list[i]=list[j]
        list[j]=tmp



# testlist = [50056,12,34,55,44555,2355,77,33,4,10004,6,8,2,43,32,3,5.6,100,30,666,43,4343,4546,234,445,5,0,0,6,0]
# testlist = np.random.randn(1000000)
# print(testlist)
# # BubbleSorting(testlist)
# QuickSort().startQuickSort(testlist)  #数据量越大，效果越明显
# CocktailSort(testlist)
# SelectionSort().startSelectionSort(testlist)





