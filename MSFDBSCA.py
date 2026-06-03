import os
from opfunu.cec_based.cec2022 import *
import numpy as np
from copy import deepcopy

# ==== 参数设置 ====
# PopSize    = 200
PopSize    = 100
DimSize    = 100
LB         = [-100] * DimSize
UB         = [100] * DimSize
TrialRuns  = 20
MaxFEs     = 1000 * DimSize
curIter    = 0
FuncNum    = 1
MaxIter    = int(MaxFEs / PopSize * 2)

# MS-FDBSCA 参数
beta = 0.5   # FDB 权重

# ==== 全局变量 ====
Pop     = np.zeros((PopSize, DimSize))
FitPop  = np.zeros(PopSize)

# ==== 初始化种群 ====
def Initialization(func):
    global Pop, FitPop
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])

# ==== 边界检查（镜像修正） ====
def Check(indi):
    global LB, UB, DimSize
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
    return indi

# ==== MS-FDBSCA 更新算子（基于原始 SCA 框架） ====
def MS_FDBSCA_operator(func):
    global Pop, FitPop, curIter, MaxIter, DimSize, beta

    # 当前全局最优
    BestIdx = np.argmin(FitPop)
    Best = deepcopy(Pop[BestIdx])

    # 步长衰减因子（SCA 原式）
    a = 2 - curIter * (2 / MaxIter)

    # FDB 所需统计量
    fmin, fmax = np.min(FitPop), np.max(FitPop)
    dists = np.linalg.norm(Pop - Best, axis=1)
    dmin, dmax = np.min(dists), np.max(dists)

    for i in range(PopSize):
        oldSol = deepcopy(Pop[i])
        oldFit = FitPop[i]
        newSol = deepcopy(oldSol)

        # Multi-Score
        f_norm = (FitPop[i] - fmin) / (fmax - fmin + 1e-12)
        d_norm = (dists[i] - dmin) / (dmax - dmin + 1e-12)
        S_F = f_norm
        S_D = d_norm
        S_FDB = beta * S_F + (1 - beta) * S_D

        for j in range(DimSize):
            # MS-FDB 随机调制参数
            r1 = a * np.random.rand() * (1 - S_FDB)
            r2 = 2 * np.pi * np.random.rand() * (1 - S_D)
            r3 = 2 * np.random.rand() * (1 - S_F)
            r4 = np.random.rand() * (1 - S_FDB)

            if r4 < 0.5:
                newSol[j] = oldSol[j] + r1 * np.sin(r2) * abs(r3 * Best[j] - oldSol[j])
            else:
                newSol[j] = oldSol[j] + r1 * np.cos(r2) * abs(r3 * Best[j] - oldSol[j])

        newSol = Check(newSol)
        newFit = func(newSol)

        # 贪婪选择
        if newFit < oldFit:
            Pop[i] = newSol
            FitPop[i] = newFit
        else:
            Pop[i] = oldSol
            FitPop[i] = oldFit

# ==== 运行算法 ====
def RunMS_FDBSCA(func):
    global FitPop, curIter, TrialRuns, DimSize
    All_Trial_Best = []

    for i in range(TrialRuns):
        BestList = []
        curIter = 0
        np.random.seed(2024 + 24 * i)
        Initialization(func)

        BestList.append(min(FitPop))
        while curIter < MaxIter:
            MS_FDBSCA_operator(func)
            curIter += 1
            BestList.append(min(FitPop))

        All_Trial_Best.append(BestList)

    np.savetxt("./MS-FDBSCA/MS-FDBSCA_Data/CEC2022/F" + str(FuncNum + 1) + "_" + str(DimSize) + "D.csv",
               All_Trial_Best, delimiter=",")

# ==== 主函数 ====
def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize * 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(dim), F22022(dim), F32022(dim), F42022(dim), F52022(dim), F62022(dim),
               F72022(dim), F82022(dim), F92022(dim), F102022(dim), F112022(dim), F122022(dim)]

    for i in range(len(CEC2022)):
        FuncNum = i
        RunMS_FDBSCA(CEC2022[i].evaluate)

if __name__ == "__main__":
    if not os.path.exists('./MS-FDBSCA/MS-FDBSCA_Data/CEC2022'):
        os.makedirs('./MS-FDBSCA/MS-FDBSCA_Data/CEC2022')

    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)
