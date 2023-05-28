from PIL import Image
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import copy
import math

# построение обратной матрицы, где M - матрица, для которой необходимо построить обратную матрицу
def inverse_m(M):
  det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
  return np.array([[M[1,1]/det, -M[0,1]/det], [-M[1,0]/det, M[0,0]/det]])

# генерация квадратного изображения с заданным размером N
def random_img(N):
    A = []
    for j in range(0, N):
        a = []
        for i in range(0, N):
            a.append(randint(0, 10000)/10000)
        A.append(a)
    return A

# диагональный сдвиг на shift пикселей квадратного изображения A с размером N
def right_diag_shift(A, N, shift):
    B = []
    for j in range(0, N):
        b =[]
        for i in range(0, N):
            if i < shift or j >= (N - shift):
                b.append(1)
            else:
                b.append(A[j+shift][i-shift])
        B.append(b)
    return B

# построение рамки шириной frame_width для квадратного изображения A с размером N
def frame(A, N, frame_width):
    B =[]
    for j in range(0, N+2*frame_width):
        b = []
        for i in range(0, N+2*frame_width):
            if i < frame_width or i > N+frame_width-1 or j < frame_width or j > N+frame_width-1:
                b.append(1)
            else:
                b.append(A[j-frame_width][i-frame_width])
        B.append(b)
    return B

# аппроксимация производной функции яркости, где A1, A2 - входные квадратные изображения, N - их размер, frame_width - ширина рамки
def xyz_differences(A1, A2, N, frame_width):
    mX = []
    for j in range(0, N):
        x = []
        for i in range(0, N):
            x.append(0)
        mX.append(x)
    mY = copy.deepcopy(mX)
    mT = copy.deepcopy(mX)
    for j in range(frame_width, N+frame_width):
        for i in range(frame_width, N+frame_width):
            if i < N+frame_width-1 and j < N+frame_width-1:
                mX[i-frame_width][j-frame_width] = (1/4)*(A1[i][j+1]-A1[i][j]+A1[i+1][j+1]-A1[i+1][j]+A2[i][j+1]-A2[i][j]+A2[i+1][j+1]-A2[i+1][j])
                mY[i-frame_width][j-frame_width] = (1/4)*(A1[i+1][j]-A1[i][j]+A1[i+1][j+1]-A1[i][j+1]+A2[i+1][j]-A2[i][j]+A2[i+1][j+1]-A2[i][j+1])
                mT[i-frame_width][j-frame_width] = (1/4)*(A2[i][j]-A1[i][j]+A2[i+1][j]-A1[i+1][j]+A2[i][j+1]-A1[i][j+1]+A2[i+1][j+1]-A1[i+1][j+1])
            else:
                if i == N+frame_width-1 and j == N+frame_width-1:
                    mX[i-frame_width][j-frame_width] = (1/4)*(A1[i-1][j]-A1[i-1][j-1]+A1[i][j]-A1[i][j-1]+A2[i-1][j]-A2[i-1][j-1]+A2[i][j]-A2[i][j-1])
                    mY[i-frame_width][j-frame_width] = (1/4)*(A1[i][j-1]-A1[i-1][j-1]+A1[i][j]-A1[i-1][j]+A2[i][j-1]-A2[i-1][j-1]+A2[i][j]-A2[i-1][j])
                    mT[i-frame_width][j-frame_width] = (1/4)*(A2[i-1][j-1]-A1[i-1][j-1]+A2[i][j-1]-A1[i][j-1]+A2[i-1][j]-A1[i-1][j]+A2[i][j]-A1[i][j])
                else:
                    if i == N+frame_width-1:
                        mX[i-frame_width][j-frame_width] = (1/4)*(A1[i-1][j+1]-A1[i-1][j]+A1[i][j+1]-A1[i][j]+A2[i-1][j+1]-A2[i-1][j]+A2[i][j+1]-A2[i][j])
                        mY[i-frame_width][j-frame_width] = (1/4)*(A1[i][j]-A1[i-1][j]+A1[i][j+1]-A1[i-1][j+1]+A2[i][j]-A2[i-1][j]+A2[i][j+1]-A2[i-1][j+1])
                        mT[i-frame_width][j-frame_width] = (1/4)*(A2[i-1][j]-A1[i-1][j]+A2[i][j]-A1[i][j]+A2[i-1][j+1]-A1[i-1][j+1]+A2[i][j+1]-A1[i][j+1])
                    if j == N+frame_width-1:
                        mX[i-frame_width][j-frame_width] = (1/4)*(A1[i][j]-A1[i][j-1]+A1[i+1][j]-A1[i+1][j-1]+A2[i][j]-A2[i][j-1]+A2[i+1][j]-A2[i+1][j-1])
                        mY[i-frame_width][j-frame_width] = (1/4)*(A1[i+1][j-1]-A1[i][j-1]+A1[i+1][j]-A1[i][j]+A2[i+1][j-1]-A2[i][j-1]+A2[i+1][j]-A2[i][j])
                        mT[i-frame_width][j-frame_width] = (1/4)*(A2[i][j-1]-A1[i][j-1]+A2[i+1][j-1]-A1[i+1][j-1]+A2[i][j]-A1[i][j]+A2[i+1][j]-A1[i+1][j])
    return mX, mY, mT

# вычисление диагонали матрицы H для пятиточечной аппроксимации лапласиана, 
# где N - размер входного изображения, mX - матрица из значений производной функции яркости по x, mY - матрица из значений производной функции яркости по y, alpha - параметр регуляризации по А. Н. Тихонову
def m_H_5(N, mX, mY, alpha):
    M = []
    for i in range(0, N):
        for j in range(0,N):
            M.append(4*alpha*alpha + mX[i, j]**2)
            M.append(4*alpha*alpha + mY[i, j]**2)
    return M

# вычисление диагонали матрицы H для девятиточечной аппроксимации лапласиана, 
# где N - размер входного изображения, mX - матрица из значений производной функции яркости по x, mY - матрица из значений производной функции яркости по y, alpha - параметр регуляризации по А. Н. Тихонову
def m_H_9(N, mX, mY, alpha):
    M = []
    for i in range(0, N):
        for j in range(0,N):
            M.append(alpha*alpha + mX[i, j]**2)
            M.append(alpha*alpha + mY[i, j]**2)
    return M

# вычсление вектора q для пятиточечной аппроксимации лапласиана, 
# где N - размер входного изображения, mX - матрица из значений производной функции яркости по x, mY - матрица из значений производной функции яркости по y, mT - матрица из значений производной функции яркости по t, u, v - искомые функции,  alpha - параметр регуляризации по А. Н. Тихонову
def q5(N, mX, mY, mT, u, v, alpha):
    q = []
    for i in range(0, N):
        for j in range(0, N):
            q.append([-mX[i,j]*mT[i,j]])
            q.append([-mY[i,j]*mT[i,j]])
            if i == 0:
                q[-2][0] += (alpha**2)*u[i][j+1]
                q[-1][0] += (alpha**2)*v[i][j+1]
            if i == N-1:
                q[-2][0] += (alpha**2)*u[-1][j+1]
                q[-1][0] += (alpha**2)*v[-i][j+1]
            if j == 0:
                q[-2][0] += (alpha**2)*u[i+1][j]
                q[-1][0] += (alpha**2)*v[i+1][j]
            if j == N-1:
                q[-2][0] += (alpha**2)*u[i+1][-1]
                q[-1][0] += (alpha**2)*v[i+1][-1]
    return np.array(q)

# вычсление вектора q для девятиточечной аппроксимации лапласиана,
# где N - размер входного изображения, mX - матрица из значений производной функции яркости по x, mY - матрица из значений производной функции яркости по y, mT - матрица из значений производной функции яркости по t, u, v - искомые функции,  alpha - параметр регуляризации по А. Н. Тихонову
def q9(N, mX, mY, mT, u, v, alpha):
    q = []
    for i in range(0, N):
        for j in range(0, N):
            q.append([-mX[i,j]*mT[i,j]])
            q.append([-mY[i,j]*mT[i,j]])
            if (i == 0 and j == 0) or (i == 0 and j == N-1) or (i == N-1 and j == 0) or (i == N-1 and j == N-1):
                q[-2][0] += (7/12)*(alpha**2)*u[i][j+1]
                q[-1][0] += (7/12)*(alpha**2)*v[i][j+1]
            elif i == 0 or j == 0 or i == N-1 or j == N-1:
                q[-2][0] += (1/3)*(alpha**2)*u[i][j+1]
                q[-1][0] += (1/3)*(alpha**2)*v[i][j+1]
    return np.array(q)

# метода Гаусса-Зейделя для пятиточечной аппроксимации лапласиана,
# где N - размер входного изображения, b - вектор q из Hz=q, H - диагональ матрицы H, alpha - параметр регуляризации по А. Н. Тихонову, B - диагональ матрицы B, accur - массив с полученными погрешностями, count - массив с количеством итераций, u, v - искомые функции
def Seidel_5(N, b, H, alpha, B, accur, count, u, v):
    x_0 = np.array(b.copy())
    x_1 = np.array(b.copy())
    for i in range(0, N**2):
        C = inverse_m(np.array([[H[2*i], B[i]], 
                           [B[i], H[2*i+1]]]))
        if i % N == 0:
            x_1[i*2:i*2+2] = C.dot(-np.array([[-x_1[(i+1)*2,0]*(alpha**2)], [-x_1[(i+1)*2+1,0]*(alpha**2)]]) + b[i*2:i*2+2])
        elif (i+1) % N == 0:
            x_1[i*2:i*2+2] = C.dot(-np.array([[-x_1[(i-1)*2,0]*(alpha**2)], [-x_1[(i-1)*2+1,0]*(alpha**2)]]) + b[i*2:i*2+2])
        else:
            x_1[i*2:i*2+2] = C.dot(-np.array([[-x_1[(i+1)*2,0]*(alpha**2)], [-x_1[(i+1)*2+1,0]*(alpha**2)]]) -np.array([[-x_1[(i-1)*2,0]*(alpha**2)], [-x_1[(i-1)*2+1,0]*(alpha**2)]]) + b[i*2:i*2+2])

        if i >= 0 and i <= N**2 - N - 1:
            x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(N+i)*2,0]*(alpha**2)], [-x_1[(N+i)*2+1,0]*(alpha**2)]]))
        if i >= N:
            x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-N)*2,0]*(alpha**2)], [-x_1[(i-N)*2+1,0]*(alpha**2)]]))

    n1 = x_1.copy()
    n1 = n1 - x_0
    x_0 = x_1.copy()
    k = 1
    while k <= count[-1]:
        if k in count:
            с = 0
            for i in range(1, N+2*frame_width-1):
                for j in range(1, N+2*frame_width-1):
                    u[i][j] = x_1[с, 0]
                    v[i][j] = x_1[с+1, 0]
                    с = с + 2
            angle = []
            for j in range(0, N):
                an = []
                for i in range(0, N):
                    an.append(0)
                angle.append(an)
            for i in range(frame_width, N+frame_width):
                for j in range(frame_width, N+frame_width):
                    angle[i-frame_width][j-frame_width] = abs(math.atan(-1)*(180/math.pi)-math.atan(v[i][j]/u[i][j])*(180/math.pi))
            sum = 0
            for j in range(0, N):
                for i in range(0, N):
                    sum += angle[i][j]
            accur.append(sum/((N)*(N)))
        print(" norm = ", norm(n1,np.inf), "\t k = ", k)
        for i in range(0, N**2):
            C = inverse_m(np.array([[H[2*i], B[i]], 
                                [B[i], H[2*i+1]]]))
            if i % N == 0:
                x_1[i*2:i*2+2] = C.dot(-np.array([[-x_1[(i+1)*2,0]*(alpha**2)], [-x_1[(i+1)*2+1,0]*(alpha**2)]]) + b[i*2:i*2+2])
            elif (i+1) % N == 0:
                x_1[i*2:i*2+2] = C.dot(-np.array([[-x_1[(i-1)*2,0]*(alpha**2)], [-x_1[(i-1)*2+1,0]*(alpha**2)]]) + b[i*2:i*2+2])
            else:
                x_1[i*2:i*2+2] = C.dot(-np.array([[-x_1[(i+1)*2,0]*(alpha**2)], [-x_1[(i+1)*2+1,0]*(alpha**2)]]) -np.array([[-x_1[(i-1)*2,0]*(alpha**2)], [-x_1[(i-1)*2+1,0]*(alpha**2)]]) + b[i*2:i*2+2])

            if i >= 0 and i <= N**2 - N - 1:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(N+i)*2,0]*(alpha**2)], [-x_1[(N+i)*2+1,0]*(alpha**2)]]))
            if i >= N:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-N)*2,0]*(alpha**2)], [-x_1[(i-N)*2+1,0]*(alpha**2)]]))

        n1 = x_1.copy()
        for t in range(0, len(x_1)):
            n1[t] = n1[t] - x_0[t]

        k += 1
        x_0 = x_1.copy()

    return x_1

# метода Гаусса-Зейделя для девятиточечной аппроксимации лапласиана,
# где N - размер входного изображения, b - вектор q из Hz=q, H - диагональ матрицы H, alpha - параметр регуляризации по А. Н. Тихонову, B - диагональ матрицы B, accur - массив с полученными погрешностями, count - массив с количеством итераций, u, v - искомые функции
def Seidel_9(N, b, H, alpha, B, accur, count, u, v):
    x_0 = np.array(b.copy())
    x_1 = np.array(b.copy())
    for i in range(0, N**2):
        x_1[i*2:i*2+2] = np.array([[0.0],[0.0]])
        C = inverse_m(np.array([[H[2*i], B[i]], 
                           [B[i], H[2*i+1]]]))
        if i < N**2-1:
            if i >= N**2-N:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i+1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+1)*2+1,0]*(1/6)*(alpha**2)]]))
            elif i % N == 0:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i+1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+1)*2+1,0]*(1/6)*(alpha**2)]])
                                                                        -np.array([[-x_1[(i+N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                        -np.array([[-x_1[(i+N+1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i+N+1)*2+1,0]*(1/12)*(alpha**2)]]))
            elif (i+1) % N == 0:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i+N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                        -np.array([[-x_1[(i+N-1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i+N-1)*2+1,0]*(1/12)*(alpha**2)]]))
            else:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i+1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+1)*2+1,0]*(1/6)*(alpha**2)]])
                                                                        -np.array([[-x_1[(i+N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                        -np.array([[-x_1[(i+N+1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i+N+1)*2+1,0]*(1/12)*(alpha**2)]])
                                                                        -np.array([[-x_1[(i+N-1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i+N-1)*2+1,0]*(1/12)*(alpha**2)]]))
        if i > 0:
            if i < N:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-1)*2+1,0]*(1/6)*(alpha**2)]]))
            elif i%N == 0:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                        -np.array([[-x_1[(i-N+1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i-N+1)*2+1,0]*(1/12)*(alpha**2)]]))
            elif (i+1)%N == 0:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-1)*2+1,0]*(1/6)*(alpha**2)]])
                                                                        -np.array([[-x_1[(i-N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                        -np.array([[-x_1[(i-N-1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i-N-1)*2+1,0]*(1/12)*(alpha**2)]]))
            else:
                x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-1)*2+1,0]*(1/6)*(alpha**2)]])
                                                                        -np.array([[-x_1[(i-N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                        -np.array([[-x_1[(i-N+1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i-N+1)*2+1,0]*(1/12)*(alpha**2)]])
                                                                        -np.array([[-x_1[(i-N-1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i-N-1)*2+1,0]*(1/12)*(alpha**2)]]))
        x_1[i*2:i*2+2] += C.dot(b[i*2:i*2+2])    
    n1 = x_1.copy()
    n1 = n1 - x_0
    x_0 = x_1.copy()
    k = 1
    while k <= count[-1]:
        if k in count:
            с = 0
            for i in range(1, N+2*frame_width-1):
                for j in range(1, N+2*frame_width-1):
                    u[i][j] = x_1[с, 0]
                    v[i][j] = x_1[с+1, 0]
                    с = с + 2
            angle = []
            for j in range(0, N):
                an = []
                for i in range(0, N):
                    an.append(0)
                angle.append(an)
            for i in range(frame_width, N+frame_width):
                for j in range(frame_width, N+frame_width):
                    angle[i-frame_width][j-frame_width] = abs(math.atan(-1)*(180/math.pi)-math.atan(v[i][j]/u[i][j])*(180/math.pi))
            sum = 0
            for j in range(0, N):
                for i in range(0, N):
                    sum += angle[i][j]
            accur.append(sum/((N)*(N)))
        print("norm = ", norm(n1,np.inf), "\t k = ", k)
        for i in range(0, N**2):
            x_1[i*2:i*2+2] = np.array([[0.0],[0.0]])
            C = inverse_m(np.array([[H[2*i], B[i]], 
                                [B[i], H[2*i+1]]]))
            x_1[i*2:i*2+2] += C.dot(b[i*2:i*2+2])  
            if i < N**2-1:
                if i >= N**2-N:
                    x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i+1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+1)*2+1,0]*(1/6)*(alpha**2)]]))
                elif i % N == 0:
                    x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i+1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+1)*2+1,0]*(1/6)*(alpha**2)]])
                                                                            -np.array([[-x_1[(i+N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                            -np.array([[-x_1[(i+N+1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i+N+1)*2+1,0]*(1/12)*(alpha**2)]]))
                elif (i+1) % N == 0:
                    x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i+N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                            -np.array([[-x_1[(i+N-1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i+N-1)*2+1,0]*(1/12)*(alpha**2)]]))
                else:
                    x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i+1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+1)*2+1,0]*(1/6)*(alpha**2)]])
                                                                            -np.array([[-x_1[(i+N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i+N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                            -np.array([[-x_1[(i+N+1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i+N+1)*2+1,0]*(1/12)*(alpha**2)]])
                                                                            -np.array([[-x_1[(i+N-1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i+N-1)*2+1,0]*(1/12)*(alpha**2)]]))
            if i > 0:
                if i < N:
                    x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-1)*2+1,0]*(1/6)*(alpha**2)]]))
                elif i%N == 0:
                    x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                            -np.array([[-x_1[(i-N+1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i-N+1)*2+1,0]*(1/12)*(alpha**2)]]))
                elif (i+1)%N == 0:
                    x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-1)*2+1,0]*(1/6)*(alpha**2)]])
                                                                            -np.array([[-x_1[(i-N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                            -np.array([[-x_1[(i-N-1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i-N-1)*2+1,0]*(1/12)*(alpha**2)]]))
                else:
                    x_1[i*2:i*2+2] += C.dot(-np.array([[-x_1[(i-1)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-1)*2+1,0]*(1/6)*(alpha**2)]])
                                                                            -np.array([[-x_1[(i-N)*2,0]*(1/6)*(alpha**2)],[-x_1[(i-N)*2+1,0]*(1/6)*(alpha**2)]]) 
                                                                            -np.array([[-x_1[(i-N+1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i-N+1)*2+1,0]*(1/12)*(alpha**2)]])
                                                                            -np.array([[-x_1[(i-N-1)*2,0]*(1/12)*(alpha**2)],[-x_1[(i-N-1)*2+1,0]*(1/12)*(alpha**2)]]))
            
        n1 = x_1.copy()
        for t in range(0, len(x_1)):
            n1[t] = n1[t] - x_0[t]

        k += 1
        x_0 = x_1.copy()

    return x_1

# вычисление поля скоростей,
# где A1, A2 - входные квадратные изображения, N - их размер, frame_width - ширина рамки, alpha - параметр регуляризации по А. Н. Тихонову, u_boundary, v_boundary - граничные условия, variant - переменная отвечающая за выбор аппроксимации лапласиана (5 - пятиточная, 9 - девятиточечная), , accur - массив с полученными погрешностями, count - массив с количеством итераций
def Optical_Flow(A1, A2, N, frame_width, alpha, u_boundary, v_boundary, variant, accur, count):
    u = []
    v = []
    for j in range(0, N + 2*frame_width):
        U = []
        V = []
        for i in range(0, N + 2*frame_width):
            U.append(u_boundary)
            V.append(v_boundary)
        u.append(U)
        v.append(V)
    mX, mY, mT = xyz_differences(A1, A2, N, frame_width)
    a = []
    for i in range(0, N):
        for j in range(0, N):
            a.append(mX[i][j]*mY[i][j])

    mX = np.array(mX)
    mY = np.array(mY)
    mT = np.array(mT)

    if variant == 5:
        q = q5(N, mX, mY, mT, u, v, alpha)
        M = m_H_5(N, mX, mY, alpha)
        vu = Seidel_5(N, q, M, alpha, a, accur, count, u, v)
    else:
        q = q9(N, mX, mY, mT, u, v, alpha)
        M = m_H_9(N, mX, mY, alpha)
        vu = Seidel_9(N, q, M, alpha, a, accur, count, u, v)

    k = 0
    for j in range(frame_width, N+frame_width):
        for i in range(frame_width, N+frame_width):
            u[i][j] = vu[k, 0]
            v[i][j] = vu[k+1, 0]
            k = k + 2
    return u, v

frame_width = 1 # ширина рамки
alpha = 1 # alpha - параметр регуляризации по А. Н. Тихонову
A1 = []
A2 = []
N = 100 # размер стороны входного изображения
shift = 1
A1 = random_img(N)
A2 = right_diag_shift(A1, N, shift)
A1 = frame(A1, N, frame_width)
A2 = frame(A2, N, frame_width)
plt.imshow(A1, cmap="Greys")
plt.show()
plt.imshow(A2, cmap="Greys")
plt.show()

step = N//10 # количество отображаемых векторов

A3 = [[255] * (N+2*frame_width)] * (N+2*frame_width)
A3 = Image.fromarray(np.array(A3))

accur = {5:[], 9:[]}
count = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850] # количество итераций

v = []
u = []
for i in [5, 9]:
    u, v = Optical_Flow(A1, A2, N, frame_width, alpha, 0, 0, i, accur[i], count)

    # построение поля скоростей (вектора не нормированы)
    ax = plt.figure().gca()
    ax.imshow(A3, cmap = 'gray')

    for i in range(0, N + 2*frame_width, step):
        for j in range(0, N + 2*frame_width, step):
            ax.arrow(i, j, step*u[i][j], step*v[i][j], length_includes_head = True, head_width = 0.15*step, overhang = 1, head_length = 0.2*step, width = 0.01)
    
    plt.draw()
    plt.show()

    # построение поля скоростей (вектора нормированы и их длина умножена на step для лучшей наглядности)
    ax = plt.figure().gca()
    ax.imshow(A3, cmap = 'gray')

    for i in range(0, N + 2*frame_width, step):
        for j in range(0, N + 2*frame_width, step):
            c = math.sqrt(u[i][j]**2+v[i][j]**2)
            if c != 0:
                ax.arrow(i, j, step*u[i][j]/c, step*v[i][j]/c, length_includes_head = True, head_width = 0.15*step, overhang = 1, head_length = 0.2*step, width = 0.01)
            else:
                ax.arrow(i, j, step*u[i][j], step*v[i][j], length_includes_head = True, head_width = 0.15*step, overhang = 1, head_length = 0.2*step, width = 0.01)
    
    plt.draw()            
    plt.show()

# построение графика зависимости точности от количества итераций
fig, ax = plt.subplots()

plt.plot(count, accur[5], color='r', label='Аппроксимация по 5 точкам')
plt.plot(count, accur[9], color='b', label='Аппроксимация по 9 точкам')
plt.xlabel('Количество итераций')
plt.legend()

plt.show()