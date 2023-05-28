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

# метода Гаусса-Зейделя для пятиточечной аппроксимации лапласиана,
# где N - размер входного изображения, b - вектор q из Hz=q, H - диагональ матрицы H, alpha - параметр регуляризации по А. Н. Тихонову, B - диагональ матрицы B
def Seidel_5(N, b, H, alpha, B):
    x_0 = np.array(b.copy())
    x_1 = np.array(b.copy())
    eps = 10**(-6)
    print(eps)
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
    while norm(n1,np.inf) > eps:
        print(norm(n1,np.inf) - eps, "\t norm = ", norm(n1,np.inf), "\t k = ", k)
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

# вычисление поля скоростей,
# где A1, A2 - входные квадратные изображения, N - их размер, frame_width - ширина рамки, alpha - параметр регуляризации по А. Н. Тихонову, u_boundary, v_boundary - граничные условия
def Optical_Flow(A1, A2, N, frame_width, alpha, u_boundary, v_boundary):
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

    q = q5(N, mX, mY, mT, u, v, alpha)
    M = m_H_5(N, mX, mY, alpha)
    vu = Seidel_5(N, q, M, alpha, a)
       
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
N = 100 # размер стороны изображения
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

accur = {(0,0): [], (1, -1): [], (1, 0): []}
frame_w = []

for (u_b, v_b) in [(0.0,0.0), (1.0, -1.0), (1.0, 0.0)]:
    v = []
    u = []
    u, v = Optical_Flow(A1, A2, N, frame_width, alpha, u_b, v_b)

    angle = []
    for i in range(frame_width, N + frame_width):
        an = []
        for j in range(frame_width, N + frame_width):
            an.append(abs(math.atan(-1)*(180/math.pi)-math.atan(v[i][j]/u[i][j])*(180/math.pi)))
        angle.append(an)

    frame_w = []
    for t in range(0, N//2-1):
        sum = 0
        for j in range(t, N-t):
            for i in range(t, N-t):
                sum += angle[i][j]
        accur[(int(u_b), int(v_b))].append(sum/((N-t)**2))
        frame_w.append(t+1)

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

fig, ax = plt.subplots()

plt.plot(frame_w, accur[(0,0)], color='r', label='граничные условия 0, 0')
plt.plot(frame_w, accur[(1, -1)], color='b', label='граничные условия 1, -1')
plt.plot(frame_w, accur[(1, 0)], color='g', label='граничные условия 1, 0')
plt.xlabel('Расстояние от границы')
plt.legend()

plt.show()