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

# построение последовательности изображений,
# где N - размер стороны изображения, frame_width - ширина рамки, A - входное изображение, t - номер кадра, w - коэффициент в функции
def sequence(N, frame_width, A, t, w):
    A1 = []
    for i in range(0, N + 2*frame_width):
        A1.append([0]*(N + 2*frame_width))
    for i in range(0, N + 2*frame_width):
        for j in range(0, N + 2*frame_width):
            if A[j][i] == 1:
                A1[j+round(math.cos((np.pi/w)*t))][i+1] = 1
    return A1

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
A2 = [] # размер стороны изображения
N = 23
w = 10
A1 = []
for i in range(0, N + 2*frame_width):
    A1.append([0]*(N + 2*frame_width))

A1[7][3] = 1
A1[7][4] = 1
A1[7][5] = 1
A1[8][3] = 1
A1[8][4] = 1
A1[8][5] = 1
A1[9][3] = 1
A1[9][4] = 1
A1[9][5] = 1

A1[15][8] = 1

A0 = copy.deepcopy(A1)

plt.imshow(A1, cmap="Greys")
plt.show()

points = {0: {(3,7):(0,0), (3,8):(0,0), (3,9):(0,0), (8,15): (0,0)}}

for t in range(16):
    A2 = sequence(N, frame_width, A1, t+1, w)
    plt.imshow(A2, cmap="Greys")
    plt.show()
    A3 = copy.deepcopy(A1)
    for i in range(0, N + 2*frame_width):
        for j in range(0, N + 2*frame_width):
            if A1[j][i] == 1:
                A3[j][i] = 0
            else:
                A3[j][i] = 1

    v = []
    u = []
    u, v = Optical_Flow(A1, A2, N, frame_width, alpha, 0, 0)

    # построение поля скоростей (вектора не нормированы)
    ax = plt.figure().gca()
    ax.imshow(A3, cmap = 'gray')

    for (i, j) in points[t]:
        ax.arrow(i, j, u[j][i], v[j][i], length_includes_head = True, head_width = 0.15, overhang = 1, head_length = 0.2, width = 0.01, color = 'r')

    plt.draw()
    plt.show()

    # построение поля скоростей (вектора нормированы)
    ax = plt.figure().gca()
    ax.imshow(A3, cmap = 'gray')
    points[t+1] = {}
    for (i, j) in points[t]:
        points[t][(i,j)] = (u[j][i], v[j][i])
        c = math.sqrt(u[j][i]**2+v[j][i]**2)
        if c != 0:
            ax.arrow(i, j, u[j][i]/c, v[j][i]/c, length_includes_head = True, head_width = 0.15, overhang = 1, head_length = 0.2, width = 0.01, color = 'r')
        else:
            ax.arrow(i, j, u[j][i], v[j][i], length_includes_head = True, head_width = 0.15, overhang = 1, head_length = 0.2, width = 0.01, color = 'r')
        points[t+1][(round(i + u[j][i]/c), round(j + v[j][i]/c))] = (0,0)

    plt.draw()            
    plt.show()
    A1 = copy.deepcopy(A2)

A3 = [[255] * (N+2*frame_width)] * (N+2*frame_width)
A3 = Image.fromarray(np.array(A3))



ax = plt.figure().gca()
ax.imshow(A3, cmap = 'gray')

x1 = np.linspace(0, 25, 100)

for (x, y) in points[0]:
    y1 = lambda x1: y+np.sin((np.pi/w)*(x1-x))/(np.pi/w)
    ax.plot(x1, y1(x1), color = 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.show()

ax = plt.figure().gca()
ax.imshow(A3, cmap = 'gray')
for t in points:
    for (x, y) in points[t]:
        (u, v) = points[t][(x,y)]
        c = math.sqrt(u**2+v**2)
        if c > 0:
            ax.arrow(x, y, u/c, v/c, length_includes_head = True, head_width = 0.15, overhang = 1, head_length = 0.2, width = 0.02, color = 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.show()
