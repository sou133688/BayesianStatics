import numpy as np
import matplotlib.pyplot as plt
import math
import sys

#dim_data次多項式にノイズを与えた点(X,noised_Y)の集合を出力する関数
def give_data(coefficient,number_of_points,dim_data,variance_of_noise,xmin=-1.5,xmax=1.5):

    X = np.zeros(number_of_points)
    Y = np.zeros(number_of_points)
    noised_Y = np.zeros(number_of_points)

    for i in range(0,number_of_points):
        X[i] = np.random.uniform(xmin,xmax)

        for j in range(0,dim_data+1):
            Y[i] = Y[i] + coefficient[j]*X[i]**j

        noised_Y[i] = np.random.normal(loc=Y[i],scale=variance_of_noise)

    return X,noised_Y


def predict(dim_predict, dim_data, coefficient, expectation_vector, precision_matrix,number_of_data,variance_of_noise):
    
    #dim_data次多項式にノイズを与えた点(X,noised_Y)の集合を格納する。
    X,Y = give_data(coefficient,number_of_data,dim_data,variance_of_noise)

    #更新後のパラメータを計算するための準備
    input_vec = np.zeros((number_of_data,dim_predict+1))

    Sum1 = np.zeros((dim_predict+1,dim_predict+1))
    Sum2 = np.c_[np.zeros(dim_predict+1)] #縦ベクトルにしておく

    for j in range(0,number_of_data):
        #各xについてinput_vector = (1,x,x^2,x^3,x^4,…)をつくる。
        for i in range(0,dim_predict+1):
            input_vec[j][i] = X[j]**i

        #new_precision_matrix,new_expectation_vectorに必要なSum1とSum2を求める。
        Sum1 = Sum1 + np.dot(input_vec[[j]].transpose(),input_vec[[j]])
        Sum2 = Sum2 + Y[j]*input_vec[[j]].transpose()

    #学習後のパラメータ
    new_precision_matrix = variance_of_noise * Sum1 + precision_matrix
    new_expectation_vector = np.dot(np.linalg.inv(new_precision_matrix),(variance_of_noise*Sum2 + np.dot(precision_matrix,np.c_[expectation_vector]))) #縦ベクトル

    #predicted functionをpredict_y[x]に格納
    xmin_predicted = -1.5
    xmax_predicted = +1.5

    predict_x = np.linspace(xmin_predicted, xmax_predicted, 100)
    predict_y = np.zeros(100)
    for j in range(0,100):
        for i in range(0,dim_predict+1):
            predict_y[j] = predict_y[j] + new_expectation_vector[i]*predict_x[j]**i

    #predicted function predict_y[x]と入力データ(X,Y)を同一グラフにプロット
    plt.scatter(X,Y,label="input data")
    plt.plot(predict_x,predict_y,c="red",label="predicted y")
    plt.legend()
    plt.title("number of data="+str(number_of_data)+" variance of noise="+str(variance_of_noise))
    plt.xlabel("x",fontsize=16,fontweight='bold')
    plt.ylabel("y",fontsize=16,fontweight='bold')
    plt.xlim([xmin_predicted,xmax_predicted])
    plt.ylim([0,3.5])
    plt.grid(True)
    plt.show()

    return new_precision_matrix,new_expectation_vector

def probability(dim_predict,variance_of_noise,new_precision_matrix,new_expectation_vector):
    
    xmin_probability_density = -1.5
    xmax_probability_density = +1.5

    N = 30 #カラープロットの細かさ
    x = np.linspace(xmin_probability_density,xmax_probability_density,N)
    y = np.linspace(0,3.5,N)
    probability = np.zeros((N,N))

    for i in range(0,N):
        predicted_input_vector = np.zeros(dim_predict+1)

        #predicted_input_vector = (1,x,x^2,x^3,x^4,…)をつくる。
        for k in range(0,dim_predict+1):
            predicted_input_vector[k] = x[i]**k 

        expectation = np.dot(new_expectation_vector.transpose(),predicted_input_vector.transpose())
        precision = 1.0/variance_of_noise + predicted_input_vector @ np.linalg.inv(new_precision_matrix) @ predicted_input_vector.transpose()

        for j in range(0,N):
            probability[j][i] = normal_distribution(y[j],expectation,precision)

    #確率密度関数をカラープロット
    x, y = np.meshgrid(x, y) #グリッドをつくる
    plt.pcolor(x,y,probability) 
    cbar = plt.colorbar(ticks=[]) #カラーバーのticksをかかない。
    cbar.set_label('Probability', fontsize=16,fontweight='bold')
    plt.title("number of data="+str(number_of_data)+" variance of noise="+str(variance_of_noise))
    plt.clim(0,100) #カラーバーの範囲
    plt.xlabel("x",fontsize=16,fontweight='bold')
    plt.ylabel("y",fontsize=16,fontweight='bold')
    plt.xlim([xmin_probability_density,xmax_probability_density])
    plt.ylim([0,3.5])
    plt.show()

def normal_distribution(x,expectation,precision):
    
    convariance = 1.0/precision

    return 1.0/(math.sqrt(2.0*math.pi)*convariance**2)*math.exp(-(x-expectation)**2/(2.0*convariance**2))

if __name__ == "__main__":
    
    #次元を指定。dim_data次元の正解データをdim_predict次元の関数でfittingする。
    dim_predict = 4
    dim_data = 4

    #正解のdim_data次元関数の係数。dim_data + 1 要素必要。
    coefficient = [2,0.5,-2,0,1] #左から0次,1次,2次,3次,4次

    #データ(X,noised_Y)を与えるパラメータ
    number_of_data = 1000 #データの数
    variance_of_noise = 0.1 #ノイズの大きさ

    #coefficientの要素数がdim_data + 1でない場合は終了。
    if len(coefficient) != dim_data + 1:
        print("Error. Number of coefficients are wrong.")
        sys.exit()

    #ベイズ推定のHyper parameters
    expectation_vector = np.ones(dim_predict+1)
    precision_matrix = np.zeros(dim_predict+1)

    new_precision_matrix,new_expectation_vector= predict(dim_predict, dim_data, coefficient, expectation_vector, precision_matrix,number_of_data,variance_of_noise)
    probability(dim_predict, variance_of_noise,new_precision_matrix, new_expectation_vector)