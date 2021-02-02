import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import scipy
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

df = pd.DataFrame()

probit = [
    [0.01,2.67],
[0.02,2.95],
[0.03,3.12],
[0.04,3.25],
[0.05,3.36],
[0.06,3.45],
[0.07,3.52],
[0.08,3.59],
[0.09,3.66],
[0.1,3.72],
[0.11,3.77],
[0.12,3.82],
[0.13,3.87],
[0.14,3.92],
[0.15,3.96],
[0.16,4.01],
[0.17,4.05],
[0.18,4.08],
[0.19,4.12],
[0.2,4.16],
[0.21,4.19],
[0.22,4.23],
[0.23,4.26],
[0.24,4.29],
[0.25,4.33],
[0.26,4.36],
[0.27,4.39],
[0.28,4.42],
[0.29,4.45],
[0.3,4.48],
[0.31,4.5],
[0.32,4.53],
[0.33,4.56],
[0.34,4.59],
[0.35,4.61],
[0.36,4.64],
[0.37,4.67],
[0.38,4.69],
[0.39,4.72],
[0.4,4.75],
[0.41,4.77],
[0.42,4.8],
[0.43,4.82],
[0.44,4.85],
[0.45,4.87],
[0.46,4.9],
[0.47,4.92],
[0.48,4.95],
[0.49,4.97],
[0.5,5],
[0.51,5.03],
[0.52,5.05],
[0.53,5.08],
[0.54,5.1],
[0.55,5.13],
[0.56,5.15],
[0.57,5.18],
[0.58,5.2],
[0.59,5.23],
[0.6,5.25],
[0.61,5.28],
[0.62,5.31],
[0.63,5.33],
[0.64,5.36],
[0.65,5.39],
[0.66,5.41],
[0.67,5.44],
[0.68,5.47],
[0.69,5.5],
[0.7,5.52],
[0.71,5.55],
[0.72,5.58],
[0.73,5.61],
[0.74,5.64],
[0.75,5.67],
[0.76,5.71],
[0.77,5.74],
[0.78,5.77],
[0.79,5.81],
[0.8,5.84],
[0.81,5.88],
[0.82,5.92],
[0.83,5.95],
[0.84,5.99],
[0.85,6.04],
[0.86,6.08],
[0.87,6.13],
[0.88,6.18],
[0.89,6.23],
[0.9,6.28],
[0.91,6.34],
[0.92,6.41],
[0.93,6.48],
[0.94,6.55],
[0.95,6.64],
[0.96,6.75],
[0.97,6.88],
[0.98,7.05],
[0.99,7.33]]
probit = pd.DataFrame(probit)
probit.columns=['freq','probit']

logit = [
    [0.01,-4.5951],
[0.02,-3.8918],
[0.03,-3.4761],
[0.04,-3.1781],
[0.05,-2.9444],
[0.06,-2.7515],
[0.07,-2.5867],
[0.08,-2.4423],
[0.09,-2.3136],
[0.1,-2.1972],
[0.11,-2.0907],
[0.12,-1.9924],
[0.13,-1.901],
[0.14,-1.8153],
[0.15,-1.7346],
[0.16,-1.6582],
[0.17,-1.5856],
[0.18,-1.5163],
[0.19,-1.45],
[0.2,-1.3863],
[0.21,-1.3249],
[0.22,-1.2657],
[0.23,-1.2083],
[0.24,-1.1527],
[0.25,-1.0986],
[0.26,-1.046],
[0.27,-0.9946],
[0.28,-0.9445],
[0.29,-0.8954],
[0.3,-0.8473],
[0.31,-0.8001],
[0.32,-0.7538],
[0.33,-0.7082],
[0.34,-0.6633],
[0.35,-0.619],
[0.36,-0.5754],
[0.37,-0.5322],
[0.38,-0.4895],
[0.39,-0.4473],
[0.4,-0.4055],
[0.41,-0.364],
[0.42,-0.3228],
[0.43,-0.2819],
[0.44,-0.2412],
[0.45,-0.2007],
[0.46,-0.1603],
[0.47,-0.1201],
[0.48,-0.08],
[0.49,-0.04],
[0.5,0],
[0.51,0.04],
[0.52,0.08],
[0.53,0.1201],
[0.54,0.1603],
[0.55,0.2007],
[0.56,0.2412],
[0.57,0.2819],
[0.58,0.3228],
[0.59,0.364],
[0.6,0.4055],
[0.61,0.4473],
[0.62,0.4895],
[0.63,0.5322],
[0.64,0.5754],
[0.65,0.619],
[0.66,0.6633],
[0.67,0.7082],
[0.68,0.7538],
[0.69,0.8001],
[0.7,0.8473],
[0.71,0.8954],
[0.72,0.9445],
[0.73,0.9946],
[0.74,1.046],
[0.75,1.0986],
[0.76,1.1527],
[0.77,1.2083],
[0.78,1.2657],
[0.79,1.3249],
[0.8,1.3863],
[0.81,1.45],
[0.82,1.5163],
[0.83,1.5856],
[0.84,1.6582],
[0.85,1.7346],
[0.86,1.8153],
[0.87,1.901],
[0.88,1.9924],
[0.89,2.0907],
[0.9,2.1972],
[0.91,2.3136],
[0.92,2.4423],
[0.93,2.5867],
[0.94,2.7515],
[0.95,2.9444],
[0.96,3.1781],
[0.97,3.4761],
[0.98,3.8918],
[0.99,4.5951]]
logit = pd.DataFrame(logit)
logit.columns=['freq','logit']

def get_column_names(df):
    for column in df.columns:
        listbox.insert(END, str(column))
        listbox.pack()

def read_data():
    global df
    file_path = askopenfilename()
    df = pd.read_excel(file_path)
    text.pack()
    
    for column in df.columns:
        listbox.insert(END, str(column))
    listbox.pack()
    button_ok.pack()
    
def print_col(l):
    print(l)
    
def read_df():
    text.pack_forget()
    button.pack_forget()
    button_ok.pack_forget()
    listbox.pack_forget()
    global data
    data = [listbox.get(idx) for idx in listbox.curselection()]
    data = df.loc[:, data]
    text2.pack()
    text_input.pack()
    button_input_ok.pack()
    
def get_logit(row):
    res = logit.loc[logit['freq'] == row['Częstosc p']]
    return res['logit'].iloc[0]

def get_probit(row):
    res = probit.loc[probit['freq'] == row['Częstosc p']]
    return res['probit'].iloc[0]

def min_sqr_logit(result):
    global poly_logit
    x = pd.DataFrame()
    x['sr'] = result['srodek przedzialu']
    x['1'] = 1
    x = x[['1', 'sr']]
    x = x.to_numpy().tolist()
    x = np.matrix(x)
    print(x)
    xt = []
    xt.append([1 for x in range(len(result))])
    xt.append([result['srodek przedzialu'].iloc[x] for x in range(len(result))])
    xt = np.matrix(xt)
    print(xt)
    xtx = np.matmul(xt, x)
    print(xtx)
    xtx_inv = np.linalg.inv(xtx)
    print(xtx_inv)
    logit = [result['logit'].iloc[x] for x in range(len(result))]
    xty = np.matmul(xt, logit)
    xty = np.array(xty).ravel()
    alfa = np.matmul(xtx_inv, xty)
    print(alfa)
    x = result['srodek przedzialu']
    y = result['logit']
    coef = np.polyfit(x,y,1)
    poly_logit = np.poly1d(coef)
    print(poly_logit)
    df = pd.DataFrame(x)
    df.columns = ['xj']
    df['lj'] = poly_logit[1] * df['xj'] + poly_logit[0]
    e = 2.7182
    df['wiek'] = 36 * df['xj']
    df['pj'] = (1) / (1+e**-(poly_logit[0]-(-poly_logit[1])*df['xj']))
    df['czestosc'] = result['Częstosc p']
    print(df)
    text_logit.insert(INSERT, "Metoda najmniejszych kwadratow (logit):" + "\n")
    text_logit.insert(INSERT, "XTY:" + "\n" + str(xty) + "\n")
    text_logit.insert(INSERT, "Alfa:" + "\n" + str(alfa) + "\n")
    text_logit.insert(INSERT, "Model:"  + str(poly_logit) + "\n")
    text_logit_prob.insert(INSERT, "Prawdopodobieństwa:\n" + str(df))
    text_logit.pack()
    text_logit_prob.pack()
    
def fert():
    text_plodnosc.insert(INSERT, "Podaj szanse plodnosci w %")
    text_plodnosc.pack()
    text_plodnosc_input.insert(INSERT, "")
    text_plodnosc_input.pack()
    button_plodnosc_ok.pack()
    
def fert_calc():
    global g_srodki_przedzialow
    global g_logit
    global g_probit
    input_value = float(text_plodnosc_input.get("1.0",END))
    input_value = round((input_value / 100), 2)
    age = logit.loc[logit['freq'] == input_value]
    age = age['logit'].iloc[0]
    fert_logit = age + -(poly_logit[0])
    print(poly_logit[0])
    fert_logit = fert_logit / poly_logit[1]
    print(fert_logit)
    fert_logit_age = 36 * fert_logit
    print(fert_logit_age)
    x_logit = []
    g_srodki_przedzialow = g_srodki_przedzialow.tolist()
    g_logit = g_logit.tolist()
    x_logit.append(g_srodki_przedzialow)
    x_logit.append(g_logit)
    corr_matrix = np.corrcoef(x_logit[0], x_logit[1])
    corr_xy = corr_matrix[0,1]
    r2_logit = corr_xy**2
    print("r2: " + str(r2_logit))
    text_zad2_logit.insert(INSERT, "LOGIT\nWiek, przy ktorym szansa plodnosci wynoszaca jest " + \
                           str(int(text_plodnosc_input.get("1.0",END))) + "%:" + str(fert_logit_age))
    text_zad2_logit.pack()
    text_r2_logit.insert(INSERT, "R^2=" + str(r2_logit))
    text_r2_logit.pack()
    
    input_value = float(text_plodnosc_input.get("1.0",END))
    input_value = round((input_value / 100), 2)
    age = probit.loc[probit['freq'] == input_value]
    age = age['probit'].iloc[0]
    fert_probit = age + -(poly_probit[0])
    print(poly_logit[0])
    fert_probit = fert_probit / poly_probit[1]
    print(fert_probit)
    fert_probit_age = 36 * fert_probit
    print(fert_probit_age)
    x_probit = []
    g_probit = g_probit.tolist()
    x_probit.append(g_srodki_przedzialow)
    x_probit.append(g_probit)
    corr_matrix = np.corrcoef(x_probit[0], x_probit[1])
    corr_xy = corr_matrix[0,1]
    r2_probit = corr_xy**2
    print("r2: " + str(r2_probit))
    text_zad2_probit.insert(INSERT, "PROBIT\nWiek, przy ktorym szansa plodnosci wynoszaca jest " + \
                           str(int(text_plodnosc_input.get("1.0",END))) + "%:" + str(fert_probit_age))
    text_zad2_probit.pack()
    text_r2_probit.insert(INSERT, "R^2=" + str(r2_probit))
    text_r2_probit.pack()
    
def min_sqr_probit(result):
    global poly_probit
    x = pd.DataFrame()
    x['sr'] = result['srodek przedzialu']
    x['1'] = 1
    x = x[['1', 'sr']]
    x = x.to_numpy().tolist()
    x = np.matrix(x)
    print(x)
    xt = []
    xt.append([1 for x in range(len(result))])
    xt.append([result['srodek przedzialu'].iloc[x] for x in range(len(result))])
    xt = np.matrix(xt)
    print(xt)
    xtx = np.matmul(xt, x)
    print(xtx)
    xtx_inv = np.linalg.inv(xtx)
    print(xtx_inv)
    probit = [result['probit'].iloc[x] for x in range(len(result))]
    xty = np.matmul(xt, probit)
    xty = np.array(xty).ravel()
    alfa = np.matmul(xtx_inv, xty)
    print(alfa)
    x = result['srodek przedzialu']
    y = result['probit']
    coef = np.polyfit(x,y,1)
    poly_probit = np.poly1d(coef)
    print(poly_probit)
    a1 = [poly_probit[1] for x in range(len(result))]
    df = pd.DataFrame(a1)
    df.columns = ['a1']
    df['x1'] = result['srodek przedzialu']
    #df['pj'] = 
    f = poly_probit[0] + poly_probit[1] * df['x1'] - 5
    f = f.to_numpy().tolist()
    df['wiek'] = 36 * df['x1']
    df['pj'] = scipy.stats.norm.cdf(f)
    print(f)
    print(df)
    text_probit.insert(INSERT, "Metoda najmniejszych kwadratow (probit):" + "\n")
    text_probit.insert(INSERT, "XTY:" + "\n" + str(xty) + "\n")
    text_probit.insert(INSERT, "Alfa:" + "\n" + str(alfa) + "\n")
    text_probit.insert(INSERT, "Model:"  + str(poly_probit) + "\n")
    text_probit_prob.insert(INSERT, "Prawdopodobieństwa:\n" + str(df))
    text_probit.pack()
    text_probit_prob.pack()

def min_sqr(result):
    def matmult(a,b):
        zip_b = zip(*b)
        zip_b = list(zip_b)
        return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]
    
    min_sqr_logit(result)
    min_sqr_probit(result)
    fert()
    
def transf_logit():
    global g_srodki_przedzialow
    global g_logit
    global g_probit
    button_input_ok.pack_forget()
    text_input.pack_forget()
    text2.pack_forget()
    col = [x for x in data._get_numeric_data().columns]
    col = col[0]
    col_max = max(data[col])
    col_min = min(data[col])
    input_value = int(text_input.get("1.0",END))
    diff = (col_max - col_min) / input_value
    result = []
    srodki = []
    for x in range(input_value):
        if x == input_value-1:
            count_df = data.loc[data[col] > col_min + float(diff *x)]
        else:
            count_df = data.loc[(data[col] >= col_min + float(diff *x)) & (data[col] < col_min + float(diff *(x+1)))]
        result.append([col_min + float(diff * x),
                       col_min + float(diff * (x+1)),
                       len(count_df),
                       len(count_df.loc[count_df['diagnoza'] == 'N']),
                       (len(count_df.loc[count_df['diagnoza'] == 'N'])) / (len(count_df)) ])
        srodki.append([np.mean([col_min + float(diff * x), col_min + float(diff * (x+1))])])
    result = pd.DataFrame(result)
    srodki = pd.DataFrame(srodki)
    srodki.columns = ['Częstosc p']
    srodki['Częstosc p'] = round(srodki['Częstosc p'], 2)
    result.columns = [col + ' przedział od', col + ' przedział do', 'Liczba badanych (N)', 'Liczba osób z normalną diagnozą(n)', 'Częstosc p']
    result['Częstosc p'] = round(result['Częstosc p'], 2)
    result['logit'] = result.apply(get_logit, axis=1)
    result['probit'] = result.apply(get_probit, axis=1)
    result['srodek przedzialu'] = round((result[col + ' przedział od'] + result[col + ' przedział do'])/2,3)
    srodki['logit'] = srodki.apply(get_logit, axis=1)
    print(result.head())
    text_result.insert(INSERT, str(result))
    text_result.pack()
    x = result['srodek przedzialu']
    y = result['logit']
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef)
    print(poly1d_fn)
    fig = plt.figure(figsize=(3,2), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(x,y, 'yo', x, poly1d_fn(x), '--k')
    chart_type = FigureCanvasTkAgg(fig, root)
    chart_type.get_tk_widget().pack()
    
    y = result['probit']
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef)
    fig2 = plt.figure(figsize=(3,2), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.plot(x,y, 'yo', x, poly1d_fn(x), '--k')
    chart_type2 = FigureCanvasTkAgg(fig2, root)
    chart_type2.get_tk_widget().pack()
    g_srodki_przedzialow = result['srodek przedzialu'].values
    g_logit = result['logit'].values
    g_probit = result['probit'].values
    min_sqr(result)
        
    
root = tk.Tk()
root.geometry("900x1000")
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, 
                   text="Wczytaj dane", 
                   fg="red",
                   command=read_data)
button.pack(side=tk.LEFT)

listbox = Listbox(root, selectmode = "multiple")

text = Text(root, height=1)
text.insert(INSERT, "Wybierz DWIE kolumny do zaimportowania")

button_ok = tk.Button(root, 
                   text="OK", 
                   fg="red",
                   command=read_df)

text2 = Text(root, height=1)
text2.insert(INSERT, "Wybierz liczbę przedziałów:")

text_input = Text(root, height=1, width=5)
text_input.insert(INSERT, "")

button_input_ok = tk.Button(root, 
                   text="OK", 
                   fg="red",
                   command=transf_logit)

text_result = Text(root, height=6, width=100)
text_logit = Text(root, height=7, width=100)
text_probit = Text(root, height=7, width=100)
text_logit_prob = Text(root, height=6, width=100)
text_probit_prob = Text(root, height=6, width=100)

text_plodnosc = Text(root, height=1)
text_plodnosc_input = Text(root, height=1, width=5)
button_plodnosc_ok = tk.Button(root, 
                   text="OK", 
                   fg="red",
                   command=fert_calc)
text_zad2_logit = Text(root, height=3)
text_zad2_probit = Text(root, height=3)

text_r2_logit = Text(root, height=1)
text_r2_probit = Text(root, height=1)

root.mainloop()