import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def michaelis_menten(x, DC, Vmax, n, Km):
    return DC + Vmax * (x**n) / (x**n + Km**n)


# Título do aplicativo
st.title('Avaliação da Aline')

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    # Ler o arquivo CSV usando pandas
    df = pd.read_csv(uploaded_file)

    # Selecionar as colunas 3 a 6 (pandas é baseado em índice 0, então colunas 2 a 5)
    time = df.iloc[:, 2]
    x = df.iloc[:, 3]
    y = df.iloc[:, 4]
    z = df.iloc[:, 5]

    rms_curve_x = []
    tempo_rms_x = []
    for index, valor in enumerate(x):
        tempo_rms_x.append(index)
        rms_curve_x.append(np.sqrt(np.mean(np.square(x[index:index+200]))))

    rms_curve_y = []
    tempo_rms_y = []
    for index, valor in enumerate(y):
        tempo_rms_y.append(index)
        rms_curve_y.append(np.sqrt(np.mean(np.square(y[index:index+200]))))

    rms_curve_z = []
    tempo_rms_z = []
    for index, valor in enumerate(z):
        tempo_rms_z.append(index)
        rms_curve_z.append(np.sqrt(np.mean(np.square(z[index:index+200]))))

    min = np.min(rms_curve_x[0])
    max = np.max(rms_curve_x)

    contador = 0
    for index, valor in enumerate(rms_curve_x):
        if valor > min + max*0.01:
            contador = contador + 1
            if contador > 2:
                start = index
                break

    # modelo do começo
    x_data2 = tempo_rms_x[start:start+1500]
    y_data2 = rms_curve_x[start:start+1500]
    # Ajustando a função de Michaelis-Menten aos dados
    # Suposições iniciais para os parâmetros Vmax e Km
    initial_guess = [1, 3, 1, 3]
    params, covariance = curve_fit(
        michaelis_menten, x_data2, y_data2, p0=initial_guess)

    # Obtendo os parâmetros ajustados
    DC_fit2, Vmax_fit2, n_fit2, Km_fit2 = params

    # Gerando curva ajustada
    y_fit2 = michaelis_menten(x_data2, DC_fit2, Vmax_fit2, n_fit2, Km_fit2)

    amplitude_modelo_x = np.max(y_fit2) - y_fit2[0]
    A50_x = amplitude_modelo_x/2 + y_fit2[0]
    A95_x = (amplitude_modelo_x*0.95) + y_fit2[0]

    for index, valor in enumerate(y_fit2):
        if valor > A50_x:
            t50_modelo_x = tempo_rms_x[start+index]
            break

    for index, valor in enumerate(y_fit2):
        if valor > A95_x:
            t90_modelo_x = tempo_rms_x[start+index]
            break
    t50_x = t50_modelo_x - tempo_rms_x[start]
    t90_x = t90_modelo_x - tempo_rms_x[start]

    # modelo do começo
    x_data2_y = tempo_rms_y[start:start+1500]
    y_data2_y = rms_curve_y[start:start+1500]
    # Ajustando a função de Michaelis-Menten aos dados
    # Suposições iniciais para os parâmetros Vmax e Km
    initial_guess = [1, 3, 1, 3]
    params, covariance = curve_fit(
        michaelis_menten, x_data2_y, y_data2_y, p0=initial_guess)

    # Obtendo os parâmetros ajustados
    DC_fit2, Vmax_fit2, n_fit2, Km_fit2 = params

    # Gerando curva ajustada
    y_fit2_y = michaelis_menten(x_data2_y, DC_fit2, Vmax_fit2, n_fit2, Km_fit2)

    amplitude_modelo_y = np.max(y_fit2_y) - y_fit2_y[0]
    A50_y = amplitude_modelo_y/2 + y_fit2_y[0]
    A95_y = (amplitude_modelo_y*0.95) + y_fit2_y[0]

    for index, valor in enumerate(y_fit2_y):
        if valor > A50_y:
            t50_modelo_y = tempo_rms_y[start+index]
            break

    for index, valor in enumerate(y_fit2_y):
        if valor > A95_y:
            t95_modelo_y = tempo_rms_y[start+index]
            break
    t50_y = t50_modelo_y - tempo_rms_y[start]
    t95_y = t95_modelo_y - tempo_rms_y[start]

    # modelo do começo
    x_data2_z = tempo_rms_z[start:start+1500]
    y_data2_z = rms_curve_z[start:start+1500]
    # Ajustando a função de Michaelis-Menten aos dados
    # Suposições iniciais para os parâmetros Vmax e Km
    initial_guess = [0.25, 2, 1, 3]
    params, covariance = curve_fit(
        michaelis_menten, x_data2_z, y_data2_z, p0=initial_guess)

    # Obtendo os parâmetros ajustados
    DC_fit2, Vmax_fit2, n_fit2, Km_fit2 = params

    # Gerando curva ajustada
    y_fit2_z = michaelis_menten(x_data2_z, DC_fit2, Vmax_fit2, n_fit2, Km_fit2)

    amplitude_modelo_z = np.max(y_fit2_z) - y_fit2_z[0]
    A50_z = amplitude_modelo_z/2 + y_fit2_z[0]
    print(A50_z)
    A95_z = (amplitude_modelo_z*0.95) + y_fit2_z[0]
    print(A95_z)
    for index, valor in enumerate(y_fit2_z):
        if valor > A50_z:
            t50_modelo_z = tempo_rms_z[start+index]
            break

    for index, valor in enumerate(y_fit2_z):
        if valor > A95_z:
            t95_modelo_z = tempo_rms_z[start+index]
            break
    t50_z = t50_modelo_z - tempo_rms_z[start]
    t95_z = t95_modelo_z - tempo_rms_z[start]

    col1, col2 = st.columns(2)
    with col1:

        fig, ax = plt.subplots()
        ax.plot(tempo_rms_x, rms_curve_x, 'k')
        ax.plot(x_data2, y_fit2, 'r')
        # ax.plot([tempo_rms_x[start], tempo_rms_x[start]], [0, 5], 'r--')
        # ax.plot([tempo_rms_x[start+1500], tempo_rms_x[start+1500]], [0, 5], 'r--')
        ax.plot(t50_modelo_x, A50_x, 'yo', markersize=10)
        ax.plot(t90_modelo_x, A95_x, 'yo', markersize=10)
        ax.set_ylim([0, np.max(rms_curve_x)])
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Aceleração vertical')
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(tempo_rms_y, rms_curve_y, 'k')
        ax.plot(x_data2_y, y_fit2_y, 'r')
        # ax.plot([tempo_rms_y[start], tempo_rms_y[start]], [0, 5], 'r--')
        # ax.plot([tempo_rms_y[start+1500], tempo_rms_y[start+1500]], [0, 5], 'r--')
        ax.plot(t50_modelo_y, A50_y, 'yo', markersize=10)
        ax.plot(t95_modelo_y, A95_y, 'yo', markersize=10)
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Aceleração laterolateral')
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(tempo_rms_z, rms_curve_z, 'k')
        ax.plot(x_data2_z, y_fit2_z, 'r')
        ax.plot(t50_modelo_z, A50_z, 'yo', markersize=10)
        ax.plot(t95_modelo_z, A95_z, 'yo', markersize=10)
        # ax.plot([tempo_rms_z[start], tempo_rms_z[start]], [0, 5], 'r--')
        # ax.plot([tempo_rms_z[start+1500], tempo_rms_z[start+1500]], [0, 5], 'r--')
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Aceleração anteroposterior')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader('Dados em X')
        st.markdown('Amplitude do modelo em X: ' + str(amplitude_modelo_x))
        st.markdown('T50 do modelo em X: ' + str(t50_x/100))
        st.markdown('T95 do modelo em X: ' + str(t90_x/100))
        st.markdown('Duração no máximo desempenho em X: ' +
                    str(15 - t90_x/100))

        st.markdown("")
        st.markdown("")
        st.markdown("")

        st.subheader('Dados em Y')
        st.markdown('Amplitude do modelo em Y: ' + str(amplitude_modelo_y))
        st.markdown('T50 do modelo em Y: ' + str(t50_y/100))
        st.markdown('T95 do modelo em Y: ' + str(t95_y/100))
        st.markdown('Duração no máximo desempenho em Y: ' +
                    str(15 - t95_y/100))

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.subheader('Dados em Z')
        st.markdown('Amplitude do modelo em Z: ' + str(amplitude_modelo_z))
        st.markdown('T50 do modelo em Z: ' + str(t50_z/100))
        st.markdown('T95 do modelo em Z: ' + str(t95_z/100))
        st.markdown('Duração no máximo desempenho em Z: ' +
                    str(15 - t95_z/100))
