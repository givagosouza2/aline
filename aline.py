import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def michaelis_menten(x, DC, Vmax, n, Km):
    return DC + Vmax * (x**n) / (x**n + Km**n)


st.set_page_config(layout="wide")
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
    col1, col2, col3 = st.columns(3)
    with col1:
        tempo_pico = st.number_input(
            'Indique o tempo estimado do pico em X', value=2000)
    rms_curve_x = []
    tempo_rms_x = []
    for index, valor in enumerate(x):
        tempo_rms_x.append(index)
        rms_curve_x.append(np.sqrt(np.mean(np.square(x[index:index+500]))))

    rms_curve_y = []
    tempo_rms_y = []
    for index, valor in enumerate(y):
        tempo_rms_y.append(index)
        rms_curve_y.append(np.sqrt(np.mean(np.square(y[index:index+500]))))

    rms_curve_z = []
    tempo_rms_z = []
    for index, valor in enumerate(z):
        tempo_rms_z.append(index)
        rms_curve_z.append(np.sqrt(np.mean(np.square(z[index:index+500]))))

    min = np.mean(rms_curve_x[0:200])
    max = np.max(rms_curve_x[0:tempo_pico])

    contador = 0
    for index, valor in enumerate(rms_curve_x):
        if valor > min + max*0.01:
            contador = contador + 1
            if contador > 2:
                start = index
                break

    with col1:
        escolhaX = st.checkbox('Quer modelar os dados de X?')
    with col2:
        escolhaY = st.checkbox('Quer modelar os dados de Y?')
    with col3:
        escolhaZ = st.checkbox('Quer modelar os dados de Z?')

    with col1:
        if escolhaX:
            # modelo do começo
            x_data2 = tempo_rms_x[start-100:start+1500]
            y_data2 = rms_curve_x[start-100:start+1500]
            st.text("Pistas iniciais do modelo em X")
            p1 = st.number_input('Pista da baseline em X', value=1.0, step=0.1)
            p2 = st.number_input(
                'Pista do valor máximo em X', value=np.max(y_data2))
            p3 = st.number_input(
                'Pista do índice de cooperatividade em X', value=1.0)
            p4 = st.number_input(
                'Pista do tempo de semi-saturação em X', value=3.0)

            # Ajustando a função de Michaelis-Menten aos dados
            # Suposições iniciais para os parâmetros Vmax e Km
            initial_guess = [p1, p2, p3, p4]
            try:
                params, covariance = curve_fit(
                    michaelis_menten, x_data2, y_data2, p0=initial_guess, maxfev=4000)
            except RuntimeError:
                params, covariance = curve_fit(
                    michaelis_menten, x_data2, y_data2, p0=initial_guess, maxfev=5000, full_output=True)[0]

            # Obtendo os parâmetros ajustados
            DC_fit2, Vmax_fit2, n_fit2, Km_fit2 = params

            # Gerando curva ajustada
            y_fit2 = michaelis_menten(
                x_data2, DC_fit2, Vmax_fit2, n_fit2, Km_fit2)

            amplitude_modelo_x = np.max(y_fit2) - np.min(y_fit2)
            A50_x = amplitude_modelo_x/2 + np.min(y_fit2)
            A95_x = (amplitude_modelo_x*0.95) + np.min(y_fit2)

            for index, valor in enumerate(y_fit2):
                if valor > A50_x:
                    t50_modelo_x = tempo_rms_x[start-100+index]
                    break

            for index, valor in enumerate(y_fit2):
                if valor > A95_x:
                    t90_modelo_x = tempo_rms_x[start-100+index]
                    break
            t50_x = t50_modelo_x - tempo_rms_x[start-100]
            t90_x = t90_modelo_x - tempo_rms_x[start-100]

    with col2:
        if escolhaY:
            # modelo do começo
            x_data2_y = tempo_rms_y[start-100:start+1500]
            y_data2_y = rms_curve_y[start-100:start+1500]
            # Ajustando a função de Michaelis-Menten aos dados
            # Suposições iniciais para os parâmetros Vmax e Km
            st.text("Pistas iniciais do modelo em Y")
            p1 = st.number_input('Pista da baseline em Y', value=0.1)
            p2 = st.number_input(
                'Pista do valor máximo em Y', value=np.max(y_data2_y))
            p3 = st.number_input(
                'Pista do índice de cooperatividade em Y', value=1.0)
            p4 = st.number_input(
                'Pista do tempo de semi-saturação em Y', value=3.0)
            initial_guess = [p1, p2, p3, p4]
            try:
                params, covariance = curve_fit(
                    michaelis_menten, x_data2_y, y_data2_y, p0=initial_guess, maxfev=4000)
            except RuntimeError:
                params, covariance = curve_fit(
                    michaelis_menten, x_data2_y, y_data2_y, p0=initial_guess, maxfev=5000, full_output=True)[0]

            # Obtendo os parâmetros ajustados
            DC_fit2, Vmax_fit2, n_fit2, Km_fit2 = params

            # Gerando curva ajustada
            y_fit2_y = michaelis_menten(
                x_data2_y, DC_fit2, Vmax_fit2, n_fit2, Km_fit2)

            amplitude_modelo_y = np.max(y_fit2_y) - np.min(y_fit2_y)
            A50_y = amplitude_modelo_y/2 + np.min(y_fit2_y)
            A95_y = (amplitude_modelo_y*0.95) + np.min(y_fit2_y)

            for index, valor in enumerate(y_fit2_y):
                if valor > A50_y:
                    t50_modelo_y = tempo_rms_y[start-100+index]
                    break

            for index, valor in enumerate(y_fit2_y):
                if valor > A95_y:
                    t95_modelo_y = tempo_rms_y[start-100+index]
                    break
            t50_y = t50_modelo_y - tempo_rms_y[start-100]
            t95_y = t95_modelo_y - tempo_rms_y[start-100]
    with col3:
        if escolhaZ:
            # modelo do começo
            x_data2_z = tempo_rms_z[start-100:start+1500]
            y_data2_z = rms_curve_z[start-100:start+1500]
            # Ajustando a função de Michaelis-Menten aos dados
            # Suposições iniciais para os parâmetros Vmax e Km
            st.text("Pistas iniciais do modelo")
            p1 = st.number_input('Pista da baseline em Z', value=0.1)
            p2 = st.number_input(
                'Pista do valor máximo em Z', value=np.max(y_data2_z))
            p3 = st.number_input(
                'Pista do índice de cooperatividade em Z', value=1)
            p4 = st.number_input(
                'Pista do tempo de semi-saturação em Z', value=3)
            initial_guess = [p1, p2, p3, p4]

            try:
                params, covariance = curve_fit(
                    michaelis_menten, x_data2_z, y_data2_z, p0=initial_guess, maxfev=4000)
            except RuntimeError:
                params, covariance = curve_fit(
                    michaelis_menten, x_data2_z, y_data2_z, p0=initial_guess, maxfev=5000, full_output=True)[0]
            if n_fit2 <= 0:
                raise ValueError("O valor de n deve ser positivo.")
            # Obtendo os parâmetros ajustados
            DC_fit2, Vmax_fit2, n_fit2, Km_fit2 = params

            # Gerando curva ajustada
            y_fit2_z = michaelis_menten(
                x_data2_z, DC_fit2, Vmax_fit2, n_fit2, Km_fit2)

            amplitude_modelo_z = np.max(y_fit2_z) - np.min(y_fit2_z)
            A50_z = amplitude_modelo_z/2 + np.min(y_fit2_z)
            A95_z = (amplitude_modelo_z*0.95) + np.min(y_fit2_z)

            for index, valor in enumerate(y_fit2_z):
                if valor > A50_z:
                    t50_modelo_z = tempo_rms_z[start-100+index]
                    break

            for index, valor in enumerate(y_fit2_z):
                if valor > A95_z:
                    t95_modelo_z = tempo_rms_z[start-100+index]
                    break
            t50_z = t50_modelo_z - tempo_rms_z[start-100]
            t95_z = t95_modelo_z - tempo_rms_z[start-100]

    with col1:
        fig, ax = plt.subplots()
        ax.plot(tempo_rms_x, rms_curve_x, 'k')

        if escolhaX:
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
    with col2:
        fig, ax = plt.subplots()
        ax.plot(tempo_rms_y, rms_curve_y, 'k')

        if escolhaY:
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
    with col3:
        if escolhaZ:
            ax.plot(x_data2_z, y_fit2_z, 'r')
            ax.plot(t50_modelo_z, A50_z, 'yo', markersize=10)
            ax.plot(t95_modelo_z, A95_z, 'yo', markersize=10)
        # ax.plot([tempo_rms_z[start], tempo_rms_z[start]], [0, 5], 'r--')
        # ax.plot([tempo_rms_z[start+1500], tempo_rms_z[start+1500]], [0, 5], 'r--')
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Aceleração anteroposterior')
        ax.legend()
        st.pyplot(fig)

    with col1:
        st.subheader('Dados em X')
        st.markdown('Amplitude do modelo em X: ' + str(amplitude_modelo_x))
        st.markdown('T50 do modelo em X: ' + str(t50_x/100))
        st.markdown('T95 do modelo em X: ' + str(t90_x/100))
        st.markdown('Duração no máximo desempenho em X: ' +
                    str(15 - t90_x/100))

    with col2:
        st.subheader('Dados em Y')
        st.markdown('Amplitude do modelo em Y: ' + str(amplitude_modelo_y))
        st.markdown('T50 do modelo em Y: ' + str(t50_y/100))
        st.markdown('T95 do modelo em Y: ' + str(t95_y/100))
        st.markdown('Duração no máximo desempenho em Y: ' +
                    str(15 - t95_y/100))

    with col3:
        st.subheader('Dados em Z')
        st.markdown('Amplitude do modelo em Z: ' + str(amplitude_modelo_z))
        st.markdown('T50 do modelo em Z: ' + str(t50_z/100))
        st.markdown('T95 do modelo em Z: ' + str(t95_z/100))
        st.markdown('Duração no máximo desempenho em Z: ' +
                    str(15 - t95_z/100))
