import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.stats import skew
from scipy.stats import kurtosis
from matplotlib.patches import Ellipse
from scipy.optimize import leastsq
from scipy.spatial import ConvexHull
from scipy.signal import detrend
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.optimize import Bounds


def calcular_rms(serie_temporal):
    return np.sqrt(np.mean(np.square(serie_temporal)))


def calcular_entropia_aproximada(serie_temporal, m, r):
    N = len(serie_temporal)

    def _contar_similaridade(serie_temporal, m, r):
        # Cria segmentos de comprimento m
        segmentos = np.array([serie_temporal[i:i + m]
                             for i in range(N - m + 1)])

        # Calcula a contagem de pares de segmentos semelhantes
        similaridade = []
        for i in range(len(segmentos)):
            dists = np.max(np.abs(segmentos - segmentos[i]), axis=1)
            # Exclui auto-comparação
            similaridade.append(np.sum(dists <= r) - 1)

        # Normaliza pela quantidade de segmentos
        return np.sum(similaridade) / (N - m + 1)

    # Calcula C(m, r) e C(m+1, r)
    C_m = _contar_similaridade(serie_temporal, m, r) / (N - m + 1)
    C_m1 = _contar_similaridade(serie_temporal, m + 1, r) / (N - m)

    # Calcula a entropia aproximada
    ApEn = np.log(C_m) - np.log(C_m1)
    return ApEn


def plot_fft(dados, taxa_amostragem):
    # Número de pontos de dados
    N = len(dados)

    # Calculando a Transformada de Fourier
    fft_result = np.fft.fft(dados)

    # Frequências correspondentes
    freqs = np.fft.fftfreq(N, d=1/taxa_amostragem)

    # Calculando a energia (magnitude ao quadrado)
    energia = np.abs(fft_result)**2

    # Filtrando para mostrar apenas frequências positivas
    mask = freqs > 0
    freqs = freqs[mask]
    energia = energia[mask]

    # Plotando a energia em função das frequências
    fig, ax = plt.subplots()
    ax.plot(freqs, energia, 'k')
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel('Energia')
    ax.set_xlim([0, 30])

    return fig, freqs, energia


def michaelis_menten(x, DC, Vmax, n, Km):
    return DC + Vmax * (x**n) / (x**n + Km**n)


def set_ellipse(fpML, fpAP):
    points = np.column_stack((fpML, fpAP))
    hull = ConvexHull(points)

    # Get the boundary points of the convex hull
    boundary_points = points[hull.vertices]

    # Calculate the centroid of the boundary points
    centroidx = np.min(fpML) + (np.max(fpML) - np.min(fpML))/2
    centroidy = np.min(fpAP) + (np.max(fpAP) - np.min(fpAP))/2
    centroid = centroidx, centroidy

    # Calculate the covariance matrix of the boundary points
    covariance = np.cov(boundary_points, rowvar=False)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # Calculate the major and minor axis of the ellipse
    major_axis = np.sqrt(eigenvalues[0]) * np.sqrt(-2 * np.log(1 - 0.95))/2
    minor_axis = np.sqrt(eigenvalues[1]) * np.sqrt(-2 * np.log(1 - 0.95))/2

    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    area = np.pi*major_axis*minor_axis
    num_points = 101  # 360/100 + 1
    ellipse_points = np.zeros((num_points, 2))
    a = 0
    for i in np.arange(0, 361, 360 / 100):
        ellipse_points[a, 0] = centroid[0] + \
            major_axis * np.cos(np.radians(i))
        ellipse_points[a, 1] = centroid[1] + \
            minor_axis * np.sin(np.radians(i))
        a += 1
    angle_deg = -angle
    angle_rad = np.radians(angle_deg)

    # Matrix for ellipse rotation
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    ellipse_points = np.dot(ellipse_points, R)
    return ellipse_points, area, angle_deg, major_axis, minor_axis


st.set_page_config(layout='wide')
if 'contador' not in st.session_state:
    st.session_state.contador = 0

st.title("Análise de dados")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Introdução', 'Domínio do tempo', 'Domínio das frequências', 'RMS', 'Elipse'])

with tab1:
    st.markdown('Esta rotina é referente às análises de dados da dissertação de mestrado de Aline da Silva Oliveira do Programa de Pós-graduação em Ciência do Movimento Humano. Os dados do teste devem ser carrgados no botão abaixo e as análises poderão ser visualizadas nos tabs ao lado.')
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        # Ler o arquivo CSV usando pandas
        df = pd.read_csv(uploaded_file)

        # Selecionar as colunas 3 a 6 (pandas é baseado em índice 0, então colunas 2 a 5)
        time = df.iloc[:, 2]
        x_ = df.iloc[:, 3]
        y_ = df.iloc[:, 4]
        z_ = df.iloc[:, 5]

        x = detrend(x_)
        y = detrend(y_)
        z = detrend(z_)

        rms_curve_x = []
        tempo_rms_x = []
        for index, valor in enumerate(x):
            tempo_rms_x.append(index/100)
            rms_curve_x.append(np.sqrt(np.mean(np.square(x[index:index+500]))))

        rms_curve_y = []
        tempo_rms_y = []
        for index, valor in enumerate(y):
            tempo_rms_y.append(index/100)
            rms_curve_y.append(np.sqrt(np.mean(np.square(y[index:index+500]))))

        rms_curve_z = []
        tempo_rms_z = []
        for index, valor in enumerate(z):
            tempo_rms_z.append(index/100)
            rms_curve_z.append(np.sqrt(np.mean(np.square(z[index:index+500]))))

        min = np.mean(rms_curve_x[0:200])
        max = np.max(rms_curve_x[0:2000])

        contador = 0
        for index, valor in enumerate(rms_curve_x):
            if valor > min + max*0.05:
                contador = contador + 1
                if contador > 5:
                    start = index
                    break

        st.balloons()
        st.success("Arquivo carregado")
        st.session_state.contador = 1


with tab2:
    if st.session_state.contador > 0:
        st.title("Análise no domínio do tempo")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(time, x, 'k')
            ax.plot([tempo_rms_x[start-100+500],
                    tempo_rms_x[start-100+500]], [-10, 20], '--r')
            ax.plot([tempo_rms_x[start+1500+500], tempo_rms_x[start+1500+500]],
                    [-10, 20], '--r')
            ax.set_title('Eixo Vertical')
            ax.set_ylim([-10, 20])
            ax.set_xlabel('Tempo')
            ax.set_ylabel('Aceleração vertical')
            ax.legend()
            st.pyplot(fig)

            RMS_x = calcular_rms(x[start-100+500:start+1500+500])
            st.markdown(f'RMS amplitude vertical = {RMS_x}')
            skewness_x = skew(x[start-100+500:start+1500+500])
            st.text("Assimetria vertical = " + str(skewness_x))
            kurtose_x = kurtosis(x[start-100+500:start+1500+500], fisher=True)
            st.text("Curtose vertical = " + str(kurtose_x))
            m = 2
            r = 0.2 * np.std(x[start-100+500:start+1500+500])
            ApEn_x = calcular_entropia_aproximada(
                x[start-100+500:start+1500+500], m, r)
            st.text("Entropia aproximada vertical = " + str(ApEn_x))

        with col2:
            fig, ax = plt.subplots()
            ax.plot(time, y, 'k')
            ax.plot([tempo_rms_x[start-100+500],
                    tempo_rms_x[start-100+500]], [-10, 20], '--r')
            ax.plot([tempo_rms_x[start+1500+500], tempo_rms_x[start+1500+500]],
                    [-10, 20], '--r')
            ax.set_title('Eixo ML')
            ax.set_ylim([-10, 20])
            ax.set_xlabel('Tempo')
            ax.set_ylabel('Aceleração lateral')
            ax.legend()
            st.pyplot(fig)
            RMS_y = calcular_rms(y[start-100+500:start+1500+500])
            st.markdown(f'RMS amplitude ML = {RMS_y}')
            skewness_y = skew(y[start-100+500:start+1500+500])
            st.text("Assimetria ML = " + str(skewness_y))
            kurtose_y = kurtosis(y[start-100+500:start+1500+500], fisher=True)
            st.text("Curtose ML = " + str(kurtose_y))
            m = 2
            r = 0.2 * np.std(y[start-100+500:start+1500+500])
            ApEn_y = calcular_entropia_aproximada(
                y[start-100+500:start+1500+500], m, r)
            st.text("Entropia aproximada ML = " + str(ApEn_y))

        with col3:
            fig, ax = plt.subplots()
            ax.plot(time, z, 'k')
            ax.plot([tempo_rms_x[start-100+500],
                    tempo_rms_x[start-100+500]], [-10, 20], '--r')
            ax.plot([tempo_rms_x[start+1500+500], tempo_rms_x[start+1500+500]],
                    [-10, 20], '--r')
            ax.set_title('Eixo anteroposterior - AP')
            ax.set_ylim([-10, 20])
            ax.set_xlabel('Tempo')
            ax.set_ylabel('Aceleração anteroposterior')
            ax.legend()
            st.pyplot(fig)
            RMS_z = calcular_rms(z[start-100+500:start+1500+500])
            st.markdown(f'RMS amplitude AP = {RMS_z}')
            skewness_z = skew(z[start-100+500:start+1500+500])
            st.text("Assimetria AP = " + str(skewness_z))
            kurtose_z = kurtosis(z[start-100+500:start+1500+500], fisher=True)
            st.text("Curtose AP = " + str(kurtose_y))
            m = 2
            r = 0.2 * np.std(z[start-100+500:start+1500+500])
            ApEn_z = calcular_entropia_aproximada(
                z[start-100+500:start+1500+500], m, r)
            st.text("Entropia aproximada AP = " + str(ApEn_z))
    else:
        st.warning("Carregue o arquivo antes de visualizar as análises")

with tab3:
    if st.session_state.contador > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, f, en = plot_fft(x[start-100+500:start+1500+500], 100)
            st.pyplot(fig)
            peak_x = np.max(en)
            st.text("Energia de pico Vertical = " + str(peak_x))
            for index, valor in enumerate(en):
                if valor == peak_x:
                    peak_freq_x = f[index]
                    break
            st.text("Frequência de pico Vertical = " + str(peak_freq_x))
            total_en = np.sum(en)
            for index, valor in enumerate(en):
                if np.sum(en[1:index]) > total_en/2:
                    f_med_x = f[index-1]
                    break
            f_media_x = np.sum(f * en) / np.sum(en)
            variancia_espectral_x = np.sum(
                ((f - f_media_x) ** 2) * en) / np.sum(en)
            kurtose_espectral_x = np.sum(
                ((f - f_media_x) ** 4) * en) / (variancia_espectral_x ** 2 * np.sum(en))
            st.text("Frequência mediana Vertical = " + str(f_med_x))
            st.text("Centróide espectral Vertical = " + str(f_media_x))
            st.text("Variância espectral Vertical = " +
                    str(variancia_espectral_x))
            st.text("Curtose espectral Vertical = " + str(kurtose_espectral_x))
            st.text('Número de salto estimado por V =' +
                    str(round(15*peak_freq_x/2)))

        with col2:
            fig, f, en = plot_fft(y[start-100+500:start+1500+500], 100)
            st.pyplot(fig)
            peak_y = np.max(en)
            st.text("Energia de pico ML = " + str(peak_y))
            for index, valor in enumerate(en):
                if valor == peak_y:
                    peak_freq_y = f[index]
                    break
            st.text("Frequência de pico ML = " + str(peak_freq_y))
            total_en = np.sum(en)
            for index, valor in enumerate(en):
                if np.sum(en[1:index]) > total_en/2:
                    f_med_y = f[index-1]
                    break
            f_media_y = np.sum(f * en) / np.sum(en)
            variancia_espectral_y = np.sum(
                ((f - f_media_y) ** 2) * en) / np.sum(en)
            kurtose_espectral_y = np.sum(
                ((f - f_media_y) ** 4) * en) / (variancia_espectral_y ** 2 * np.sum(en))
            st.text("Frequência mediana ML = " + str(f_med_y))
            st.text("Centróide espectral ML = " + str(f_media_y))
            st.text("Variância espectral ML = " + str(variancia_espectral_y))
            st.text("Curtose espectral ML = " + str(kurtose_espectral_y))
            st.text('Número de salto estimado por ML =' +
                    str(round(15*peak_freq_y/2)))
        with col3:
            fig, f, en = plot_fft(z[start-100+500:start+1500+500], 100)
            st.pyplot(fig)
            peak_z = np.max(en)
            st.text("Energia de pico AP = " + str(peak_z))
            for index, valor in enumerate(en):
                if valor == peak_z:
                    peak_freq_z = f[index]
                    break
            st.text("Frequência de pico Vertical = " + str(peak_freq_z))
            for index, valor in enumerate(en):
                if np.sum(en[1:index]) > total_en/2:
                    f_med_z = f[index-1]
                    break
            f_media_z = np.sum(f * en) / np.sum(en)
            variancia_espectral_z = np.sum(
                ((f - f_media_z) ** 2) * en) / np.sum(en)
            kurtose_espectral_z = np.sum(
                ((f - f_media_z) ** 4) * en) / (variancia_espectral_z ** 2 * np.sum(en))
            st.text("Frequência mediana AP = " + str(f_med_z))
            st.text("Centróide espectral AP = " + str(f_media_z))
            st.text("Variância espectral AP = " +
                    str(variancia_espectral_z))
            st.text("Curtose espectral AP = " + str(kurtose_espectral_z))
            st.text('Número de salto estimado por AP =' +
                    str(round(15*peak_freq_z/2)))

    else:
        st.warning("Carregue o arquivo antes de visualizar as análises")
with tab4:
    if st.session_state.contador > 0:
        col1, col2, col3 = st.columns(3)
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
                p1 = st.number_input(
                    'Pista da baseline em X', value=1.0, step=0.1)
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
                ax.plot(t50_modelo_x, A50_x, 'yo', markersize=10)
                ax.plot(t90_modelo_x, A95_x, 'yo', markersize=10)
            ax.set_ylim([0, np.max(rms_curve_x)])
            ax.set_xlabel('Tempo')
            ax.set_ylabel('Aceleração vertical')
            ax.legend()
            st.pyplot(fig)
            if escolhaX:
                st.subheader('Dados em X')
                st.markdown('Amplitude do modelo em X: ' +
                            str(amplitude_modelo_x))
                st.markdown('T50 do modelo em X: ' + str(t50_x/100))
                st.markdown('T95 do modelo em X: ' + str(t90_x/100))
                st.markdown('Duração no máximo desempenho em X: ' +
                            str(15 - t90_x/100))
                serie1 = rms_curve_x
                interpolated_model = interp1d(
                    x_data2, y_fit2, kind='linear', fill_value="extrapolate")
                y_model_interpolated = interpolated_model(tempo_rms_x)
                serie2 = y_model_interpolated

                corr_vertical = np.corrcoef(serie1, serie2)[0, 1]
                st.markdown('Ajuste ao modelo em V: ' +
                            str(corr_vertical))
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

            if escolhaY:
                fig, ax = plt.subplots()
                ax.plot(tempo_rms_z, rms_curve_z, 'k')
                st.subheader('Dados em Y')
                st.markdown('Amplitude do modelo em Y: ' +
                            str(amplitude_modelo_y))
                st.markdown('T50 do modelo em Y: ' + str(t50_y/100))
                st.markdown('T95 do modelo em Y: ' + str(t95_y/100))
                st.markdown('Duração no máximo desempenho em Y: ' +
                            str(15 - t95_y/100))
                serie1 = rms_curve_y
                interpolated_model = interp1d(
                    x_data2, y_fit2_y, kind='linear', fill_value="extrapolate")
                y_model_interpolated = interpolated_model(tempo_rms_y)
                serie2 = y_model_interpolated

                corr_ml = np.corrcoef(serie1, serie2)[0, 1]
                st.markdown('Ajuste ao modelo em ML: ' +
                            str(corr_ml))
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
            if escolhaZ:
                st.subheader('Dados em Z')
                st.markdown('Amplitude do modelo em Z: ' +
                            str(amplitude_modelo_z))
                st.markdown('T50 do modelo em Z: ' + str(t50_z/100))
                st.markdown('T95 do modelo em Z: ' + str(t95_z/100))
                st.markdown('Duração no máximo desempenho em Z: ' +
                            str(15 - t95_z/100))
                serie1 = rms_curve_z
                interpolated_model = interp1d(
                    x_data2, y_fit2_z, kind='linear', fill_value="extrapolate")
                y_model_interpolated = interpolated_model(tempo_rms_z)
                serie2 = y_model_interpolated

                corr_ap = np.corrcoef(serie1, serie2)[0, 1]
                st.markdown('Ajuste ao modelo em AP: ' +
                            str(corr_ap))
    else:
        st.warning("Carregue o arquivo antes de visualizar as análises")
with tab5:
    if st.session_state.contador > 0:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            # Ajustar a elipse
            ellipse_fit, area_value1, angle_deg_value, a_xy, b_xy = set_ellipse(
                x[start-100:start+1500], y[start-100:start+1500])
            dc1 = (np.max(x[start-100:start+1500]) -
                   np.min(x[start-100:start+1500]))*0.5
            dc2 = (np.max(y[start-100:start+1500]) -
                   np.min(y[start-100:start+1500]))*0.5
            fig, ax = plt.subplots()
            ax.plot(x[start-100:start+1500], y[start-100:start+1500], 'k')
            ax.plot(ellipse_fit[:, 0], ellipse_fit[:, 1], 'r')
            ax.fill(ellipse_fit[:, 0], ellipse_fit[:,
                                                   1], color='tomato', alpha=0.5)
            ax.set_aspect('equal')
            ax.set_xlabel('Aceleração vertical')
            ax.set_ylabel('Aceleração mediolateral')
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.legend()
            st.pyplot(fig)
            st.text("Area 1 = " + str(area_value1))
            dado1 = y[start-100:start+1500]
            dado2 = x[start-100:start+1500]
            desvio_total_1 = np.sum(np.sqrt(dado1**2+dado2**2))

            st.text("Desvio total 1 = " + str(desvio_total_1))
        with col2:
            ellipse_fit, area_value2, angle_deg_value, a_xz, b_xz = set_ellipse(
                x[start-100:start+1500], z[start-100:start+1500])
            dc1 = (np.max(x[start-100:start+1500]) -
                   np.min(x[start-100:start+1500]))*0.5
            dc2 = (np.max(z[start-100:start+1500]) -
                   np.min(z[start-100:start+1500]))*0.5
            fig, ax = plt.subplots()
            ax.plot(x[start-100:start+1500], z[start-100:start+1500], 'k')
            ax.plot(ellipse_fit[:, 0], ellipse_fit[:, 1], 'r')
            ax.fill(ellipse_fit[:, 0], ellipse_fit[:,
                                                   1], color='tomato', alpha=0.5)
            ax.set_aspect('equal')
            ax.set_xlabel('Aceleração vertical')
            ax.set_ylabel('Aceleração anteroposterior')
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.legend()
            st.pyplot(fig)
            st.text("Area 2 = " + str(area_value2))
            dado1 = z[start-100:start+1500]
            dado2 = x[start-100:start+1500]
            desvio_total_2 = np.sum(np.sqrt(dado1**2+dado2**2))
            st.text("Desvio total 2 = " + str(desvio_total_2))
        with col3:
            ellipse_fit, area_value3, angle_deg_value, a_yz, b_yz = set_ellipse(
                y[start-100:start+1500], z[start-100:start+1500])
            fig, ax = plt.subplots()
            ax.plot(y[start-100:start+1500], z[start-100:start+1500], 'k')
            ax.plot(ellipse_fit[:, 0], ellipse_fit[:, 1], 'r')
            ax.fill(ellipse_fit[:, 0], ellipse_fit[:,
                                                   1], color='tomato', alpha=0.5)
            ax.set_aspect('equal')
            ax.set_xlabel('Aceleração mediolateral')
            ax.set_ylabel('Aceleração anteroposterior')
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.legend()
            st.pyplot(fig)
            st.text("Area 3 = " + str(area_value3))
            dado1 = y[start-100:start+1500]
            dado2 = x[start-100:start+1500]
            desvio_total_3 = np.sum(np.sqrt(dado1**2+dado2**2))
            st.text("Desvio total 3 = " + str(desvio_total_3))

            # Cálculo dos semi-eixos do elipsoide
            a = np.max([a_xy, a_xz])
            b = np.max([b_xy, a_yz])
            c = np.max([b_xz, b_yz])

            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 50)
            xe = a * np.outer(np.cos(u), np.sin(v))
            ye = b * np.outer(np.sin(u), np.sin(v))
            ze = c * np.outer(np.ones_like(u), np.cos(v))

            # Criação do gráfico 3D com matplotlib
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, color='r')
            ax.plot_surface(xe, ye, ze, color='b', alpha=0.2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title("Elipsoide Ajustado a partir das Elipses")
            limite = 20
            ax.set_xlim([-limite, limite])
            ax.set_ylim([-limite, limite])
            ax.set_zlim([-limite, limite])

            # Exibe o gráfico no Streamlit
            st.pyplot(fig)
            volume = (4/3) * np.pi * a * b * c
            st.text("Volume = " + str(volume))
