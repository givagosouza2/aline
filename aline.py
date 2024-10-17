import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skew
from matplotlib.patches import Ellipse
from scipy.optimize import leastsq
from scipy.spatial import ConvexHull


def set_ellipse(fpML, fpAP):
    points = np.column_stack((fpML, fpAP))
    hull = ConvexHull(points)

    # Get the boundary points of the convex hull
    boundary_points = points[hull.vertices]

    # Calculate the centroid of the boundary points
    centroidx = np.mean(fpML)
    centroidy = np.mean(fpAP)
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
        if escolhaX:
            st.subheader('Dados em X')
            st.markdown('Amplitude do modelo em X: ' + str(amplitude_modelo_x))
            st.markdown('T50 do modelo em X: ' + str(t50_x/100))
            st.markdown('T95 do modelo em X: ' + str(t90_x/100))
            st.markdown('Duração no máximo desempenho em X: ' +
                        str(15 - t90_x/100))

    with col2:
        if escolhaY:
            st.subheader('Dados em Y')
            st.markdown('Amplitude do modelo em Y: ' + str(amplitude_modelo_y))
            st.markdown('T50 do modelo em Y: ' + str(t50_y/100))
            st.markdown('T95 do modelo em Y: ' + str(t95_y/100))
            st.markdown('Duração no máximo desempenho em Y: ' +
                        str(15 - t95_y/100))

    with col3:
        if escolhaZ:
            st.subheader('Dados em Z')
            st.markdown('Amplitude do modelo em Z: ' + str(amplitude_modelo_z))
            st.markdown('T50 do modelo em Z: ' + str(t50_z/100))
            st.markdown('T95 do modelo em Z: ' + str(t95_z/100))
            st.markdown('Duração no máximo desempenho em Z: ' +
                        str(15 - t95_z/100))
    with col1:
        # Ajustar a elipse
        ellipse_fit, area_value1, angle_deg_value, major_axis_value, minor_axis_value = set_ellipse(
            y[start-100:start+1500], x[start-100:start+1500])
        dc1 = (np.max(x[start-100:start+1500]) -
               np.min(x[start-100:start+1500]))*0.5
        dc2 = (np.max(y[start-100:start+1500]) -
               np.min(y[start-100:start+1500]))*0.5
        fig, ax = plt.subplots()
        ax.plot(y[start-100:start+1500], x[start-100:start+1500], 'k')
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
    with col2:
        ellipse_fit, area_value2, angle_deg_value, major_axis_value, minor_axis_value = set_ellipse(
            z[start-100:start+1500], x[start-100:start+1500])
        dc1 = (np.max(x[start-100:start+1500]) -
               np.min(x[start-100:start+1500]))*0.5
        dc2 = (np.max(z[start-100:start+1500]) -
               np.min(z[start-100:start+1500]))*0.5
        fig, ax = plt.subplots()
        ax.plot(z[start-100:start+1500], x[start-100:start+1500], 'k')
        ax.plot(ellipse_fit[:, 0], ellipse_fit[:, 1], 'r')
        ax.fill(ellipse_fit[:, 0], ellipse_fit[:,
                                               1], color='tomato', alpha=0.5)
        ax.set_xlabel('Aceleração vertical')
        ax.set_ylabel('Aceleração anteroposterior')
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.legend()
        st.pyplot(fig)
    with col3:
        ellipse_fit, area_value3, angle_deg_value, major_axis_value, minor_axis_value = set_ellipse(
            y[start-100:start+1500], z[start-100:start+1500])
        fig, ax = plt.subplots()
        ax.plot(y[start-100:start+1500], z[start-100:start+1500], 'k')
        ax.plot(ellipse_fit[:, 0], ellipse_fit[:, 1], 'r')
        ax.fill(ellipse_fit[:, 0], ellipse_fit[:,
                                               1], color='tomato', alpha=0.5)
        ax.set_xlabel('Aceleração mediolateral')
        ax.set_ylabel('Aceleração anteroposterior')
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.legend()
        st.pyplot(fig)
    with col1:
        skewness_vertical = skew(x[start-100:start+1500])
        skewness_ml = skew(y[start-100:start+1500])
        skewness_ap = skew(z[start-100:start+1500])
        st.text("Assimetria vertical = " + str(skewness_vertical))
        st.text("Assimetria médio-lateral = " + str(skewness_ml))
        st.text("Assimetria antero-posterior = " + str(skewness_ap))
        st.text("Area 1 = " + str(area_value1))
        st.text("Area 2 = " + str(area_value2))
        st.text("Area 3 = " + str(area_value3))
