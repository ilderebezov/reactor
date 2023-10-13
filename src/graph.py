import base64
import io

import matplotlib.pyplot as plt
from numpy import float64
from pandas import DataFrame


def plot(df_a: DataFrame, df_b: DataFrame, r2: float64, graph_name: str, file: bool) -> str:
    """
    plot graph based on two input Pandas DataFrame
    :param file: select mode of the output picture
                false - output picture as img_b64 object
                true - output picture as "pic.png" file
    :param graph_name: plot name
    :param r2: R2 param
    :param df_a: the first input Pandas DataFrame
    :param df_b: the second input Pandas DataFrame
    :return: string plot
    """
    # clean plot
    plt.clf()
    # Create plot
    plt.plot(df_a['t'].values, df_a['A'].values, color='r', label='A-init')
    plt.plot(df_b['t'], df_b['a'], color='g', label='A-predict')
    plt.legend()
    plt.title(f"{graph_name}")
    plt.xlabel('time, min')
    plt.ylabel('A component concentration, mol/m^3')
    plt.text(2.0, 3.0, f'R^2 score = {round(r2, 4)}', fontsize=12)
    plt.grid(True)
    if file:
        # Save plot to file
        plt.savefig('src/static/image/picture_00.png')
    else:
        # Save plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        # Convert BytesIO object to base64 string
        img_b64 = base64.b64encode(img.getvalue()).decode()
        return img_b64
