import webbrowser
from json import loads
from os import path

from flask import Flask, request, render_template_string, render_template

from src.predict import create_update_model
from src.predict import predict_data

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')


@app.route('/graph', methods=['GET'])
def plot_graph():
    """
    endpoint to show graph
    :return:
    """
    return render_template('img_render.html')


@app.route('/create_update_model', methods=['GET'])
def create_upgrade_model():
    """
    endpoint to create or update predict model
    :return:
    """
    filename_model = "rnd_reg_model.sav"
    check_file = path.isfile(filename_model)
    cru_model = create_update_model()
    if check_file:
        if cru_model[0]:
            html = f'<img src="data:image/png;base64,{cru_model[1]}" class="blog-image">'
            return render_template_string(html)
        else:
            return "The data file is absent"
    else:
        if cru_model[0]:
            html = f'<img src="data:image/png;base64,{cru_model[1]}" class="blog-image">'
            return render_template_string(html)
        else:
            return "The data file is absent"


@app.route('/model', methods=['POST'])
def predict():
    """
    endpoint to predict based on user data
    :return:
    """
    data_in_json = loads(request.data)
    pred_data = predict_data(data_in=data_in_json)
    webbrowser.open_new_tab('http://127.1.1.1:5000/graph')
    return pred_data
