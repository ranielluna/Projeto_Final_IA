from flask import Flask, redirect, render_template, request, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Função para converter os dados do formulário para um array NumPy
def convert_to_array(features):
    return np.array([float(feature) for feature in features])

# Definir a rota para a página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Definir a rota para processar os dados do formulário
@app.route('/predict', methods=['POST'])
def predict():
    # Receber os dados do formulário
    ph = float(request.form['ph'])
    hardness = float(request.form['hardness'])
    solids = float(request.form['solids'])
    chloramines = float(request.form['chloramines'])
    sulfate = float(request.form['sulfate'])
    conductivity = float(request.form['conductivity'])
    organic_carbon = float(request.form['organic_carbon'])
    turbidity = float(request.form['turbidity'])
    trihalomethanes = float(request.form['trihalomethanes'])
    
    # Converter os dados do formulário para um array NumPy
    features = convert_to_array([ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, turbidity, trihalomethanes])
    
    # Fazer a previsão com o modelo
    prediction = model.predict([features])  # Passar os recursos como uma lista
    
    # Determinar a mensagem de retorno com base na previsão
    if prediction[0] == 1:
        result = 'Água potável'
    else:
        result = 'Água não potável'
    # Redirecionar o usuário para a página de resultado com a mensagem
    return redirect(url_for('result', result=result))

# Definir a rota para a página de resultado
@app.route('/result')
def result():
    result = request.args.get('result')  # Obter o resultado da previsão da URL
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
