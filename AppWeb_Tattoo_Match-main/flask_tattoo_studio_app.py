import os
import io
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from dotenv import load_dotenv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tattoo_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class ClientRequest(db.Model):
    __tablename__ = 'client_requests'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(50), nullable=True)
    tattoo = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=func.now())


with app.app_context():
    db.create_all()

#Análise de Dados

# --- DADOS FICTÍCIOS ---

data = {
    'genero': ['M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'F'],
    'idade': [25, 30, 22, 40, 35, 28, 19, 45, 27, 32],
    'profissao': ['TI', 'Advogada', 'Artes', 'Executivo', 'Mecanico', 'Saude', 'Estudante', 'Professora', 'Design', 'Vendas'],
    'estilo_musical': ['Rock', 'Pop', 'Indie', 'Jazz', 'Metal', 'MPB', 'Trap', 'Classica', 'Eletronica', 'Sertanejo'],
    'hobby': ['Games', 'Leitura', 'Pintura', 'Viagem', 'Moto', 'Yoga', 'Skate', 'Jardinagem', 'Fotografia', 'Academia'],
    'categoria_sugerida': ['Caveiras', 'Flores', 'Mandalas', 'Tribais', 'Tribais', 'Flores', 'Animais', 'Mandalas', 'Animais', 'Palavras']
}
df_treino = pd.DataFrame(data)
# Encoders para transformar texto em números para o algoritmo
encoders = {}
for col in ['genero', 'profissao', 'estilo_musical', 'hobby']:
    le = LabelEncoder()
    df_treino[col] = le.fit_transform(df_treino[col])
    encoders[col] = le

X = df_treino.drop('categoria_sugerida', axis=1)
y = df_treino['categoria_sugerida']
model = DecisionTreeClassifier()
model.fit(X, y)


TATTOOS_CATEGORIZADAS = {
    'Flores': [
        {'id': 'f1', 'label': 'Rosa Tradicional', 'file': 't1.jpg'},
        {'id': 'f2', 'label': 'Lótus Geométrica', 'file': 'Lotus geometrica.jpg'},
        {'id': 'f3', 'label': 'Girassol Blackwork', 'file': 'Girassol Blackwork.jpg'},
        {'id': 'f4', 'label': 'Ramo de Lavanda', 'file': 'Ramo de lavanda.jpg'},
    ],
    'Caveiras': [
        {'id': 'c1', 'label': 'Caveira Realista', 'file': 'Caveira realista.jpg'},
        {'id': 'c2', 'label': 'Caveira Mexicana', 'file': 'Caveira mexicana.jpg'},
        {'id': 'c3', 'label': 'Pirata Old School', 'file': 'Pirata old School.jpg'},
        {'id': 'c4', 'label': 'Caveira com Flores', 'file': 'Caveira com flores.jpg'},
    ],
    'Animais': [
        {'id': 'a1', 'label': 'Leão Majestoso', 'file': 'Leao majestoso.jpg'},
        {'id': 'a2', 'label': 'Lobo em Pontilhismo', 'file': 'Lobo em pontilhismo.jpg'},
        {'id': 'a3', 'label': 'Águia de Ataque', 'file': 'Aguia de ataque.jpg'},
        {'id': 'a4', 'label': 'Tigre de Bengala', 'file': 'Tigre de bengala.jpg'},
    ],
    'Mandalas': [
        {'id': 'm1', 'label': 'Mandala de Lótus', 'file': 'Mandala de lotus.jpg'},
        {'id': 'm2', 'label': 'Mandala Ornamental', 'file': 'Mandala Ornamental.jpg'},
        {'id': 'm3', 'label': 'Mandala de Pulso', 'file': 'Mandala de pulso.jpg'},
        {'id': 'm4', 'label': 'Mandala Sagrada', 'file': 'Mandala sagrada.jpg'},
    ],
    'Tribais': [
        {'id': 't1', 'label': 'Maori de Braço', 'file': 'Maiori de braco.jpg'},
        {'id': 't2', 'label': 'Polinésia Peitoral', 'file': 'Polinesia peitoral.jpg'},
        {'id': 't3', 'label': 'Nórdica Viking', 'file': 'Nordica viking.jpg'},
        {'id': 't4', 'label': 'Celta Entrelaçado', 'file': 'Celta entrelacado.jpg'},
    ],
    'Palavras': [
        {'id': 'p1', 'label': 'Caligrafia "Resiliência"', 'file': 'Resiliencia.jpg'},
        {'id': 'p2', 'label': 'Frase "Carpe Diem"', 'file': 'Carpe diem.jpg'},
        {'id': 'p3', 'label': 'Nome Personalizado', 'file': 'Nome personalizado.jpg'},
        {'id': 'p4', 'label': 'Palavra "Gratidão"', 'file': 'Gratidao.jpg'},
    ]
}


@app.route('/')
def index():
    return render_template('pagina.html', categorias=TATTOOS_CATEGORIZADAS)


@app.route('/dashboard')
def dashboard():
    pedidos = ClientRequest.query.order_by(ClientRequest.created_at.desc()).all()
    return render_template('dashboard.html', pedidos=pedidos)


@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json() or {}
    req = ClientRequest(
        name=data.get('name'),
        email=data.get('email'),
        phone=data.get('phone'),
        tattoo=data.get('tattoo')
    )
    db.session.add(req)
    db.session.commit()
    return jsonify({'ok': True})



@app.route('/delete/<int:id>', methods=['POST'])
def delete_request(id):
    pedido = ClientRequest.query.get_or_404(id)
    db.session.delete(pedido)
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/export')
def export_excel():
    pedidos = ClientRequest.query.all()
    data = [{
        'Data': p.created_at.strftime('%d/%m/%Y %H:%M'),
        'Nome': p.name,
        'E-mail': p.email,
        'Telefone': p.phone,
        'Tatuagem': p.tattoo
    } for p in pedidos]

    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Pedidos')

    output.seek(0)
    return send_file(output, download_name="pedidos_tattoo.xlsx", as_attachment=True)


@app.route('/sugestao')
def pagina_sugestao():
    return render_template('sugestao.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.form
    try:
        # Preparar entrada do usuário (com tratamento para categorias novas)
        input_row = []
        for col in ['genero', 'profissao', 'estilo_musical', 'hobby']:
            val = user_data.get(col)
            # Se o valor não existe no treino, usamos o primeiro disponível para não quebrar
            if val not in encoders[col].classes_:
                input_row.append(0)
            else:
                input_row.append(encoders[col].transform([val])[0])

        input_row.insert(1, int(user_data.get('idade')))  # idade é na posição 1

        prediction = model.predict([input_row])[0]

        # Escolhemos a primeira tattoo da categoria sugerida
        sugestao_tattoo = TATTOOS_CATEGORIZADAS[prediction][0]

        return render_template('resultado.html', categoria=prediction, tattoo=sugestao_tattoo)
    except Exception as e:
        return f"Erro no processamento: {e}", 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)