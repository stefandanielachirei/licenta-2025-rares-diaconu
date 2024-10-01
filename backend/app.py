from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)

# Configurarea conexiunii la PostgreSQL prin variabile de mediu
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@postgres:5432/{os.getenv("POSTGRES_DB")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inițializarea SQLAlchemy cu aplicația Flask
db = SQLAlchemy(app)

# Definirea unui model pentru baza de date
class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(80), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Ruta principală
@app.route('/')
def index():
    # Preluăm primul rând din tabelă (sau toate rândurile, după nevoie)
    first_entry = Test.query.first()  # Sau `.all()` pentru toate rândurile
    if first_entry:
        return f"Text from DB: {first_entry.text}"
    else:
        return "No entries found in the database!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)
