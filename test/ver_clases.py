# ver_clases.py
import pickle

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

clases = list(encoder.classes_)

with open("test/palabras.txt", "w") as f:
    f.write("Clases de palabras reconocidas:\n")
    f.write(clases[0] + "\n")  # Escribir la clase "silence" al principio
    for palabra in sorted(clases[1:]):  # Escribir el resto de clases ordenadas alfabéticamente
        f.write(palabra + "\n")

print(f"Total de palabras: {len(clases)}\n")
for i, palabra in enumerate(sorted(clases)):
    print(f"  {i+1:>3}. {palabra}")