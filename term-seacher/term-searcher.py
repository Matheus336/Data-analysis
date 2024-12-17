from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt

file_path = '/content/drive/My Drive/clientes/gateways.csv'
data = pd.read_csv(file_path, on_bad_lines='skip', sep=";")

text = ' '.join(data['Mensageria'].astype(str))

cleaned_text = re.sub(r'[^\w\s]', '', text)

terms = ["Email SMTP", "App Provedor", "Telegram", "ZapMe", "SMSFire", "Zenvia V1", "Zenvia V2", "iAgente", "KingSMS", "SMSNet", "EAI", "Comtele", "HopChat", "SZ.CHAT", "OwenBrasil", "Digisac", "SMSNET", "HiperSend",
         "Botfy", "ZAPISP", "Athostec", "ConectaZap", "Virtushost", "Meu James", "ZapShow", "Evotrix", "Chatmix", ]

count_terms = {term: len(re.findall(r'\b' + re.escape(term) + r'\b', cleaned_text)) for term in terms}

sorted_count_terms = dict(sorted(count_terms.items(), key=lambda item: item[1], reverse=True))

for term, count in sorted_count_terms.items():
    print(f"{term}: {count}")

print("\n")
plt.figure(figsize=(10, 6))
plt.bar(count_terms.keys(), count_terms.values(), color='skyblue')
plt.title('Gráfico de clientes que utilizam mensageiros')
plt.xlabel('Gatewayss')
plt.ylabel('Quantidade de clientes')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
