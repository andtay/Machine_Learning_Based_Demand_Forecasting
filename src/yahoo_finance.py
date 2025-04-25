import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd

# Current dir
current_dir = os.path.dirname(os.path.abspath(__file__))

# Opciones de driver
options = webdriver.ChromeOptions()
options.page_load_strategy = "normal"
# Hacerse pasar por un navegador real para que no detecte que es un bot
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
driver = webdriver.Chrome(
    service = Service(ChromeDriverManager().install()),
    options = options
)
# Acceso a Yahoo Finance
url="https://finance.yahoo.com/quote/BZ%3DF/history/?period1=1514764800&period2=1525824000"
driver.get(url=url)
try:
    # Rechazar el Botón de Consentimiento (Cookies)
    print("Buscando botón de consentimento...")
    consent_button = WebDriverWait(driver, 300).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.secondary.reject-all"))
        )
    consent_button.click()
    print("Botón 'Rechazar Todo' clickeado")
    time.sleep(5)

    # Esperar a que el elemento esté presente antes de seleccionarlo
    # Cabecera de datos
    data_head = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-container table.table thead"))
    )
    # Datos de la tabla
    data_body = driver.find_element(By.CSS_SELECTOR, "div.table-container table.table tbody")

    print(data_head.text)

except Exception as e:
    print("No ha aparecido el popup")
# Obtener cabecera de datos
headers = []
for row in data_head.find_elements(By.CSS_SELECTOR, "tr"):
    print(row.text)
    header =  row.find_elements(By.CSS_SELECTOR, "th")
    headers= [h.text for h in header]
#print(headers)

def get_dataframe(data_body):
    list_stock = []
    for row in data_body.find_elements(By.CSS_SELECTOR, "tr"):
        cells = row.find_elements(By.CSS_SELECTOR, "td") 
        list_stock.append([c.text for c in cells])
        print("Procesando fila...")
    return list_stock

stocks = get_dataframe(data_body=data_body)

# Dataframe de datos
df = pd.DataFrame(stocks, columns=headers)
print(df)

## Guardar en CSV
# Path
brent_oil_path = os.path.join(current_dir, '../data/raw/macrodata')
# Crear un directorio si no existe
os.makedirs(brent_oil_path, exist_ok=True)
# Archivo CSV
brent_oil_data = os.path.join(brent_oil_path, "brent_oil.csv")
# Guardar dataframe en CSV
df.to_csv(brent_oil_data, index=False)



