import streamlit as st
import paho.mqtt.client as mqtt
import json

# Variables para guardar el último mensaje recibido
st.set_page_config(page_title="Selector de Animal", page_icon=":paw_prints:")
if "last_animal" not in st.session_state:
    st.session_state.last_animal = None
    st.session_state.last_valor = None

# Función de callback al recibir mensaje
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        st.session_state.last_animal = data.get("animal")
        st.session_state.last_valor = data.get("valor")
    except Exception as e:
        st.session_state.last_animal = "Error"
        st.session_state.last_valor = str(e)

# Configuración del cliente MQTT
client = mqtt.Client()
client.on_message = on_message

client.connect("broker.mqttdashboard.com", 1883, 60)
client.subscribe("selector/animal")
client.loop_start()

# Interfaz Streamlit
st.title("Visualizador de Animal por Potenciómetro")
st.write("Pulsa el botón para mostrar el animal seleccionado actualmente.")

if st.button("Ver Animal Actual"):
    if st.session_state.last_animal:
        st.success(f"Animal seleccionado: **{st.session_state.last_animal}**")
        st.write(f"Valor del potenciómetro: `{st.session_state.last_valor}`")
    else:
        st.warning("Aún no se ha recibido ningún dato. Espera unos segundos...")
