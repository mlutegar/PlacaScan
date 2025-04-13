from ultralytics import YOLO

# Carrega o modelo
model = YOLO('placa-veicular-model.pt')

# Define onde salvar:
#   project = pasta raiz (ex: "results")
#   name    = subpasta específica (ex: "plates")
# Com save=True e save_crop=True, o YOLO vai:
#  - salvar as imagens completas com bounding‑boxes em results/plates
#  - salvar cada crop de placa em results/plates/crops
model(source=0,
      conf=0.4,
      show=True,
      save=True,
      save_crop=True,
      project='results',
      name='plates')

# Se quiser também no stream IP:
windows_ip = "172.20.10.5"
stream_url = f"http://{windows_ip}:8080/video"

model(source=stream_url,
      conf=0.4,
      show=True,
      save=True,
      save_crop=True,
      project='results',
      name='plates')
