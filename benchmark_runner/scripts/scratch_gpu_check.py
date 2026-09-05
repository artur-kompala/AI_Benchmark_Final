import time
import numpy as np
import torch
import openvino as ov

# Wczytaj checkpoint referencyjny bezposrednio (ten sam plik co uzywa benchmark_runner)
checkpoint_path = "data/checkpoints/resnet50_cifar10_seed42.pt"

# Podmien na realna definicje modelu z Twojego repo - sprawdz dokladna nazwe funkcji/klasy
# w tasks/image_classification/models.py (np. build_resnet50())
from benchmark_runner.tasks.image_classification.models import build_resnet50

model = build_resnet50()
state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)
onnx_path = "/tmp/scratch_resnet50_test.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=13)
print(f"Wyeksportowano do {onnx_path}")

core = ov.Core()
print("Dostepne urzadzenia:", core.available_devices)

ov_model = core.read_model(onnx_path)
compiled = core.compile_model(ov_model, device_name="GPU")
print("EXECUTION_DEVICES:", compiled.get_property("EXECUTION_DEVICES"))

input_shape = compiled.inputs[0].shape
dummy = np.random.rand(*input_shape).astype(np.float32)
infer_req = compiled.create_infer_request()

print("Startuje petle inferencji na 15 sekund - patrz na intel_gpu_top w drugim terminalu...")
start = time.time()
count = 0
while time.time() - start < 15:
    infer_req.infer([dummy])
    count += 1
print(f"Wykonano {count} inferencji w 15s")