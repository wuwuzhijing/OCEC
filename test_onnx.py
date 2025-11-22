import onnx

model = onnx.load("runs/ocec/ocec_epoch_1000_nodynamic.onnx")

print("=== Operators in Model ===")
ops = [n.op_type for n in model.graph.node]
print(set(ops))

print("\n=== GlobalAveragePool nodes ===")
for n in model.graph.node:
    if n.op_type == "GlobalAveragePool":
        print(n.name, n.input, n.output)
