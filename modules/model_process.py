def export_py_model(model, output_file):
    coef = model.coef_[0]
    bias = model.intercept_[0]
    with open(output_file, "w") as f:
        f.write("# Auto Generated\n\n")
        f.write("import numpy as np\n\n")
        f.write(f"BIAS = {bias:.9f}\n\n")
        f.write("WEIGHT = [\n")
        for row in range(5):
            start = row * 8
            end = start + 8
            values = [f"{coef[i]: .9f}" for i in range(start, end)]
            f.write("    ")
            f.write(", ".join(values))
            f.write(",\n")
        f.write("]\n\n")
        f.write("""
def predict_score(feature):
    feature = np.asarray(feature, dtype=np.float32)
    return np.dot(feature, WEIGHT) + BIAS

def predict_prob(feature):
    score = predict_score(feature)
    return 1.0 / (1.0 + np.exp(-score))

def predict_class(feature, threshold=0.5):
    prob = predict_prob(feature)
    return 1 if prob >= threshold else 0

def predict_batch(features):
    features = np.asarray(features, dtype=np.float32)
    scores = np.dot(features, WEIGHT) + BIAS
    probs = 1.0 / (1.0 + np.exp(-scores))
    return probs
""")

def export_h_model(model, output_file):
    coef = model.coef_[0]
    bias = model.intercept_[0]
    with open(output_file, "w") as f:
        f.write("#ifndef FAN_MODEL_H\n")
        f.write("#define FAN_MODEL_H\n\n")
        f.write(
             f"static const float g_bias = {bias:.9f}f;\n\n"
        )
        f.write(
            "static const float g_weight[40] = {\n"
        )
        for i,w in enumerate(coef):
            if i != len(coef)-1:
                f.write(f"    {w:.9f}f,\n")
            else:
                f.write(f"    {w:.9f}f\n")
        f.write("};\n\n")
        f.write("#endif\n")

