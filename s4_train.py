import os
from modules.config_process import read_config
from modules.train_process import model_train, eval_groups

def export_py_model(model, output_file):
    coef = model.coef_[0]
    bias = model.intercept_[0]
    with open(output_file, "w") as f:
        f.write("# Auto Generated\n\n")
        f.write(f"BIAS = {bias:.9f}\n\n")
        f.write("WEIGHT = [\n")
        for w in coef:
            f.write(f"    {w:.9f},\n")
        f.write("]\n")

def export_c_model(model, output_file):
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

def s4_dataset_train(conf):
    dsp32_path = conf['dsp32']['output']
    val_group = conf['dsp32']['groups']
    pos_group = conf['train']['pos_group']
    neg_group = conf['train']['neg_group']
    model_file_py = conf['train']['model_py']
    model_file_h = conf['train']['model_h']
    print(model_file_py, model_file_h)
    model = model_train(dsp32_path, pos_group, neg_group)
    export_py_model(model, model_file_py)
    export_c_model(model, model_file_h)
    eval_groups(model, dsp32_path, val_group)


if __name__ == "__main__":
    conf = read_config()
    s4_dataset_train(conf)
