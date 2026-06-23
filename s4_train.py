import os
from modules.config_process import read_config
from modules.train_process import model_train, eval_group


def s4_dataset_train(conf):
    dsp32_path = conf['dsp32']['output']
    val_group = conf['dsp32']['groups']
    pos_group = conf['train']['pos_group']
    neg_group = conf['train']['neg_group']
    model = model_train(dsp32_path, pos_group, neg_group)
    for group in val_group:
        eval_group(model, dsp32_path, group)

if __name__ == "__main__":
    conf = read_config()
    s4_dataset_train(conf)
