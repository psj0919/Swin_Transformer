from Config.config_test import get_test_config_dict
from Core.engine_Swin_transformer_test import Trainer





if __name__=='__main__':
    cfg = get_test_config_dict()
    trainer = Trainer(cfg)
    trainer.test()