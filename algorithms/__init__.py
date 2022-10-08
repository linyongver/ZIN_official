from .infer_irmv1 import InferIrmV1
from .erm import ERM 
from .infer_irmv1_multi_class import Infer_Irmv1_Multi_Class
from .eiil import EIIL 
from .lff import LfF 

def algorithm_builder(flags, dp):
    class_name = flags.irm_type
    return {
        'infer_irmv1': InferIrmV1,
        'erm': ERM,
        'infer_irmv1_multi_class': Infer_Irmv1_Multi_Class,
        'eiil': EIIL, 
        'lff': LfF, 
    }[class_name](flags, dp)