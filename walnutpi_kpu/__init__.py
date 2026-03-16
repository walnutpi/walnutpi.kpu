from typing import Literal, Dict, Any
import importlib

# 支持多个版本
NNCASEVersionType = Literal["2.10","2.11"]


def get_nncase(nncase_version: NNCASEVersionType) -> Any:
    """
    获取指定版本的nncase库
    """
    module_name_map: Dict[NNCASEVersionType, str] = {
        "2.10": "nncase_2_10",
        "2.11": "nncase_2_11",
    }

    module_name = module_name_map.get(nncase_version)
    if not module_name:
        available_versions = list(module_name_map.keys())
        raise ValueError(
            f"Unsupported nncase version: {nncase_version}. "
            f"Available versions: {available_versions}"
        )

    try:
        nncase_lib = importlib.import_module(
            f".{module_name}", package=__name__.rsplit(".", 1)[0]
        )
        return nncase_lib
    except ImportError as e:
        raise ImportError(f"Failed to import .{module_name} from {__name__}: {e}")
