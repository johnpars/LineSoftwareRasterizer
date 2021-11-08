import os
import sys
import pathlib
import coalpy.gpu as gpu

gModulePath = os.path.dirname(pathlib.Path(sys.modules[__name__].__file__)) + "\\"

print ("Devices:")
[print("{}: {}".format(idx, nm)) for (idx, nm) in gpu.get_adapters()]

gpu.set_current_adapter(0)
gpu.add_data_path(gModulePath)

def GetModulePath():
    return gModulePath