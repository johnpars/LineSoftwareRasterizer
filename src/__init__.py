import os
import sys
import coalpy.gpu as gpu

print ("Devices:")
[print("{}: {}".format(idx, nm)) for (idx, nm) in gpu.get_adapters()]

settings_obj = gpu.get_settings()
settings_obj.adapter_index = 0
settings_obj.dump_shader_pdbs = True

# try:
root = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     root = "{}/../src/".format(os.path.dirname(os.path.abspath(sys.argv[0])))

gpu.add_data_path("{}/shaders/".format(root))
gpu.add_data_path("{}/data/".format(root))
