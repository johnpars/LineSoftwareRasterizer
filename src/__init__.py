import os
import sys
import coalpy.gpu as gpu

print ("Devices:")
[print("{}: {}".format(idx, nm)) for (idx, nm) in gpu.get_adapters()]

gpu.set_current_adapter(
    index=1,
    dump_shader_pdbs=True
)

# try:
root = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     root = "{}/../src/".format(os.path.dirname(os.path.abspath(sys.argv[0])))

gpu.add_data_path("{}/shaders/".format(root))
gpu.add_data_path("{}/data/".format(root))