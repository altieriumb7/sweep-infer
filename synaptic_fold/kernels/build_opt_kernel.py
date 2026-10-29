import subprocess
import os

nasm = r"C:\Users\Umberto\AppData\Local\bin\NASM\nasm.exe"
clang = r"C:\Users\Umberto\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20261020-ucrt-x86_64\bin\clang.exe"
dir_path = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\kernels"

asm_file = os.path.join(dir_path, "synaptic_column_avx2_opt.asm")
obj_file = os.path.join(dir_path, "synaptic_column_avx2_opt.obj")
dll_file = os.path.join(dir_path, "synaptic_column_avx2_opt.dll")

print("1. Assembling with NASM...")
cmd_nasm = [nasm, "-f", "win64", asm_file, "-o", obj_file]
res = subprocess.run(cmd_nasm, capture_output=True, text=True)
if res.returncode != 0:
    print("NASM ERROR:", res.stderr)
    exit(1)
print("NASM success! Object size:", os.path.getsize(obj_file))

print("2. Linking with Clang...")
cmd_clang = [clang, "-shared", "-O3", "-mavx2", "-mfma", obj_file, "-o", dll_file]
res2 = subprocess.run(cmd_clang, capture_output=True, text=True)
if res2.returncode != 0:
    print("Clang ERROR:", res2.stderr)
    exit(1)

print("Clang success! DLL size:", os.path.getsize(dll_file))
