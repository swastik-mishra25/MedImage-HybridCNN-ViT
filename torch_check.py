import torch

print("====================================")
print("🔥 PyTorch Installation Test")
print("====================================")

try:
    print("✅ Torch version:", torch.__version__)
    print("✅ CUDA available:", torch.cuda.is_available())
    print("✅ Running on CPU (safe mode)...")

    # Create a simple tensor
    x = torch.rand(2, 3)
    print("\n✅ Test tensor created successfully:")
    print(x)

    print("\n🎉 PyTorch is working correctly! No DLL issues detected.")
except Exception as e:
    print("\n❌ Error:", e)
    print("Something’s wrong with your PyTorch setup.")
