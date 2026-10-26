import tensorflow as tf
import os

print("=" * 60)
print("TensorFlow GPU Diagnostic")
print("=" * 60)

print("\n1. TensorFlow Version:")
print(f"   {tf.__version__}")

print("\n2. GPU Devices:")
gpus = tf.config.list_physical_devices('GPU')
print(f"   Found {len(gpus)} GPU(s)")
for gpu in gpus:
    print(f"   - {gpu}")

print("\n3. CUDA Details:")
print(f"   CUDA available: {tf.test.is_built_with_cuda()}")

print("\n4. cuDNN Details:")
print(f"   cuDNN available: {tf.sysconfig.get_build_info()['cuda_version']}")

print("\n5. Environment Variables:")
print(f"   PATH includes CUDA: {'CUDA' in os.environ.get('PATH', '')}")

print("\n6. Trying GPU Operation:")
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        c = tf.matmul(a, b)
        print(f"   ✓ GPU operation successful")
except Exception as e:
    print(f"   ✗ GPU operation failed: {e}")

print("\n" + "=" * 60)
