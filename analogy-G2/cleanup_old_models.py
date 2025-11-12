#!/usr/bin/env python3
"""
Clean up old Qwen2 models that are no longer used
"""

import os
import shutil
from pathlib import Path

def cleanup_old_qwen_models():
    """Remove old Qwen2-0.5B model cache"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    old_model = cache_dir / "models--Qwen--Qwen2-0.5B"
    old_lock = cache_dir / ".locks" / "models--Qwen--Qwen2-0.5B"
    
    current_model = cache_dir / "models--Qwen--Qwen3-0.6B"
    
    print("=" * 70)
    print("🧹 Cleaning up old Qwen models")
    print("=" * 70)
    print()
    
    cleaned = False
    
    # Check current model
    if current_model.exists():
        size = sum(f.stat().st_size for f in current_model.rglob('*') if f.is_file()) / (1024**3)
        print(f"✅ Current model: Qwen3-0.6B")
        print(f"   Location: {current_model}")
        print(f"   Size: {size:.2f} GB")
        print()
    else:
        print("⚠️  Warning: Qwen3-0.6B not found in cache")
        print("   This is OK if you haven't used QWEN model yet")
        print()
    
    # Check and remove old model
    if old_model.exists():
        size = sum(f.stat().st_size for f in old_model.rglob('*') if f.is_file()) / (1024**3)
        print(f"🗑️  Found old model: Qwen2-0.5B")
        print(f"   Location: {old_model}")
        print(f"   Size: {size:.2f} GB")
        print()
        
        try:
            shutil.rmtree(old_model)
            print(f"✅ Deleted: {old_model}")
            print(f"   Freed up: {size:.2f} GB")
            cleaned = True
        except Exception as e:
            print(f"❌ Error deleting {old_model}: {e}")
            print(f"   You may need to delete it manually")
    else:
        print("✅ No old Qwen2-0.5B model found")
    
    # Check and remove old lock directory (may contain multiple lock files)
    if old_lock.exists():
        print()
        try:
            if old_lock.is_dir():
                # Remove all lock files in the directory
                lock_files = list(old_lock.glob("*.lock"))
                if lock_files:
                    for lock_file in lock_files:
                        try:
                            lock_file.unlink()
                            print(f"✅ Deleted lock file: {lock_file.name}")
                        except Exception as e:
                            print(f"⚠️  Could not delete {lock_file.name}: {e}")
                # Try to remove the directory
                try:
                    old_lock.rmdir()
                    print(f"✅ Deleted lock directory: {old_lock}")
                    cleaned = True
                except Exception as e:
                    print(f"⚠️  Could not remove lock directory: {e}")
            else:
                old_lock.unlink()
                print(f"✅ Deleted lock file: {old_lock}")
                cleaned = True
        except Exception as e:
            print(f"⚠️  Could not delete lock: {e}")
            print(f"   Note: Lock files are usually safe to ignore")
            print(f"   They will be automatically cleaned up by HuggingFace")
    
    print()
    print("=" * 70)
    if cleaned:
        print("✅ Cleanup complete! Old Qwen2-0.5B model removed.")
    else:
        print("✅ No cleanup needed. All models are up to date.")
    print("=" * 70)
    
    return cleaned

if __name__ == "__main__":
    cleanup_old_qwen_models()
