import os
#GPU_TO_USE = int(open("gpu_to_use.txt", encoding="utf").read().splitlines()[0]) # type: ignore
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" if GPU_TO_USE == -1 else str(GPU_TO_USE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
