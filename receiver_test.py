import numpy as np
import cv2
from multiprocessing import shared_memory
import time

def main():
    try:
        # 首先获取帧的形状
        shape_shm = shared_memory.SharedMemory(name='frame_shape')
        shape_str = bytes(shape_shm.buf).decode().strip('\x00')
        height, width, channels = map(int, shape_str.split(','))
        frame_shape = (height, width, channels)
        print(f"接收端 - Frame shape: {frame_shape}")

        # 连接到帧数据的共享内存
        shm = shared_memory.SharedMemory(name='frame_share')
        shared_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
        
        while True:
            frame = shared_frame.copy()
            cv2.imshow('Received Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except FileNotFoundError:
        print("共享内存未找到，请确保发送方程序正在运行")
    except Exception as e:
        print(f"接收错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()
        if 'shm' in locals():
            shm.close()
        if 'shape_shm' in locals():
            shape_shm.close()

if __name__ == "__main__":
    main()
