from memory_profiler import profile
import Core as md
import numpy as np

@profile
def test_memory():
    for i in range(10):
        x = md.Variable(np.random.randn(10000))
        y = md.square(md.square(md.square(x)))

if __name__ == '__main__':
    test_memory

# 使用mprof命令运行脚本
# mprof run .\MuxiDeep\test2.py

# 使用 mprof plot 命令绘制内存使用情况图表