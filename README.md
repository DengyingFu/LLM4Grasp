# LLM4Grasp

LLM4Grasp 是一个基于Multi-Agents和视觉感知的抓取规划框架，结合了多模态感知与机械臂控制

![image](https://github.com/DengyingFu/LLM4Grasp/blob/main/sim.png)

## 目录结构

- [安装](#安装)
- [运行](#运行)

## 安装

### 安装 GraspNet 1B 所需依赖（KNN、Pointnet2）

```bash
cd Sim-Grasp
cd libs/knn
python setup.py install

cd ../pointnet2
python setup.py install
```

### 安装 Genesis 仿真所需依赖

```bash
cd ../..
pip install -r requirements.txt
```

> 注意：请确保你位于 `Sim-Grasp` 目录下再执行以下命令。

### 安装 Grounded-Segment-Anything 模块

```bash
cd Grounded-Segment-Anything
pip install -r requirements.txt
```

## 运行

### 启动 Grounding-DINO 服务
需要自行下载权重 https://github.com/IDEA-Research/Grounded-Segment-Anything

```bash
cd Grounded-Segment-Anything
python Flask_DINO.py
```

> **注意**：需要修改 `Flask_DINO.py` 最后一行以适配本地或远程访问：
>
> ```python
> app.run(host='127.0.0.1')  # 本地使用
> ```
>
> 或者
>
> ```python
> # app.run(host='0.0.0.0', port=5000)  # 作为服务器供外部连接
> ```

#### 测试服务是否正常运行

```bash
python Flask_Demo.py
```

### 使用 Multi-Agent 控制机械臂

在两个不同的终端中分别运行以下命令：

```bash
# 终端1
cd Sim-Grasp
python DINO.py
```

```bash
# 终端2
python Agents-VLM.py
```

该部分实现了基于视觉语言模型（VLM）的多智能体协同控制，用于机械臂的抓取任务。

---

如需进一步了解各模块功能及原理，请查看项目文档或相关论文。欢迎贡献代码或提出 Issues！

