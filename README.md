# Cortex-M-KWS

## 简介
**Cortex-M-KWS** 由 **OPEN AI LAB** 主导开发，该项目主要用于基于 **Cortex-M4** 核心的微控制通过 **Tengine Lite** 推理框架简单部署关键词识别（**KWS**, Keyword Spotting）的功能。方便开发者进行技术评估。

## 开发环境
硬件需求

- STM32F469I-DISCO：STM32F469I开发评估套件（基于 Cortex-M4 核心）

软件需求

- Keil IDE
- CMSIS
- FreeRTOS
- Tengine Lite

## 运行说明 

- Keil IDE 安装

  - 独立完成 Keil-MDK 5.3 版本。

- 下载本项目

```bash
git clone  https://github.com/OAID/cortex-m-kws.git
```

- 编译及烧写
  
  - 使用 Keil-MDK 打开工程文件
  
    `cortex-m-kws/Projects/STM32469I-Discovery/Applications/AID_newmodel/aid_speech/MDK-ARM/Project.uvprojx`
  
  - 根据提示完成相关依赖库安装
  
  - 使用 USB 连接电脑和开发板
  
  - 编译工程，生成 hex 文件
  
  - 下载 hex 文件至开发板

## 关键词 
本项目包含11个关键词，分别是：小智小智、打开窗帘、关闭窗帘、打开空调、关闭空调、加热模式、制冷模式、降低温度、调高温度、开启扫风、启动空调。



