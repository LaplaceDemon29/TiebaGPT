# 贴吧智能回复助手 (Tieba GPT Assistant)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flet](https://img.shields.io/badge/UI-Flet-green.svg)](https://flet.dev/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-purple.svg)](https://ai.google.dev/)
[![aiotieba](https://img.shields.io/badge/API-aiotieba-orange.svg)](https://github.com/Starry-OvO/aiotieba)

**贴吧智能回复助手**是一款基于 Google Gemini AI 模型的桌面应用程序，旨在帮助用户智能分析百度贴吧的帖子，并根据不同的策略生成高质量的回复。无论你是想加入讨论、平息争论还是想“抬杠”，它都能成为你的得力助手。


## ✨ 功能亮点

-   **帖子浏览**: 输入贴吧名称，即可分页浏览该吧的热门帖子。
-   **智能战况分析**: 采用 Gemini 的 JSON 模式，一键分析帖子中各方的观点、主要论点以及讨论是否具有争议性。
-   **多模式回复生成**:
    -   **支持模型**: 站在主流观点一边，生成表示支持和赞同的回复。
    -   **抬杠模型**: 针对主流或特定观点，生成有理有据的反驳或“抬杠”回复。
    -   **自定义模型**: 输入你自己的观点，让 AI 为你组织语言，生成符合你想法的回复。

## 🚀 安装与启动

### 1. 环境准备

-   确保你已经安装了 Python 3.9 或更高版本。
-   建议使用虚拟环境以隔离项目依赖。

### 2. 克隆与安装依赖

首先，克隆本仓库到你的本地：
```bash
git clone https://github.com/LaplaceDemon29/TiebaGPT.git
cd TiebaGPT
```

然后，创建并激活虚拟环境（可选）：
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

随后，安装所有必需的库：
```bash
pip install -r requirements.txt
```

最后，运行应用：
```bash
python gui.py
```

### 3. 配置

在首次使用前，你需要配置你的 Google Gemini API Key。

1.  运行应用后，点击主界面右上角的 **设置** 图标。
2.  在 **Gemini API Key** 输入框中，填入你从 [Google AI Studio](https://makersuite.google.com/app/apikey) 获取的 API Key。
3.  
    -   **分析模型**: 推荐使用 `gemini-1.5-flash-latest`，速度快且便宜。
    -   **生成模型**: 可以使用 `gemini-1.5-flash-latest` 或效果更好的 `gemini-1.5-pro-latest`。
4.  点击 **保存设置**。应用会自动使用新的配置初始化。

配置文件 `settings.json` 会在首次保存后自动生成在项目根目录。


## 🌟 未来计划

-   [ ] 支持帖子内图片内容的识别与分析（多模态）。
-   [ ] 增加更多预设的回复模型（如“中立和事佬”、“吃瓜群众”等）。
-   [ ] 历史记录功能，保存分析过的帖子和生成的回复。
-   [ ] 将应用打包成可执行文件，方便分发。

## 🤝 贡献

欢迎提交 Pull Requests 或开启 Issues 来为这个项目做出贡献！

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。