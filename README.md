# 贴吧智能回复助手 (Tieba GPT Assistant)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flet](https://img.shields.io/badge/UI-Flet-green.svg)](https://flet.dev/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-purple.svg)](https://ai.google.dev/)
[![aiotieba](https://img.shields.io/badge/API-aiotieba-orange.svg)](https://github.com/Starry-OvO/aiotieba)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**贴吧智能回复助手**是一款桌面应用，它利用 Google Gemini AI 模型，帮助用户智能地分析百度贴吧的帖子，并根据用户的需求生成各种风格的回复。无论您是想理性讨论、激情对线，还是想成为故事接龙大师，本工具都能为您提供强大的支持。

## ✨ 功能特性

*   **帖子浏览与搜索**: 按回复时间、发布时间、热门度浏览指定贴吧的帖子，或通过关键词精确搜索。
*   **智能讨论分析**:
    *   **分块处理**: 自动将长篇帖子（多页回复）分块，逐一发送给AI进行分析，突破上下文长度限制。
    *   **摘要整合**: 将各分块的分析摘要再次整合，生成一份全面、连贯、高质量的全局讨论摘要。
*   **多模式AI回复生成**:
    *   内置多种预设回复模式，如“支持模型”、“抬杠模型”、“学术考据党”、“故事接龙大师”等。
    *   支持需要用户提供自定义观点的“自定义模型”。
*   **高度可定制的Prompt系统**:
    *   **回复模式编辑器**: 用户可以在图形化界面中**添加、删除、编辑**任何回复模式，包括其名称、描述、AI扮演的`角色(Role)`和要执行的`任务(Task)`。
    *   **AI辅助创建模式**: 在创建新模式时，只需输入名称和描述，即可一键调用AI**自动生成**对应的`Role`和`Task`。
    *   **元编辑功能**: 对于高级用户，可以编辑用于“生成新模式”的AI Prompt本身，实现对工具行为的完全控制。
*   **现代化图形界面**: 基于 Flet 框架构建，提供清爽、直观、跨平台的用户体验。

## 🛠️ 安装与运行

### 1. 先决条件

*   Python 3.9 或更高版本。
*   Git (用于获取开发版本号)。

### 2. 获取 Gemini API Key

您需要一个 Google Gemini API 密钥才能使用本工具的AI功能。
1.  访问 [Google AI Studio](https://aistudio.google.com/)。
2.  使用您的 Google 账户登录。
3.  点击 "**Get API key**" -> "**Create API key in new project**"。
4.  复制生成的 API 密钥。

### 3. 安装依赖

克隆本仓库到本地：
```bash
git clone https://github.com/LaplaceDemon29/TiebaGPT.git
cd TiebaGPT
```

安装所需的 Python 包：
```bash
pip install flet google-genai aiotieba
```

### 4. 运行应用

执行主程序文件：
```bash
python gui.py
```

首次运行后，请：
1.  点击主界面右上角的**设置图标**。
2.  在“API 设置”中，粘贴您获取的 Gemini API 密钥。
3.  点击“**测试Key并获取模型**”按钮，如果成功，下方的模型下拉框将被填充。
4.  选择您想用于“分析”和“生成”的模型。
5.  点击“**保存设置**”。

现在，您可以返回主界面，开始使用了！

## 📖 使用指南

1.  **获取帖子**: 在主界面输入“贴吧名称”和可选的“关键词”，选择排序方式，点击“获取帖子”。
2.  **选择帖子**: 在帖子列表中，点击您感兴趣的帖子进入分析页面。
3.  **分析帖子**:
    *   在分析页面，点击“**分析整个帖子**”按钮。
    *   程序将开始分批次获取和分析所有回复，状态日志和进度条会显示当前进度。
    *   分析完成后，左侧会显示帖子预览，中间会显示AI生成的讨论状况摘要。
4.  **生成回复**:
    *   在右侧的“生成回复”卡片中，从下拉框选择一个**回复模式**。
    *   如果选择了需要自定义观点的模式，下方的输入框将变为可见，请输入您的观点。
    *   点击“**生成回复**”按钮，AI将根据讨论摘要和您选择的模式生成回复内容。
    *   使用“复制”按钮将内容复制到剪贴板。

## 🔧 高级自定义

所有自定义操作都在**设置 -> 高级：自定义 Prompt**折叠面板中进行。

*   **编辑基础Prompt**: 您可以直接修改分析器、摘要器和通用回复规则的系统级Prompt。
*   **管理回复模式**:
    *   **添加**: 点击“添加新模式”。
    *   **AI辅助**: 在弹出的对话框中，填写“模式名称”和“描述”，然后点击“**AI生成Role和Task**”按钮，AI会自动填充下方的输入框。
    *   **编辑/删除**: 在模式列表中，点击对应条目右侧的编辑或删除图标。
    *   **保存**: 在对话框中点击“保存”后，您的更改会**立即写入**配置文件，无需再点击全局的“保存 Prompts”按钮。

## 🤝 贡献

欢迎任何形式的贡献！如果您有好的想法、功能建议或发现了Bug，请随时提交 Pull Request 或创建 Issue。

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。