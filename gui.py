import flet as ft
import asyncio
import json
from google import genai
import core_logic as core
import aiotieba as tb
from aiotieba import ThreadSortType
from aiotieba import typing as tb_typing
# from aiotieba.api.search_exact import ExactSearches

class TiebaGPTApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "贴吧智能回复助手"
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.window_width = 1200
        self.page.window_height = 800
        self.app_version = core.get_app_version()

        # --- 状态变量 ---
        self.settings = {}; self.gemini_client = None; self.threads = []; self.selected_thread = None
        self.discussion_text = ""; self.analysis_result = None; self.current_mode = None
        self.custom_viewpoint = None; self.current_page_num = 1; self.previous_view_name = "initial"; self.thread_list_scroll_offset = 0.0
        self.analysis_cache = {}
        self.current_analysis_tid = None
        self.current_search_query = None

        # --- UI 控件 ---
        # -- 通用 --
        self.status_log = ft.TextField(
            label="状态日志",
            multiline=True,
            read_only=True,
            expand=True,
            border=ft.InputBorder.NONE, # Cleaner look inside a container
            min_lines=5, # Ensure it has some initial height
            text_size=10
        )
        self.progress_ring = ft.ProgressRing(visible=False)
        self.settings_button = ft.IconButton(icon=ft.Icons.SETTINGS, on_click=self.open_settings_view, tooltip="设置")
        # -- 初始页/列表页 --
        self.tieba_name_input = ft.TextField(label="贴吧名称", width=250)
        self.search_query_input = ft.TextField(label="帖子关键词 (可选)", hint_text="留空则按排序浏览", width=250, expand=False)
        self.sort_type_dropdown = ft.Dropdown(
            label="排序方式",
            width=150,
            expand=False,
            options=[
                ft.dropdown.Option(key=ThreadSortType.REPLY, text="按回复时间"),
                ft.dropdown.Option(key=ThreadSortType.HOT, text="热门排序"),
                ft.dropdown.Option(key=ThreadSortType.CREATE, text="按发布时间"),
            ],
            value=ThreadSortType.REPLY, # Default value
        )
        self.search_button = ft.ElevatedButton("获取帖子", on_click=self.search_tieba, icon=ft.Icons.FIND_IN_PAGE)
        self.thread_list_view = ft.ListView(expand=1, spacing=10, auto_scroll=False)
        self.prev_page_button = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_LEFT, on_click=self.load_prev_page, tooltip="上一页", disabled=True)
        self.next_page_button = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_RIGHT, on_click=self.load_next_page, tooltip="下一页", disabled=True)
        self.page_num_display = ft.Text(f"第 {self.current_page_num} 页", weight=ft.FontWeight.BOLD)
        
        # -- 分析页 --
        self.preview_display = ft.ListView(expand=True, spacing=10, auto_scroll=False)
        self.analysis_display = ft.Markdown(selectable=True, code_theme="atom-one-dark")
        self.reply_display = ft.Markdown(selectable=True, code_theme="atom-one-light")
        self.analyze_button = ft.ElevatedButton("分析讨论状况", icon=ft.Icons.INSIGHTS_ROUNDED, on_click=self.analyze_thread_click, tooltip="点击开始使用AI分析帖子内容", disabled=True)
        self.mode_selector = ft.RadioGroup(content=ft.Row([ft.Radio(value="支持模型", label="支持模型"), ft.Radio(value="抬杠模型", label="抬杠模型"), ft.Radio(value="自定义模型", label="自定义模型")]), on_change=self.on_mode_change, disabled=True)
        self.mode_selector.value = "支持模型"
        self.custom_view_input = ft.TextField(label="请输入您的自定义观点或要抬杠的主题", multiline=True, max_lines=3, visible=False)
        self.generate_button = ft.ElevatedButton("生成回复", on_click=self.generate_reply_click, icon=ft.Icons.AUTO_AWESOME, disabled=True)
        self.copy_button = ft.IconButton(icon=ft.Icons.CONTENT_COPY_ROUNDED, tooltip="复制回复内容", on_click=self.copy_reply_click, disabled=True)
        
        # -- 设置页控件 ---
        self.api_key_input = ft.TextField(label="Gemini API Key", password=True, can_reveal_password=True, on_change=self.validate_settings)
        self.analyzer_model_dd = ft.Dropdown(label="分析模型", hint_text="选择一个分析模型", on_change=self.validate_settings, expand=True)
        self.generator_model_dd = ft.Dropdown(label="生成模型", hint_text="选择一个生成模型", on_change=self.validate_settings, expand=True)
        self.fetch_models_button = ft.ElevatedButton("测试Key并获取模型", on_click=self.fetch_models_click, icon=ft.Icons.CLOUD_DOWNLOAD)
        self.save_settings_button = ft.ElevatedButton("保存设置", on_click=self.save_settings_click, icon=ft.Icons.SAVE, disabled=True)

        self.prompt_text_fields = {} # 存储 prompt 文本框的字典
        self.save_prompts_button = ft.ElevatedButton(
            "保存 Prompts",
            on_click=self.save_prompts_click,
            icon=ft.Icons.SAVE_ALT,
            disabled=True
        )
        self.restore_prompts_button = ft.ElevatedButton(
            "恢复默认 Prompts",
            on_click=self.restore_prompts_click,
            icon=ft.Icons.RESTORE
        )
        
        self.view_container = ft.Column([self.build_initial_view()], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER)


    # --- 视图构建方法 ---
    def build_initial_view(self):
        # Group inputs together
        input_row = ft.Row(
            [
                self.tieba_name_input,
                self.search_query_input,
                self.sort_type_dropdown,
                self.search_button,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10
        )
        app_info_row = ft.Row(
            [
                ft.Text(f"v{self.app_version}", color=ft.Colors.GREY_600), 
                ft.Icon(ft.Icons.CIRCLE, size=8, color=ft.Colors.GREY_400), # 分隔点
                ft.Text("作者: LaplaceDemon", color=ft.Colors.GREY_600), # 你的名字/ID
                ft.Icon(ft.Icons.CIRCLE, size=8, color=ft.Colors.GREY_400), # 分隔点
                ft.Text("Made with Gemini", color=ft.Colors.GREY_600), # 协助制作信息
                ft.Icon(ft.Icons.AUTO_AWESOME, size=14, color=ft.Colors.AMBER_500), # Gemini 的小图标
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8
        )
        return ft.Column([
                ft.Row([self.settings_button], alignment=ft.MainAxisAlignment.END),
                ft.Text("贴吧智能回复助手", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                input_row, # Use the new row
                self.progress_ring,
                ft.Divider(),
                # ft.Text("状态日志:", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Container(self.status_log, border=ft.border.all(1, ft.Colors.OUTLINE), expand=True, border_radius=5, padding=10),
                ft.Container(content=app_info_row, padding=ft.padding.only(top=10, bottom=5))
            ],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10
        )

    def build_thread_list_view(self):
        title_text = ""
        if self.current_search_query:
            title_text = f"在“{self.tieba_name_input.value}”吧中搜索“{self.current_search_query}”的结果"
        else:
            title_text = f"“{self.tieba_name_input.value}”吧的帖子"

        return ft.Column([
                ft.Row([ft.ElevatedButton("返回", on_click=self.back_to_initial, icon=ft.Icons.ARROW_BACK), ft.Container(expand=True), self.settings_button]),
                ft.Text(title_text, style=ft.TextThemeStyle.HEADLINE_SMALL),
                self.progress_ring,
                ft.Divider(),
                ft.Container(self.thread_list_view, border=ft.border.all(1, ft.Colors.OUTLINE), expand=True, border_radius=5, padding=5),
                ft.Row([self.prev_page_button,self.page_num_display,self.next_page_button], alignment=ft.MainAxisAlignment.CENTER)
            ],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10
        )

    def build_analysis_view(self):
        preview_card = ft.Column(
            controls=[
                ft.Text("帖子预览", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Container(
                    content=self.preview_display, # 直接使用 ListView
                    border=ft.border.all(1, ft.Colors.OUTLINE), border_radius=5, padding=10, expand=True
                )
            ],
            expand=True, spacing=10
        )
        analysis_card = ft.Column(controls=[ft.Text("讨论状况分析", style=ft.TextThemeStyle.TITLE_MEDIUM),ft.Container(content=ft.Column([self.analyze_button, self.analysis_display],scroll=ft.ScrollMode.ADAPTIVE, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),border=ft.border.all(1, ft.Colors.OUTLINE), border_radius=5, padding=10, expand=True)],expand=True, spacing=10)
        reply_card = ft.Column(controls=[ft.Text("生成回复", style=ft.TextThemeStyle.TITLE_MEDIUM),self.mode_selector,self.custom_view_input,ft.Row([self.generate_button, self.copy_button], alignment=ft.MainAxisAlignment.CENTER),ft.Divider(),ft.Container(content=ft.Column([self.reply_display], scroll=ft.ScrollMode.ADAPTIVE, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),border=ft.border.all(1, ft.Colors.OUTLINE),border_radius=5,padding=10,expand=True,bgcolor=ft.Colors.LIGHT_BLUE_50)],expand=True, spacing=10)
        return ft.Column([ft.Row([ft.ElevatedButton("返回帖子列表", on_click=self.back_to_thread_list, icon=ft.Icons.ARROW_BACK), ft.Container(expand=True), self.settings_button, self.progress_ring]),ft.Text(self.selected_thread.title if self.selected_thread else "帖子", style=ft.TextThemeStyle.HEADLINE_SMALL, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),ft.Divider(),ft.Row(controls=[preview_card, analysis_card, reply_card], spacing=10, expand=True),ft.Divider(),ft.Text("状态日志:", style=ft.TextThemeStyle.TITLE_MEDIUM),ft.Container(self.status_log, border=ft.border.all(1, ft.Colors.OUTLINE), height=100, border_radius=5, padding=10)], expand=True, spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    
    def build_settings_view(self):
        api_settings_content = ft.Column(
            [
                ft.Text("API 设置", style=ft.TextThemeStyle.TITLE_LARGE),
                ft.Text("请在这里配置您的Gemini API。"),
                self.api_key_input,
                self.fetch_models_button,
                self.progress_ring,
                ft.Divider(),
                ft.Row(controls=[self.analyzer_model_dd, self.generator_model_dd], spacing=20),
                ft.Divider(),
                self.save_settings_button
            ],
            spacing=15,
        )

        # --- Prompt 设置部分 ---
        self.prompt_text_fields.clear() # 清空旧的控件
        prompt_panel_content = self._build_prompt_editors()
        
        prompt_settings_content = ft.ExpansionPanelList(
            expand_icon_color=ft.Colors.BLUE_GREY,
            elevation=2,
            controls=[
                ft.ExpansionPanel(
                    header=ft.ListTile(title=ft.Text("高级：自定义 Prompt", style=ft.TextThemeStyle.TITLE_LARGE)),
                    content=ft.Container(
                        ft.Column([
                            ft.Text("警告：不正确的修改可能导致程序功能异常。请仅修改文本内容。", color=ft.Colors.ORANGE_700),
                            ft.Row([self.save_prompts_button, self.restore_prompts_button], spacing=20),
                            ft.Divider(height=20),
                            prompt_panel_content,
                        ]),
                        padding=ft.padding.all(15),
                    )
                )
            ]
        )

        # --- 整体布局 ---
        settings_main_column = ft.Column(
            [
                api_settings_content,
                ft.Divider(height=30),
                prompt_settings_content,
            ],
            spacing=15,
            width=800,
        )

        return ft.Column(
            [
                ft.Row([
                    ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=self.close_settings_view, tooltip="返回")
                ]),
                settings_main_column
            ],
            expand=True, 
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            scroll=ft.ScrollMode.ADAPTIVE
        )

    def _build_prompt_editors(self) -> ft.Column:
        """动态构建Prompt编辑器的UI"""
        controls_list = []
        prompts = core.PROMPTS
        
        def create_editor(key_path, value):
            label = " -> ".join(key_path)
            tf = ft.TextField(
                label=label,
                value=value,
                multiline=True,
                min_lines=2,
                max_lines=5,
                text_size=12,
                on_change=lambda e: self.on_prompt_change(e),
            )
            self.prompt_text_fields[tuple(key_path)] = tf
            return tf

        # 1. 分析器系统提示
        analyzer_sys_prompt_path = ('stance_analyzer', 'system_prompt')
        controls_list.append(create_editor(analyzer_sys_prompt_path, prompts['stance_analyzer']['system_prompt']))
        
        controls_list.append(ft.Divider(height=10))
        controls_list.append(ft.Text("回复生成器 Prompts", style=ft.TextThemeStyle.TITLE_MEDIUM))
        
        # 2. 通用规则
        common_rules_path = ('reply_generator', 'common_rules', 'title')
        controls_list.append(create_editor(common_rules_path, prompts['reply_generator']['common_rules']['title']))
        
        # 3. 各个模式的 Role 和 Scenarios
        for mode, mode_config in prompts['reply_generator']['modes'].items():
            controls_list.append(ft.Divider(height=5))
            controls_list.append(ft.Text(f"模式: {mode}", weight=ft.FontWeight.BOLD))
            
            # Role
            role_path = ('reply_generator', 'modes', mode, 'role')
            controls_list.append(create_editor(role_path, mode_config['role']))
            
            # Scenarios
            for scenario, text in mode_config['scenarios'].items():
                scenario_path = ('reply_generator', 'modes', mode, 'scenarios', scenario)
                controls_list.append(create_editor(scenario_path, text))

        return ft.Column(controls_list, spacing=15)

    def _create_post_widget(self, user_name: str, content_str: str, floor_text: str, is_lz: bool = False, is_comment: bool = False) -> ft.Control:
        """创建一个帖子或楼中楼的UI组件"""
        user_info_row = ft.Row(
            controls=[
                ft.Icon(ft.Icons.ACCOUNT_CIRCLE, color=ft.Colors.BLUE_GREY_400, size=20),
                ft.Text(user_name, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_700),
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=5
        )
        if is_lz:
            user_info_row.controls.append(
                ft.Chip(
                    label=ft.Text("楼主", size=10, weight=ft.FontWeight.BOLD),
                    bgcolor=ft.Colors.BLUE_100,
                    padding=ft.padding.all(2),
                    height=20
                )
            )
        
        header_row = ft.Row(
            controls=[
                user_info_row,
                ft.Text(floor_text, color=ft.Colors.GREY_600, size=12)
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )

        content_display = ft.Text(content_str, selectable=True)
        
        post_column = ft.Column(
            controls=[header_row, content_display],
            spacing=5
        )

        if is_comment:
            return ft.Container(
                content=post_column,
                padding=ft.padding.only(left=30, top=8, bottom=8, right=5),
                border=ft.border.only(left=ft.border.BorderSide(2, ft.Colors.GREY_300)),
                bgcolor=ft.Colors.GREY_100,
                border_radius=ft.border_radius.all(4)
            )
        else: # 主楼或回复
            return ft.Column([post_column, ft.Divider(height=1, thickness=1)])

    # --- 核心逻辑与事件处理 ---
    async def initialize_app(self):
        self.settings = core.load_settings()
        await self.log_message("设置已加载。")
        success, msg = core.load_prompts()
        await self.log_message(msg)
        if not success: self.search_button.disabled = True
        if self.settings.get("api_key"):
            try:
                self.gemini_client = genai.Client(api_key=self.settings["api_key"])
                await self.log_message("Gemini Client 初始化成功。")
            except Exception as e: await self.log_message(f"使用已保存的Key初始化失败: {e}，请前往设置更新。")
        else:
            await self.log_message("未找到API Key，请前往设置页面配置。"); self.search_button.disabled = True
        self.page.update()

    async def log_message(self, message: str):
        current_log = self.status_log.value if self.status_log.value else ""
        new_log = f"{current_log}{message}\n"
        lines = new_log.splitlines()
        if len(lines) > 100:
            new_log = "\n".join(lines[-100:]) + "\n"
        self.status_log.value = new_log
        self.status_log.cursor_position = len(new_log)
        self.page.update()
    
    async def open_settings_view(self, e):
        if isinstance(self.view_container.controls[0], ft.Column) and len(self.view_container.controls[0].controls) > 1 and hasattr(self.view_container.controls[0].controls[1], 'value'):
            view_title = self.view_container.controls[0].controls[1].value
            if view_title == "贴吧智能回复助手": self.previous_view_name = "initial"
            elif "吧的帖子" in view_title: 
                self.previous_view_name = "thread_list"
                if self.page.scroll and self.thread_list_view.uid in self.page.scroll:
                    self.thread_list_scroll_offset = self.page.scroll[self.thread_list_view.uid].offset
            else: self.previous_view_name = "analysis"
        else: self.previous_view_name = "analysis"
        self.view_container.controls = [self.build_settings_view()]
        self.api_key_input.value = self.settings.get("api_key", "")
        self._populate_model_dropdowns(self.settings.get("available_models", []))
        self.analyzer_model_dd.value = self.settings.get("analyzer_model")
        self.generator_model_dd.value = self.settings.get("generator_model")
        self.save_prompts_button.disabled = True
        await self.log_message("已打开设置页面。"); self.validate_settings(None); self.page.update()

    async def close_settings_view(self, e):
        if self.previous_view_name == "initial": await self.back_to_initial(e)
        elif self.previous_view_name == "thread_list": await self.back_to_thread_list(e)
        elif self.previous_view_name == "analysis": self.view_container.controls = [self.build_analysis_view()]; self.page.update()

    def on_prompt_change(self, e):
        self.save_prompts_button.disabled = False
        self.page.update()

    async def save_prompts_click(self, e):
        current_prompts = core.PROMPTS
        
        # 遍历 UI 控件，用其值更新字典
        for key_path, text_field in self.prompt_text_fields.items():
            temp_dict = current_prompts
            for i, key in enumerate(key_path):
                if i == len(key_path) - 1: # 到达路径末端
                    temp_dict[key] = text_field.value
                else:
                    temp_dict = temp_dict[key]
        
        core.save_prompts(current_prompts)
        self.save_prompts_button.disabled = True
        self.page.open(ft.SnackBar(ft.Text("Prompts 保存成功！"), bgcolor=ft.Colors.GREEN))
        await self.log_message("Prompts 已更新并保存。")
        self.page.update()

    async def restore_prompts_click(self, e):
        self.progress_ring.visible = True; self.page.update()

        success, msg = core.restore_default_prompts()
        await self.log_message(msg)

        if success:
            self.view_container.controls = [self.build_settings_view()]
            self.page.open(ft.SnackBar(ft.Text("已恢复默认 Prompts！"), bgcolor=ft.Colors.BLUE))
        else:
            self.page.open(ft.SnackBar(ft.Text(f"恢复失败: {msg}"), bgcolor=ft.Colors.RED))

        self.progress_ring.visible = False
        self.save_prompts_button.disabled = True
        self.page.update()

    async def fetch_models_click(self, e):
        api_key = self.api_key_input.value.strip()
        if not api_key:
            await self.log_message("请输入API Key后再获取模型。")
            return

        self.progress_ring.visible = True; self.fetch_models_button.disabled = True; self.page.update()

        success, result = await core.fetch_gemini_models(api_key)
        if success:
            await self.log_message(f"成功获取 {len(result)} 个可用模型！正在刷新UI...")
            self.settings["available_models"] = result; self.view_container.controls = [self.build_settings_view()]
            self.api_key_input.value = api_key; self.progress_ring.visible = False; self.fetch_models_button.disabled = False
            self.page.open(ft.SnackBar(ft.Text("模型列表获取并刷新成功!"), bgcolor=ft.Colors.GREEN))
        else:
            self.progress_ring.visible = False; self.fetch_models_button.disabled = False
            await self.log_message(f"获取模型失败: {result}")
            self.page.open(ft.SnackBar(ft.Text(f"获取失败: {result}"), bgcolor=ft.Colors.RED))

        self.validate_settings(None)
        self.page.update()

    async def save_settings_click(self, e):
        self.settings["api_key"] = self.api_key_input.value.strip()
        self.settings["analyzer_model"] = self.analyzer_model_dd.value
        self.settings["generator_model"] = self.generator_model_dd.value
        core.save_settings(self.settings); await self.log_message("设置已保存！")
        if self.settings["api_key"]:
            try:
                self.gemini_client = genai.Client(api_key=self.settings["api_key"]); await self.log_message("Gemini Client 已使用新设置重新初始化。")
                self.search_button.disabled = False
            except Exception as ex: await self.log_message(f"新Key无效: {ex}"); self.search_button.disabled = True
        else: self.search_button.disabled = True
        self.page.open(ft.SnackBar(ft.Text("设置已保存并应用!"), bgcolor=ft.Colors.GREEN))
        self.save_settings_button.disabled = True; self.page.update()

    def _populate_model_dropdowns(self, model_list: list):
        options = [ft.dropdown.Option(model) for model in model_list]
        self.analyzer_model_dd.options = options.copy(); self.generator_model_dd.options = options.copy()

    def validate_settings(self, e):
        is_valid = (self.api_key_input.value and self.analyzer_model_dd.value and self.generator_model_dd.value)
        self.save_settings_button.disabled = not is_valid; self.page.update()

    async def select_thread(self, e):
        # Save scroll offset
        if self.page.scroll and self.thread_list_view.uid in self.page.scroll:
            self.thread_list_scroll_offset = self.page.scroll[self.thread_list_view.uid].offset
        else:
            self.thread_list_scroll_offset = 0.0

        clicked_item = e.control.data

        self.selected_thread = clicked_item

        # Show loading state
        self.view_container.controls = [self.build_analysis_view()]
        self.progress_ring.visible = True
        self.preview_display.controls.clear()
        self.preview_display.controls.append(ft.Row([ft.ProgressRing(), ft.Text("正在加载帖子内容...")], alignment=ft.MainAxisAlignment.CENTER))
        self.page.update()

        # --- Analysis Cache Logic ---
        current_tid = self.selected_thread.tid
        self.current_analysis_tid = None # Reset the current analysis TID

        if current_tid in self.analysis_cache:
            # Load from cache
            await self.log_message(f"从缓存加载TID {current_tid}的分析结果。")
            cached_result = self.analysis_cache[current_tid]
            analysis_str = f"## 讨论状况分析结果 (缓存)\n```json\n{json.dumps(cached_result, ensure_ascii=False, indent=2)}\n```"
            self.analysis_display.value = analysis_str
            self.current_analysis_tid = current_tid # Set the current analysis to this TID
        else:
            self.analysis_display.value = "点击上方“分析讨论状况”按钮以开始"

        can_generate = self.current_analysis_tid == self.selected_thread.tid
        self.mode_selector.disabled = not can_generate
        self.generate_button.disabled = not can_generate
        self.analyze_button.disabled = False
        self.page.update()

        tid_to_fetch = clicked_item.tid

        async with tb.Client() as tieba_client:
            posts, all_comments = await core.fetch_full_thread_data(tieba_client, tid_to_fetch, self.log_message)

        # Determine the `thread_object` to use for formatting
        if not isinstance(clicked_item, tb_typing.Thread):
            main_post_content = posts[0].contents if posts else []
            self.selected_thread = tb_typing.Thread(
                title=clicked_item.title,
                tid=clicked_item.tid,
                user=None, # Search result doesn't provide a full user object
                contents=main_post_content
            )

        # Hide loading ring
        self.progress_ring.visible = False
        self.preview_display.controls.clear()

        if not posts:
            self.preview_display.controls.append(ft.Text(f"❌ 未能加载帖子 T-ID:{tid_to_fetch} 的内容。"))
            self.page.update()
            return
            
        self.discussion_text = core.format_discussion_text(self.selected_thread, posts, all_comments)
        self._build_rich_preview(self.selected_thread, posts, all_comments)
        
        self.analyze_button.disabled = False
        self.page.update()

        # Scroll preview to top
        if self.page.scroll and self.preview_display.uid in self.page.scroll:
             self.page.scroll[self.preview_display.uid].scroll_to(offset=0, duration=300)
        self.page.update()

    def _build_rich_preview(self, thread: tb_typing.Thread, posts: list[tb_typing.Post], all_comments: dict[int, list[tb_typing.Comment]]):
        """遍历数据, 构建帖子预览UI"""
        lz_user_name = thread.user.user_name if thread.user and hasattr(thread.user, 'user_name') else ''
        
        # 主楼 (楼主)
        main_post_content = core.format_contents(thread.contents).strip()
        if main_post_content:
            main_post_widget = self._create_post_widget(lz_user_name, main_post_content, "主楼", is_lz=True)
            self.preview_display.controls.append(main_post_widget)
        
        # 回复和楼中楼
        for post in posts:
            if post.floor == 1: continue
            post_content = core.format_contents(post.contents).strip()
            if not post_content: continue

            user_name = post.user.user_name if post.user and hasattr(post.user, 'user_name') else '未知用户'
            is_lz = (user_name == lz_user_name)
            
            reply_widget = self._create_post_widget(user_name, post_content, f"{post.floor}楼", is_lz=is_lz)
            self.preview_display.controls.append(reply_widget)

            # 楼中楼
            if post.pid in all_comments:
                comment_container = ft.Column(spacing=5)
                for j, comment in enumerate(all_comments[post.pid]):
                    comment_content = core.format_contents(comment.contents).strip()
                    if not comment_content: continue
                    
                    comment_user_name = comment.user.user_name if comment.user and hasattr(comment.user, 'user_name') else '未知用户'
                    is_comment_lz = (comment_user_name == lz_user_name)
                    
                    comment_widget = self._create_post_widget(comment_user_name, comment_content, f"回复", is_lz=is_comment_lz, is_comment=True)
                    comment_container.controls.append(comment_widget)
                
                # 将整个楼中楼区块作为一个控件添加
                self.preview_display.controls.append(
                    ft.Container(
                        content=comment_container,
                        padding=ft.padding.only(left=20, top=5, bottom=10)
                    )
                )

        self.preview_display.scroll_to(offset=0) # 滚动到顶部
        self.page.update()

    async def analyze_thread_click(self, e):
        current_tid = self.selected_thread.tid
        self.progress_ring.visible = True; self.analyze_button.disabled = True; self.mode_selector.disabled = True; self.generate_button.disabled = True; self.analysis_display.value = "⏳ 正在进行AI分析，请稍候..."; self.page.update()
        self.analysis_result = await core.analyze_stance(self.gemini_client, self.discussion_text, self.settings["analyzer_model"], self.log_message)
        self.progress_ring.visible = False; self.analyze_button.disabled = False

        if self.analysis_result and "error" not in self.analysis_result:
            self.analysis_cache[current_tid] = self.analysis_result
            self.current_analysis_tid = current_tid
            analysis_str = f"## 讨论状况分析结果\n```json\n{json.dumps(self.analysis_result, ensure_ascii=False, indent=2)}\n```"
            self.analysis_display.value = analysis_str; self.mode_selector.disabled = False; self.generate_button.disabled = False
        else:
            error_msg = self.analysis_result.get("error", "未知错误")
            self.analysis_display.value = f"❌ 分析失败:\n\n{error_msg}";
        self.page.update()

    async def generate_reply_click(self, e):
        current_tid = self.selected_thread.tid
        current_analysis = self.analysis_cache.get(current_tid)
        if not current_analysis:
            await self.log_message("错误：未找到当前帖子的分析结果，无法生成回复。")
            return
        self.current_mode = self.mode_selector.value
        if not self.current_mode: await self.log_message("请先选择一个回复模型！"); return
        if self.current_mode == "自定义模型":
            self.custom_viewpoint = self.custom_view_input.value.strip()
            if not self.custom_viewpoint: await self.log_message("使用自定义模型时，观点不能为空！"); return
        else: self.custom_viewpoint = None
        self.progress_ring.visible = True; self.generate_button.disabled = True; self.copy_button.disabled = True; self.reply_display.value = "⏳ 生成中，请稍候..."; self.page.update()
        generated_reply = await core.generate_reply(self.gemini_client, self.discussion_text, current_analysis, self.current_mode, self.settings["generator_model"], self.log_message, custom_viewpoint=self.custom_viewpoint)
        self.reply_display.value = generated_reply; self.progress_ring.visible = False; self.generate_button.disabled = False; self.copy_button.disabled = not bool(generated_reply); self.page.update(); await asyncio.sleep(0.05)
        if self.page.scroll and self.preview_display.uid in self.page.scroll:
            self.page.scroll[self.preview_display.uid].scroll_to(offset=0, duration=300)

    async def search_tieba(self, e):
        self.current_page_num = 1
        query = self.search_query_input.value.strip()
        self.current_search_query = query if query else None
        await self._fetch_and_display_threads()
        self.view_container.controls = [self.build_thread_list_view()]
        self.page.update()

    async def load_next_page(self, e): self.current_page_num += 1; await self._fetch_and_display_threads()
    async def load_prev_page(self, e):
        if self.current_page_num > 1: self.current_page_num -= 1; await self._fetch_and_display_threads()
    async def _fetch_and_display_threads(self):
        tieba_name = self.tieba_name_input.value.strip()
        if not tieba_name:
            await self.log_message("错误：贴吧名称不能为空。")
            return
        if not self.gemini_client:
            await self.log_message("Gemini客户端未初始化，请先在设置中配置有效的API Key。")
            return

        self.progress_ring.visible = True
        self.search_button.disabled = True
        self.prev_page_button.disabled = True
        self.next_page_button.disabled = True
        self.page.update()

        async with tb.Client() as tieba_client:
            # Check if we are in search mode or browse mode
            if self.current_search_query:
                # SEARCH MODE
                self.threads = await core.search_threads_by_page(
                    tieba_client, tieba_name, self.current_search_query, self.current_page_num, self.log_message
                )
            else:
                sort_value_str = self.sort_type_dropdown.value
                
                try:
                    sort_type = ThreadSortType(int(sort_value_str))
                except (ValueError, TypeError):
                    await self.log_message(f"警告：无效的排序值 '{sort_value_str}'。将使用默认排序。")
                    sort_type = ThreadSortType.REPLY

                self.threads = await core.fetch_threads_by_page(
                    tieba_client, tieba_name, self.current_page_num, sort_type, self.log_message
                )

        self._update_thread_list_view()
        self.progress_ring.visible = False
        self.search_button.disabled = False
        self.page_num_display.value = f"第 {self.current_page_num} 页"
        self.prev_page_button.disabled = self.current_page_num <= 1
        self.next_page_button.disabled = not self.threads
        self.page.update()

    def _update_thread_list_view(self):
        self.thread_list_view.controls.clear()
        if not self.threads:
            self.thread_list_view.controls.append(ft.Text("这一页没有找到帖子。", text_align=ft.TextAlign.CENTER))
            self.page.update()
            return

        for thread in self.threads:
            # Initialize variables with default values
            user_name = "未知用户"
            reply_num_text = "N/A"
            title = thread.title

            if hasattr(thread, 'user') and thread.user is not None:
                user_name = thread.user.user_name
                reply_num_text = str(thread.reply_num)

            elif hasattr(thread, 'show_name'):
                user_name = thread.show_name

            list_tile = ft.ListTile(
                leading=ft.Icon(ft.Icons.ARTICLE_OUTLINED),
                title=ft.Text(f"{title}", weight=ft.FontWeight.BOLD),
                subtitle=ft.Text(f"作者: {user_name} | 回复数: {reply_num_text}"),
                on_click=self.select_thread,
                data=thread
            )
            self.thread_list_view.controls.append(list_tile)

        self.page.update()

    async def on_mode_change(self, e): self.current_mode = e.control.value; self.custom_view_input.visible = (self.current_mode == "自定义模型"); self.page.update()
    async def copy_reply_click(self, e): self.page.set_clipboard(self.reply_display.value); self.page.open(ft.SnackBar(ft.Text("回复已复制到剪贴板!"), duration=2000)); self.page.update()
    async def back_to_initial(self, e): self.view_container.controls = [self.build_initial_view()]; self.page.update()
    async def back_to_thread_list(self, e): 
        self.view_container.controls = [self.build_thread_list_view()]; self.page.update(); await asyncio.sleep(0.1)
        if self.page.scroll and self.thread_list_view.uid in self.page.scroll: 
            self.page.scroll[self.thread_list_view.uid].scroll_to(offset=self.thread_list_scroll_offset, duration=100)
        self.page.update()

async def main(page: ft.Page):
    app = TiebaGPTApp(page)
    page.add(app.view_container)
    await app.initialize_app()

if __name__ == "__main__":
    ft.app(target=main)