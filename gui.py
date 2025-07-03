import flet as ft
import asyncio
import os
import json
import uuid
from google import genai
from enum import Enum, auto
import core_logic as core
import aiotieba as tb
from aiotieba import ThreadSortType
from aiotieba import typing as tb_typing

class LogLevel(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()

class TiebaGPTApp:

    LOG_LEVEL_COLOR_MAP = {
        LogLevel.INFO: "on_surface_variant",
        LogLevel.WARNING: "tertiary",
        LogLevel.ERROR: "error",
    }

    LOG_LEVEL_ICON_MAP = {
        LogLevel.INFO: ft.Icons.INFO_OUTLINE,
        LogLevel.WARNING: ft.Icons.WARNING_AMBER_ROUNDED,
        LogLevel.ERROR: ft.Icons.ERROR_OUTLINE,
    }

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
        self.discussion_text = ""; self.analysis_result = None; self.current_mode_id = None
        self.custom_input = None; self.current_page_num = 1; self.thread_list_scroll_offset = 0.0
        self.analysis_cache = {}
        self.current_analysis_tid = None
        self.current_search_query = None
        self.current_post_page = 1
        self.total_post_pages = 1

        # --- UI 控件 ---
        # -- 导航 --
        self.navigation_rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            min_extended_width=400,
            group_alignment=-0.9,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.HOME_OUTLINED, selected_icon=ft.Icons.HOME, label="主页"),
                ft.NavigationRailDestination(icon=ft.Icons.INSIGHTS_OUTLINED, selected_icon=ft.Icons.INSIGHTS, label="分析与回复"),
                ft.NavigationRailDestination(icon=ft.Icons.MODE_EDIT_OUTLINE, selected_icon=ft.Icons.MODE_EDIT, label="模式编辑"),
                ft.NavigationRailDestination(icon=ft.Icons.SETTINGS_OUTLINED, selected_icon=ft.Icons.SETTINGS, label="设置"),
                ft.NavigationRailDestination(icon=ft.Icons.INFO_OUTLINE, selected_icon=ft.Icons.INFO, label="关于")
            ],
            on_change=self.navigate,
        )
        
        # -- 通用 --
        self.status_log = ft.ListView(expand=True, spacing=5)
        self.copy_log_button = ft.IconButton(icon=ft.Icons.COPY_ALL, on_click=self.copy_log_click, tooltip="复制所有日志")
        self.progress_ring = ft.ProgressRing(visible=False)
        
        # -- 主页 --
        self.tieba_name_input = ft.TextField(label="贴吧名称", width=250)
        self.search_query_input = ft.TextField(label="帖子关键词 (可选)", hint_text="留空则按排序浏览", width=250, expand=False)
        self.sort_type_dropdown = ft.Dropdown(
            label="排序方式", width=150, expand=False,
            options=[
                ft.dropdown.Option(key=ThreadSortType.REPLY, text="按回复时间"),
                ft.dropdown.Option(key=ThreadSortType.HOT, text="热门排序"),
                ft.dropdown.Option(key=ThreadSortType.CREATE, text="按发布时间"),
            ],
            value=ThreadSortType.REPLY,
        )
        self.search_button = ft.ElevatedButton("获取帖子", on_click=self.search_tieba, icon=ft.Icons.FIND_IN_PAGE)
        self.thread_list_view = ft.ListView(expand=1, spacing=10, auto_scroll=False)
        self.prev_page_button = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_LEFT, on_click=self.load_prev_page, tooltip="上一页", disabled=True)
        self.next_page_button = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_RIGHT, on_click=self.load_next_page, tooltip="下一页", disabled=True)
        self.page_num_display = ft.Text(f"第 {self.current_page_num} 页", weight=ft.FontWeight.BOLD)
        self.main_view_content_area = ft.Container(expand=True)
        
        # -- 分析页 --
        self.preview_display = ft.ListView(expand=True, spacing=10, auto_scroll=False)
        self.analysis_display = ft.Markdown(selectable=True, code_theme="atom-one-dark")
        self.reply_display = ft.Markdown(selectable=True, code_theme="atom-one-light")
        self.analyze_button = ft.ElevatedButton("分析整个帖子", icon=ft.Icons.INSIGHTS_ROUNDED, on_click=self.analyze_thread_click, tooltip="对整个帖子进行分批AI分析", disabled=True)
        self.analysis_progress_bar = ft.ProgressBar(visible=False)
        self.mode_selector = ft.Dropdown(label="回复模式", on_change=self.on_mode_change, disabled=True)
        self.custom_view_input = ft.TextField(label="请输入此模式所需的自定义内容", multiline=True, max_lines=3, visible=False)
        self.generate_button = ft.ElevatedButton("生成回复", on_click=self.generate_reply_click, icon=ft.Icons.AUTO_AWESOME, disabled=True)
        self.generate_reply_ring = ft.ProgressRing(visible=False, width=16, height=16)
        self.copy_button = ft.IconButton(icon=ft.Icons.CONTENT_COPY_ROUNDED, tooltip="复制回复内容", on_click=self.copy_reply_click, disabled=True)
        self.reply_draft_input = ft.TextField(label="或在此处输入您的回复草稿进行优化",multiline=True,min_lines=3, max_lines=5, on_change=self.on_draft_input_change)
        self.optimize_button = ft.ElevatedButton("优化回复", on_click=self.optimize_reply_click,icon=ft.Icons.AUTO_FIX_HIGH, disabled=True)
        self.prev_post_page_button = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_LEFT, on_click=self.load_prev_post_page, tooltip="上一页", disabled=True)
        self.next_post_page_button = ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_RIGHT, on_click=self.load_next_post_page, tooltip="下一页", disabled=True)
        self.post_page_display = ft.Text("第 1 / 1 页", weight=ft.FontWeight.BOLD)
        
        # -- 设置页控件 ---
        self.api_key_input = ft.TextField(label="Gemini API Key", password=True, can_reveal_password=True, on_change=self.validate_settings)
        self.save_api_key_switch = ft.Switch(label="在配置文件中保存API Key (有安全风险)",value=False,on_change=self.validate_settings)
        self.analyzer_model_dd = ft.Dropdown(label="分析模型", hint_text="选择一个分析模型", on_change=self.validate_settings, expand=True)
        self.generator_model_dd = ft.Dropdown(label="生成模型", hint_text="选择一个生成模型", on_change=self.validate_settings, expand=True)
        self.model_selection_row = ft.Row(controls=[self.analyzer_model_dd, self.generator_model_dd], spacing=20)
        self.fetch_models_button = ft.ElevatedButton("测试Key并获取模型", on_click=self.fetch_models_click, icon=ft.Icons.CLOUD_DOWNLOAD)
        self.fetch_models_ring = ft.ProgressRing(visible=False, width=16, height=16)
        self.color_seed_input = ft.TextField(label="主题种子颜色 (Material You)",hint_text="输入颜色名 (如 blue) 或HEX值 (#6750A4)",on_change=self.validate_settings)
        self.pages_per_call_slider = ft.Slider(min=1, max=10, divisions=9,label="每次分析的页数: {value}",on_change=self.validate_settings)
        self.save_settings_button = ft.ElevatedButton("保存设置", on_click=self.save_settings_click, icon=ft.Icons.SAVE, disabled=True)
        self.prompt_text_fields = {}
        self.save_prompts_button = ft.ElevatedButton("保存 Prompts", on_click=self.save_prompts_click, icon=ft.Icons.SAVE_ALT, disabled=True)
        self.restore_prompts_button = ft.ElevatedButton("恢复默认 Prompts", on_click=self.restore_prompts_click, icon=ft.Icons.RESTORE)
        self.reply_modes_list = ft.Column(spacing=5, scroll=ft.ScrollMode.ADAPTIVE, expand=True)
        self.settings_tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[ft.Tab(text="通用设置"), ft.Tab(text="高级设置")],
            on_change=self.on_settings_tab_change
        )
        self.settings_content_area = ft.Container(padding=ft.padding.only(top=10), expand=True)

        # -- 关于页控件 --
        self.readme_display = ft.Markdown(selectable=True, expand=True)
        self.about_progress_ring = ft.ProgressRing()
        
        # -- 主视图容器 --
        self.content_view = ft.Column([self.build_main_view()], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def _create_prompt_editor(self, key_path: tuple, value: any) -> ft.Control:
        label = " -> ".join(key_path)
        is_list = isinstance(value, list)
        display_value = "\n".join(value) if is_list else str(value)
        tf = ft.TextField(label=label, value=display_value, multiline=True, min_lines=3 if is_list else 2, max_lines=8, text_size=12, on_change=self.on_prompt_change, data={'is_list': is_list})
        self.prompt_text_fields[key_path] = tf
        return tf
        
    # --- 视图构建方法 ---
    def build_main_view(self):
        input_row = ft.Row([self.tieba_name_input, self.search_query_input, self.sort_type_dropdown, self.search_button], alignment=ft.MainAxisAlignment.CENTER, spacing=10)
        app_info_row = ft.Row([ft.Text(f"v{self.app_version}", color="primary"), ft.Icon(ft.Icons.CIRCLE, size=8, color=ft.Colors.GREY_400), ft.TextButton(text="GitHub", icon=ft.Icons.CODE, url="https://github.com/LaplaceDemon29/TiebaGPT", tooltip="查看项目源代码")], alignment=ft.MainAxisAlignment.CENTER, spacing=8)
        
        return ft.Column([
            ft.Text("贴吧智能回复助手", style=ft.TextThemeStyle.HEADLINE_MEDIUM), 
            input_row, 
            self.progress_ring, 
            ft.Divider(), 
            self.main_view_content_area,
            ft.Container(content=app_info_row, padding=ft.padding.only(top=10, bottom=5))
        ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)

    def _build_main_view_content(self):
        if not self.threads:
            return self._build_status_log_section(expand=True)
        
        title_text = f"“{self.tieba_name_input.value}”吧的帖子"
        if self.current_search_query: title_text = f"在“{self.tieba_name_input.value}”吧中搜索“{self.current_search_query}”的结果"
        
        return ft.Column([
            ft.Text(title_text, style=ft.TextThemeStyle.HEADLINE_SMALL), 
            ft.Container(self.thread_list_view, border=ft.border.all(1, ft.Colors.OUTLINE), expand=True, border_radius=5, padding=5), 
            ft.Row([self.prev_page_button, self.page_num_display, self.next_page_button], alignment=ft.MainAxisAlignment.CENTER)
        ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)

    def build_analysis_view(self):
        if not self.selected_thread:
            return ft.Column([
                ft.Text("未选择帖子", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                ft.Text("请先从主页选择一个帖子，然后再进入此页面进行分析。"),
                ft.ElevatedButton("返回主页", on_click=self.back_to_main_view, icon=ft.Icons.HOME)
            ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20, alignment=ft.MainAxisAlignment.CENTER)

        self._populate_mode_dropdown()
        preview_nav = ft.Row([self.prev_post_page_button, self.post_page_display, self.next_post_page_button], alignment=ft.MainAxisAlignment.CENTER)
        preview_card = ft.Column(
            controls=[
                ft.Text("帖子预览", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Container(content=self.preview_display, border=ft.border.all(1, ft.Colors.OUTLINE), border_radius=5, padding=10, expand=True),
                preview_nav
            ], expand=True, spacing=10
        )
        analysis_card = ft.Column(
            controls=[
                ft.Text("讨论状况分析", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Container(
                    content=ft.Column(
                        controls=[self.analyze_button, self.analysis_progress_bar, ft.Container(content=ft.Column([self.analysis_display], scroll=ft.ScrollMode.ADAPTIVE), bgcolor=ft.Colors.with_opacity(0.08, "tertiary"), border_radius=ft.border_radius.all(5), padding=ft.padding.all(10), expand=True)],
                        spacing=10, horizontal_alignment=ft.CrossAxisAlignment.STRETCH
                    ), border=ft.border.all(1, ft.Colors.OUTLINE),border_radius=5,padding=10,expand=True
                ),
                ft.Divider(height=10), self._build_status_log_section(height=140)
            ], expand=True, spacing=10, horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        )
        reply_card = ft.Column(
            controls=[
                ft.Text("生成回复", style=ft.TextThemeStyle.TITLE_MEDIUM),
                self.mode_selector, self.custom_view_input, ft.Divider(height=15), self.reply_draft_input, 
                ft.Row([self.generate_button, self.optimize_button, self.copy_button, self.generate_reply_ring], alignment=ft.MainAxisAlignment.CENTER), 
                ft.Divider(height=15), 
                ft.Container(
                    content=ft.Column([self.reply_display], scroll=ft.ScrollMode.ADAPTIVE, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
                    border=ft.border.all(1, ft.Colors.OUTLINE),border_radius=5,padding=10,expand=True,bgcolor=ft.Colors.with_opacity(0.12, "primary")
                )
            ],expand=True, spacing=10
        )
        return ft.Column([
            ft.Row([ft.ElevatedButton("返回帖子列表", on_click=self.back_to_main_view, icon=ft.Icons.ARROW_BACK)]),
            ft.Text(self.selected_thread.title, style=ft.TextThemeStyle.HEADLINE_SMALL, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Divider(),
            ft.Row(controls=[preview_card, analysis_card, reply_card], spacing=10, expand=True)
        ], expand=True, spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def build_mode_editor_view(self):
        self._build_reply_modes_editor_list()
        
        editor_content = ft.Column(
            [
                ft.Text("回复模式编辑器", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Text("在这里添加、删除或修改AI的回复模式。"),
                ft.Row(
                    controls=[
                        ft.ElevatedButton("添加新模式", icon=ft.Icons.ADD, on_click=self.open_mode_dialog),
                        ft.ElevatedButton("导入新模式", icon=ft.Icons.CONTENT_PASTE_GO, on_click=self.open_import_dialog)
                    ], spacing=10
                ),
                ft.Divider(height=10),
                ft.Container(content=self.reply_modes_list, expand=True)
            ],
            spacing=10
        )

        return ft.Column(
            [
                ft.Text("模式编辑", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                ft.Card(
                    elevation=2,
                    content=ft.Container(
                        padding=ft.padding.all(15),
                        content=editor_content
                    )
                )
            ],
            spacing=15,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            expand=True,
            scroll=ft.ScrollMode.ADAPTIVE
        )
    
    def build_settings_view(self):
        self.on_settings_tab_change(None)
        
        return ft.Column([
            ft.Text("设置", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
            self.settings_tabs,
            self.settings_content_area
        ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)

    def _build_general_settings_tab(self):
        return ft.Column(
            [
                ft.Card(
                    elevation=2,
                    content=ft.Container(
                        padding=ft.padding.all(15),
                        content=ft.Column(
                            [
                                self.save_settings_button,
                                ft.Divider(),
                                ft.Container(content=ft.Text("API设置", style=ft.TextThemeStyle.TITLE_MEDIUM), margin=ft.margin.only(top=10)),
                                ft.Text("请在这里配置您的Gemini API。"),
                                self.api_key_input,
                                ft.Row([self.save_api_key_switch,ft.IconButton(icon=ft.Icons.HELP_OUTLINE,icon_color="on_surface_variant",tooltip="警告：开启此项会将您的API Key以明文形式保存在 settings.json 文件中。")]),
                                ft.Row([self.fetch_models_button,self.fetch_models_ring],vertical_alignment=ft.CrossAxisAlignment.CENTER,spacing=10),
                                self.model_selection_row,
                                ft.Divider(), 
                                ft.Container(content=ft.Text("分析设置", style=ft.TextThemeStyle.TITLE_MEDIUM), margin=ft.margin.only(top=10)),
                                ft.Text("调整每次调用AI进行分析时读取的帖子页数。", size=12, color=ft.Colors.GREY_700),
                                self.pages_per_call_slider,
                                ft.Divider(),
                                ft.Container(content=ft.Text("样式设置", style=ft.TextThemeStyle.TITLE_MEDIUM), margin=ft.margin.only(top=10)),
                                self.color_seed_input
                            ], spacing=15
                        )
                    )
                )
            ], spacing=15, horizontal_alignment=ft.CrossAxisAlignment.STRETCH, scroll=ft.ScrollMode.ADAPTIVE
        )

    def _build_advanced_settings_tab(self):
        self.prompt_text_fields.clear()
        prompt_panel_content = self._build_prompt_editors()
        
        return ft.Column(
            [
                ft.Card(
                    elevation=2,
                    content=ft.Container(
                        padding=ft.padding.all(15),
                        content=ft.Column(
                            [
                                ft.Text("警告：不正确的修改可能导致程序功能异常。请仅修改文本内容。", color="error"),
                                ft.Row([self.save_prompts_button, self.restore_prompts_button], spacing=20),
                                ft.Divider(height=20),
                                ft.Container(
                                    content=prompt_panel_content
                                )
                            ],
                            spacing=15
                        )
                    )
                )
            ], spacing=15, horizontal_alignment=ft.CrossAxisAlignment.STRETCH, expand=True, scroll=ft.ScrollMode.ADAPTIVE
        )

    def _build_prompt_editors(self) -> ft.Column:
        controls_list = []
        prompts = core.PROMPTS
        if 'stance_analyzer' in prompts:
            sa_prompts = prompts['stance_analyzer']
            controls_list.append(ft.Text("讨论分析器通用规则", style=ft.TextThemeStyle.TITLE_MEDIUM))
            controls_list.append(self._create_prompt_editor(('stance_analyzer', 'system_prompt'), sa_prompts.get('system_prompt', '')))
            controls_list.append(self._create_prompt_editor(('stance_analyzer', 'tasks'), sa_prompts.get('tasks', [])))
        if 'analysis_summarizer' in prompts:
            as_prompts = prompts['analysis_summarizer']
            controls_list.append(ft.Divider(height=10))
            controls_list.append(ft.Text("分析总结器通用规则", style=ft.TextThemeStyle.TITLE_MEDIUM))
            controls_list.append(self._create_prompt_editor(('analysis_summarizer', 'system_prompt'), as_prompts.get('system_prompt', '')))
            controls_list.append(self._create_prompt_editor(('analysis_summarizer', 'tasks'), as_prompts.get('tasks', [])))
        if 'reply_generator' in prompts:
            rg_prompts = prompts['reply_generator']
            controls_list.append(ft.Divider(height=10))
            controls_list.append(ft.Text("回复生成器通用规则", style=ft.TextThemeStyle.TITLE_MEDIUM))
            if 'common_rules' in rg_prompts:
                cr_prompts = rg_prompts.get('common_rules', {})
                controls_list.append(self._create_prompt_editor(('reply_generator', 'common_rules', 'rules'), cr_prompts.get('rules', [])))
        if 'reply_optimizer' in prompts:
            ro_prompts = prompts['reply_optimizer']
            controls_list.append(ft.Divider(height=10))
            controls_list.append(ft.Text("回复优化器通用规则", style=ft.TextThemeStyle.TITLE_MEDIUM))
            controls_list.append(self._create_prompt_editor(('reply_optimizer', 'system_prompt'), ro_prompts.get('system_prompt', [])))
        if 'mode_generator' in prompts and 'system_prompt' in prompts['mode_generator']:
            controls_list.append(ft.Divider(height=20))
            controls_list.append(ft.Text("回复模式生成与优化器 Prompt", style=ft.TextThemeStyle.TITLE_SMALL))
            controls_list.append(ft.Text("警告：修改此处将改变模式编辑‘AI生成’和‘AI优化’按钮的行为。", color="error", size=11))
            controls_list.append(self._create_prompt_editor(('mode_generator', 'system_prompt'), prompts['mode_generator']['system_prompt']))
        if 'mode_optimizer' in prompts and 'system_prompt' in prompts['mode_optimizer']:
            controls_list.append(self._create_prompt_editor(('mode_optimizer', 'system_prompt'), prompts['mode_optimizer']['system_prompt']))

        return ft.Column(controls_list, spacing=15)
    
    def _build_reply_modes_editor_list(self):
        self.reply_modes_list.controls.clear()
        sorted_modes = core.get_sorted_reply_modes()
        default_mode_ids = core.get_default_mode_ids()
        for mode_id, config in sorted_modes:
            mode_name = config.get('name', '未命名模式')
            is_built_in = mode_id in default_mode_ids
            info_row_controls = [ft.Text(mode_name, weight=ft.FontWeight.BOLD)]
            if config.get('is_custom', False): info_row_controls.append(self.create_tag("需输入自定义内容", "primary"))
            info_row_controls.append(self.create_tag("内置模式" if is_built_in else "自定义模式","tertiary"))
            self.reply_modes_list.controls.append(
                ft.Card(content=ft.Container(padding=ft.padding.symmetric(vertical=5, horizontal=10),
                    content=ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Icon(config.get('icon', "settings_suggest"), color="primary"),
                            ft.Column([ft.Row(info_row_controls), ft.Text(config.get('description', 'N/A'), max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, size=12, color="on_surface_variant")], expand=True, spacing=2),
                            ft.Row([
                                ft.IconButton(ft.Icons.SHARE, tooltip="分享此模式", on_click=self.share_mode_click, data=mode_id, icon_color="primary"),
                                ft.IconButton(ft.Icons.EDIT, tooltip="编辑此模式", on_click=self.open_mode_dialog, data=mode_id),
                                ft.IconButton(ft.Icons.DELETE_FOREVER, tooltip="删除此模式 (内置模式不可删除)", on_click=self.delete_mode_click, data=mode_id, icon_color="error", disabled=is_built_in),
                            ])
                        ]
                    )
                ))
            )
        if self.page: self.page.update()

    def _build_status_log_section(self, expand: bool = False, height: int = None) -> ft.Column:
        log_container = ft.Container(self.status_log,border=ft.border.all(1, ft.Colors.OUTLINE),border_radius=5,padding=10)
        if expand: log_container.expand = True
        elif height is not None: log_container.height = height
        return ft.Column(controls=[ft.Row([ft.Text("状态日志:", style=ft.TextThemeStyle.TITLE_MEDIUM),self.copy_log_button],alignment=ft.MainAxisAlignment.SPACE_BETWEEN,vertical_alignment=ft.CrossAxisAlignment.CENTER),log_container],expand=expand)

    def build_about_view(self):
        info_card = ft.Card(
            elevation=2,
            content=ft.Container(
                padding=ft.padding.all(15),
                content=ft.Column(
                    [
                        ft.Text("关于 TiebaGPT", style=ft.TextThemeStyle.TITLE_LARGE),
                        ft.Text(f"版本: {self.app_version}", weight=ft.FontWeight.BOLD),
                        ft.Row(
                            [
                                ft.Text("作者: LaplaceDemon29"),
                                ft.TextButton(
                                    text="GitHub 项目地址",
                                    icon=ft.Icons.CODE,
                                    url="https://github.com/LaplaceDemon29/TiebaGPT"
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.START
                        ),
                    ]
                )
            )
        )
        
        readme_card = ft.Card(
            elevation=2,
            content=ft.Container(
                padding=ft.padding.all(15),
                content=ft.Column(
                    [
                        ft.Text("项目说明 (README.md)", style=ft.TextThemeStyle.TITLE_MEDIUM),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Row([self.about_progress_ring], alignment=ft.MainAxisAlignment.CENTER),
                            expand=True,
                            alignment=ft.alignment.center
                        )
                    ]
                ),
                expand=True
            )
        )

        return ft.Column(
            [
                ft.Text("关于", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                info_card,
                readme_card,
            ],
            spacing=15,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            expand=True,
            scroll=ft.ScrollMode.ADAPTIVE
        )

    async def load_about_content(self):
        content_container = self.content_view.controls[0].controls[-1].content.content.controls[-1]
        content_container.content = ft.Row([self.about_progress_ring], alignment=ft.MainAxisAlignment.CENTER)
        self.about_progress_ring.visible = True
        self.page.update()
        success, readme_text = await core.get_readme_content()
        if not success: self.log_message(readme_text, LogLevel.ERROR)
        self.readme_display.value = readme_text
        self.about_progress_ring.visible = False
        content_container.content = self.readme_display
        self.page.update()

    def _create_post_widget_by_user(self, user, content_str: str, floor_text: str, lz_user_name: str, is_comment: bool = False) -> ft.Control:
        user_name = getattr(user, 'user_name', '未知用户'); nick_name = getattr(user, 'nick_name', '无昵称')
        is_lz = user_name == lz_user_name; is_bawu = getattr(user, 'is_bawu', False)
        user_level = getattr(user, 'level', None); ip = getattr(user, 'ip', None)
        user_info_row = ft.Row(controls=[ft.Icon(ft.Icons.ACCOUNT_CIRCLE, color="primary", size=20), ft.Text(f"{nick_name}({user_name})", weight=ft.FontWeight.BOLD, color="primary")], alignment=ft.MainAxisAlignment.START, spacing=5)
        if is_lz: user_info_row.controls.append(self.create_tag("楼主", "primary"))
        if is_bawu: user_info_row.controls.append(self.create_tag("吧务", "error"))
        if user_level is not None and user_level > 0: user_info_row.controls.append(self.create_tag(f"Lv.{user_level}", "secondary"))
        if ip: user_info_row.controls.append(self.create_tag(ip, "tertiary"))
        header_row = ft.Row(controls=[user_info_row, ft.Text(floor_text, color="on_surface_variant", size=12)], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        content_display = ft.Text(content_str, selectable=True); post_column = ft.Column(controls=[header_row, content_display], spacing=5)
        if is_comment:
            bgcolor = ft.Colors.with_opacity(0.04, "primary")
            if is_lz: bgcolor = ft.Colors.with_opacity(0.07, "primary")
            return ft.Container(content=post_column, padding=ft.padding.only(left=30, top=8, bottom=8, right=5), border=ft.border.only(left=ft.border.BorderSide(2, "outline_variant")), bgcolor=bgcolor, border_radius=ft.border_radius.all(4))
        else: return ft.Column([post_column, ft.Divider(height=1, thickness=1)])

    def _try_get_effective_api_key(self, from_ui: bool = False) -> str:
        if from_ui and hasattr(self, 'api_key_input'): return self.api_key_input.value.strip()
        saved_key = self.settings.get("api_key", ""); 
        if saved_key: return saved_key
        env_key = os.getenv("GEMINI_API_KEY", ""); 
        if env_key: return env_key
        return ""

    def _get_contrast_colors(self, base_color_role: str, text_opacity: float = 0.9) -> tuple[str, str]:
        if 0 <= text_opacity <= 0.8:
            bg_color = ft.Colors.with_opacity(1 - text_opacity, base_color_role)
            text_color = f"on_{base_color_role}"
            return (text_color, bg_color)
        elif 0.8 < text_opacity <= 1:
            text_color = ft.Colors.with_opacity(text_opacity, base_color_role)
            bg_color = ft.Colors.with_opacity(1.0 - text_opacity, base_color_role)
            return (text_color, bg_color)
        else: raise ValueError(f"Invalid text_opacity")

    def _show_snackbar(self, message: str, color_role: str = "primary", text_opacity: float = 0.1, duration: int = 2000) -> None:
        text_color, bg_color = self._get_contrast_colors(color_role,text_opacity)
        self.page.open(ft.SnackBar(content=ft.Text(message, color=text_color), bgcolor=bg_color, duration=duration))

    def create_tag(self, text: str, color_role: str, text_opacity: float = 0.88, icon: str = None) -> ft.Container:
        controls = []; text_color,bg_color = self._get_contrast_colors(color_role, text_opacity)
        if icon: controls.append(ft.Icon(name=icon, size=12, color=text_color))
        controls.append(ft.Text(text, size=10, weight=ft.FontWeight.BOLD, color=text_color))
        return ft.Container(content=ft.Row(controls, spacing=2, alignment=ft.MainAxisAlignment.CENTER), bgcolor=bg_color, border_radius=100, padding=ft.padding.symmetric(horizontal=6, vertical=2))

    def _rebuild_model_dropdowns(self, models_list: list, preferred_analyzer: str = None, preferred_generator: str = None):
        new_analyzer_dd = ft.Dropdown(label="分析模型", hint_text="选择一个分析模型", on_change=self.validate_settings, expand=True, options=[ft.dropdown.Option(model) for model in models_list])
        new_generator_dd = ft.Dropdown(label="生成模型", hint_text="选择一个生成模型", on_change=self.validate_settings, expand=True, options=[ft.dropdown.Option(model) for model in models_list])
        if models_list:
            analyzer_val = preferred_analyzer or self.settings.get("analyzer_model")
            generator_val = preferred_generator or self.settings.get("generator_model")
            if analyzer_val and analyzer_val in models_list: new_analyzer_dd.value = analyzer_val
            else: new_analyzer_dd.value = next((m for m in models_list if "flash" in m), models_list[0])
            if generator_val and generator_val in models_list: new_generator_dd.value = generator_val
            else: new_generator_dd.value = next((m for m in models_list if "flash" in m), models_list[0])
        self.analyzer_model_dd = new_analyzer_dd; self.generator_model_dd = new_generator_dd
        self.model_selection_row.controls.clear(); self.model_selection_row.controls.append(self.analyzer_model_dd); self.model_selection_row.controls.append(self.generator_model_dd)

    def initialize_app(self):
        self.settings = core.load_settings(); self.log_message("设置已加载。")
        seed_color = self.settings.get("color_scheme_seed"); 
        if seed_color: self.page.theme = ft.Theme(color_scheme_seed=seed_color)
        success, msg = core.load_prompts(); self.log_message(msg, LogLevel.INFO if success else LogLevel.ERROR)
        if not success: self.search_button.disabled = True
        status, user_v, default_v = core.check_prompts_version()
        if status == "NEEDS_UPDATE":
            self.log_message(f"配置需要更新 (用户版本: {user_v}, 最新版本: {default_v})。正在提示用户...")
            self._show_prompt_update_dialog(user_v, default_v); self.log_message("配置更新流程结束。")
        effective_key = self._try_get_effective_api_key()
        if effective_key:
            try:
                self.gemini_client = genai.Client(api_key=effective_key); self.log_message("Gemini Client 初始化成功。")
                self.search_button.disabled = False
            except Exception as e:
                self.log_message(f"使用已配置的Key初始化失败: {e}，请前往设置更新。", LogLevel.ERROR); self.search_button.disabled = True
        else:
            self.log_message("未找到API Key，请前往设置页面配置。", LogLevel.WARNING); self.search_button.disabled = True
        
        self.main_view_content_area.content = self._build_main_view_content()
        self.page.update()

    def log_message(self, message: str, level: LogLevel = LogLevel.INFO):
        log_color = self.LOG_LEVEL_COLOR_MAP.get(level, "on_surface_variant"); log_icon = self.LOG_LEVEL_ICON_MAP.get(level, ft.Icons.INFO_OUTLINE)
        log_entry = ft.Row(controls=[ft.Icon(name=log_icon, color=log_color, size=14), ft.Text(f"[{level.name}] {message}", size=11, selectable=True, color=log_color, expand=True, no_wrap=False)], spacing=5, vertical_alignment=ft.CrossAxisAlignment.START)
        self.status_log.controls.append(log_entry)
        if len(self.status_log.controls) > 100: self.status_log.controls.pop(0)
        if self.status_log.page: self.status_log.update(); self.status_log.scroll_to(offset=-1, duration=300)
        if self.page: self.page.update()

    async def navigate(self, e):
        idx = self.navigation_rail.selected_index if e is None else e.control.selected_index

        self.content_view.controls.clear()
        if idx == 0:
            self.content_view.controls.append(self.build_main_view())
        elif idx == 1:
            self.content_view.controls.append(self.build_analysis_view())
        elif idx == 2:
            self.content_view.controls.append(self.build_mode_editor_view())
        elif idx == 3:
            self.content_view.controls.append(self.build_settings_view())
            self._prepare_settings_view()
        elif idx == 4:
            self.content_view.controls.append(self.build_about_view())
            await self.load_about_content()
        self.page.update()

    def _prepare_settings_view(self):
        env_key = os.getenv("GEMINI_API_KEY", ""); saved_key = self.settings.get("api_key", "")
        self.api_key_input.disabled = False; self.save_api_key_switch.disabled = False
        if saved_key:
            self.api_key_input.value = saved_key; self.api_key_input.hint_text = "已从配置文件加载"
            self.save_api_key_switch.value = True; self.log_message("API Key 已从配置文件加载。")
        elif env_key:
            self.api_key_input.value = env_key; self.api_key_input.hint_text = "已从环境变量加载 (若保存将写入配置文件)"
            self.save_api_key_switch.value = False; self.log_message("API Key 已从环境变量加载。")
        else:
            self.api_key_input.value = ""; self.api_key_input.hint_text = "请输入您的 API Key"
            self.save_api_key_switch.value = False; self.log_message("请在输入框中配置 API Key。")
        self.color_seed_input.value = self.settings.get("color_scheme_seed", "blue")
        self.pages_per_call_slider.value = self.settings.get("pages_per_api_call", 4)
        self._rebuild_model_dropdowns(self.settings.get("available_models"))
        self.save_prompts_button.disabled = True; self.log_message("已打开设置页面。"); self.validate_settings(None); self.page.update()

    def on_settings_tab_change(self, e):
        idx = self.settings_tabs.selected_index
        if idx == 0:
            self.settings_content_area.content = self._build_general_settings_tab()
        elif idx == 1:
            self.settings_content_area.content = self._build_advanced_settings_tab()
        if self.page: self.page.update()

    def on_prompt_change(self, e):
        self.save_prompts_button.disabled = False; self.page.update()

    def save_prompts_click(self, e):
        current_prompts = core.PROMPTS
        for key_path, text_field in self.prompt_text_fields.items():
            temp_dict = current_prompts
            for i, key in enumerate(key_path):
                if i == len(key_path) - 1:
                    if text_field.data and text_field.data.get('is_list', False): temp_dict[key] = [line for line in text_field.value.splitlines() if line.strip()]
                    else: temp_dict[key] = text_field.value
                else: temp_dict = temp_dict.setdefault(key, {})
        core.save_prompts(current_prompts)
        self.save_prompts_button.disabled = True
        self._show_snackbar("Prompts 保存成功！", color_role="primary")
        self.log_message("Prompts 已更新并保存。"); self.page.update()

    def restore_prompts_click(self, e):
        restore_dialog = ft.AlertDialog(modal=True, title=ft.Text("恢复默认 Prompts"), content=ft.Text("请选择恢复方式："), actions_alignment=ft.MainAxisAlignment.END, actions_padding=ft.padding.all(20))
        def handle_full_restore(ev):
            self.progress_ring.visible = True; self.page.close(restore_dialog); self.page.update()
            success, msg = core.restore_default_prompts(); self.log_message(msg, LogLevel.INFO if success else LogLevel.ERROR)
            self.progress_ring.visible = False
            if success:
                self.settings_content_area.content = self._build_advanced_settings_tab()
                self._show_snackbar("已彻底恢复默认 Prompts！",color_role="secondary")
            else: self._show_snackbar(f"恢复失败: {msg}", color_role="error")
            self.page.update()
        def handle_incremental_restore(ev):
            self.progress_ring.visible = True; self.page.close(restore_dialog); self.page.update()
            success, msg = core.merge_default_prompts(); self.log_message(msg, LogLevel.INFO if success else LogLevel.ERROR)
            self.progress_ring.visible = False
            if success:
                self.settings_content_area.content = self._build_advanced_settings_tab()
                self._show_snackbar("增量恢复成功！自定义模式已保留。", color_role="primary")
            else: self._show_snackbar(f"增量恢复失败: {msg}", color_role="error")
            self.page.update()
        restore_dialog.actions = [
            ft.TextButton("取消", on_click=lambda _: self.page.close(restore_dialog)),
            ft.ElevatedButton("增量恢复", tooltip="保留您新增的自定义回复模式", on_click=handle_incremental_restore, icon=ft.Icons.ADD_TASK, bgcolor=ft.Colors.LIGHT_GREEN_100),
            ft.FilledButton("彻底恢复", tooltip="警告：此操作将删除您所有自定义的回复模式", on_click=handle_full_restore, icon=ft.Icons.WARNING_AMBER_ROUNDED, bgcolor=ft.Colors.RED_200),
        ]
        self.page.open(restore_dialog); self.page.update()

    async def fetch_models_click(self, e):
        api_key = self.api_key_input.value.strip()
        if not api_key: self.log_message("请输入API Key后再获取模型。", LogLevel.WARNING); return
        previous_analyzer = self.analyzer_model_dd.value; previous_generator = self.generator_model_dd.value
        self.fetch_models_ring.visible = True; self.fetch_models_button.disabled = True; self.page.update()
        success, result = await core.fetch_gemini_models(api_key)
        if success:
            self.log_message(f"成功获取 {len(result)} 个可用模型！"); self.settings["available_models"] = result
            self._rebuild_model_dropdowns(result, preferred_analyzer=previous_analyzer, preferred_generator=previous_generator)
            self._show_snackbar("模型列表获取并刷新成功!", color_role="primary")
        else:
            self.log_message(f"获取模型失败: {result}", LogLevel.ERROR); self._show_snackbar(f"获取失败: {result}", color_role="error")
        self.fetch_models_ring.visible = False; self.fetch_models_button.disabled = False; self.validate_settings(None); self.page.update()

    def save_settings_click(self, e):
        if self.save_api_key_switch.value:
            self.settings["api_key"] = self.api_key_input.value.strip(); self.log_message("API Key 已明文保存至配置文件。", LogLevel.WARNING)
        else:
            self.settings["api_key"] = ""; self.log_message("API Key 未保存至配置文件。")
        self.settings["analyzer_model"] = self.analyzer_model_dd.value; self.settings["generator_model"] = self.generator_model_dd.value
        self.settings["pages_per_api_call"] = int(self.pages_per_call_slider.value)
        new_seed_color = self.color_seed_input.value.strip(); current_seed_color = self.settings.get("color_scheme_seed", "blue")
        if new_seed_color != current_seed_color:
            try:
                self.page.theme = ft.Theme(color_scheme_seed=new_seed_color) if new_seed_color else None
                self.settings["color_scheme_seed"] = new_seed_color; self.log_message(f"主题颜色已成功更新为: {new_seed_color or '默认'}")
            except Exception as ex:
                self.log_message(f"无效的主题颜色: '{new_seed_color}'。错误: {ex}", LogLevel.ERROR)
                self._show_snackbar(f"颜色 '{new_seed_color}' 无效，主题未更改。", color_role="error"); self.color_seed_input.value = current_seed_color
        core.save_settings(self.settings); self.log_message("设置已保存！")
        current_effective_key = self._try_get_effective_api_key(from_ui=True)
        if current_effective_key:
            try:
                self.gemini_client = genai.Client(api_key=current_effective_key); self.log_message("Gemini Client 已使用新设置重新初始化。")
                self.search_button.disabled = False
            except Exception as ex:
                self.gemini_client = None; self.log_message(f"提供的 Key 无效: {ex}", LogLevel.ERROR); self.search_button.disabled = True
        else:
            self.gemini_client = None; self.search_button.disabled = True
        self._show_snackbar("设置已保存并应用!", color_role="primary"); self.save_settings_button.disabled = True; self.page.update()

    def validate_settings(self, e):
        is_valid = (self.api_key_input.value and self.analyzer_model_dd.value and self.generator_model_dd.value)
        self.save_settings_button.disabled = not is_valid; self.page.update()

    async def select_thread(self, e):
        self.thread_list_scroll_offset = self.page.scroll.get(self.thread_list_view.uid, ft.ScrollMetrics(0,0,0)).offset if self.page.scroll else 0.0
        self.selected_thread = e.control.data
        self.current_post_page = 1
        self.total_post_pages = 1
        self.current_analysis_tid = None

        self.navigation_rail.selected_index = 1
        await self.navigate(None)

        self.progress_ring.visible = True
        self.preview_display.controls.clear()
        self.preview_display.controls.append(ft.Row([ft.ProgressRing(), ft.Text("正在初始化帖子视图...")], alignment=ft.MainAxisAlignment.CENTER))
        self.page.update()
        if self.selected_thread.tid in self.analysis_cache:
            self.log_message(f"从缓存加载TID {self.selected_thread.tid}的完整分析结果。")
            cached_result = self.analysis_cache[self.selected_thread.tid]
            if "summary" in cached_result:
                self.analysis_display.value = f"## 讨论状况摘要 (缓存)\n\n{cached_result['summary']}"
                self.current_analysis_tid = self.selected_thread.tid
            else:
                self.analysis_display.value = "缓存数据格式有误，请重新分析。"
                self.log_message(f"警告: 缓存的TID {self.selected_thread.tid} 数据缺少 'summary' 键。", LogLevel.WARNING)
        else:
            self.analysis_display.value = "点击“分析整个帖子”按钮以开始"
        
        self.mode_selector.disabled = False
        self.generate_button.disabled = not (self.current_analysis_tid == self.selected_thread.tid)
        self.analyze_button.disabled = False
        await self._load_and_display_post_page(True)
    
        self.progress_ring.visible = False
        self.page.update()
    
    async def _load_and_display_post_page(self, init: bool = False):
        if init: self.current_post_page = 1
        else:
            self.prev_post_page_button.disabled = True; self.next_post_page_button.disabled = True; self.preview_display.controls.clear()
            self.preview_display.controls.append(ft.Row([ft.ProgressRing(), ft.Text(f"加载第 {self.current_post_page} 页...")]))
            self.page.update()
        async with tb.Client() as tieba_client:
            thread_obj, posts_obj, all_comments = await core.fetch_full_thread_data(tieba_client, self.selected_thread.tid, self.log_message, page_num=self.current_post_page)
        self.preview_display.controls.clear()
        if not thread_obj or not posts_obj:
            self.log_message(f"错误：无法加载TID {self.selected_thread.tid} 的第 {self.current_post_page} 页。", LogLevel.ERROR)
            self.preview_display.controls.append(ft.Text(f"加载第 {self.current_post_page} 页失败。")); return
        if init: 
            self.total_post_pages = posts_obj.page.total_page
            if not isinstance(self.selected_thread, tb_typing.Thread) or not self.selected_thread.contents: self.selected_thread = thread_obj
        self._build_rich_preview(self.selected_thread, posts_obj.objs, all_comments)
        main_post_text = core.format_main_post_text(self.selected_thread); discussion_part_text = core.format_discussion_text(self.selected_thread, posts_obj.objs, all_comments)
        self.discussion_text = f"{main_post_text}\n{discussion_part_text}"
        self.post_page_display.value = f"第 {self.current_post_page} / {self.total_post_pages} 页"
        self.prev_post_page_button.disabled = self.current_post_page <= 1; self.next_post_page_button.disabled = self.current_post_page >= self.total_post_pages
        self.page.update()
        if self.preview_display.uid in (self.page.scroll or {}): self.page.scroll[self.preview_display.uid].scroll_to(offset=0, duration=100)

    async def load_prev_post_page(self, e):
        if self.current_post_page > 1: self.current_post_page -= 1; await self._load_and_display_post_page()
    async def load_next_post_page(self, e):
        if self.current_post_page < self.total_post_pages: self.current_post_page += 1; await self._load_and_display_post_page()

    def _build_rich_preview(self, thread: tb_typing.Thread, posts: list[tb_typing.Post], all_comments: dict[int, list[tb_typing.Comment]]):
        lz_user_name = getattr(thread.user, 'user_name', '未知用户'); self.preview_display.controls.clear()
        for post in posts:
            post_content = core.format_contents(post.contents).strip() or "(无正文)"; post_floor = post.floor
            self.preview_display.controls.append(self._create_post_widget_by_user(post.user, post_content, "主楼" if post_floor == 1 else f"{post_floor}楼", lz_user_name))
            if post.pid in all_comments:
                comment_container = ft.Column(spacing=5)
                for comment in all_comments[post.pid]:
                    comment_content = core.format_contents(comment.contents).strip()
                    if not comment_content: continue
                    comment_container.controls.append(self._create_post_widget_by_user(comment.user, comment_content, "回复", lz_user_name, is_comment=True))
                self.preview_display.controls.append(ft.Container(content=comment_container, padding=ft.padding.only(left=20, top=5, bottom=10)))
        self.page.update()

    def _update_analysis_progress(self, current_chunk, total_chunks, page_start, page_end):
        self.analysis_progress_bar.value = current_chunk / total_chunks
        self.log_message(f"分析进度: {current_chunk}/{total_chunks} (正在处理第 {page_start}-{page_end} 页)"); self.page.update()

    async def analyze_thread_click(self, e):
        current_tid = self.selected_thread.tid
        self.analyze_button.disabled = True; self.generate_button.disabled = True
        self.analysis_display.value = "⏳ 开始分批次分析，请稍候..."; self.analysis_progress_bar.visible = True; self.analysis_progress_bar.value = 0; self.page.update()
        async with tb.Client() as tieba_client:
            self.analysis_result = await core.analyze_stance_by_page(tieba_client, self.gemini_client, current_tid, self.total_post_pages, self.settings["analyzer_model"], self.log_message, self._update_analysis_progress, self.settings.get("pages_per_api_call", 4))
        self.analysis_progress_bar.visible = False; self.analyze_button.disabled = False
        if "summary" in self.analysis_result:
            self.analysis_cache[current_tid] = self.analysis_result; self.current_analysis_tid = current_tid
            self.analysis_display.value = f"## 讨论状况摘要\n\n{self.analysis_result['summary']}"; self.generate_button.disabled = False
        else: self.analysis_display.value = f"❌ 分析失败:\n\n{self.analysis_result.get('error', '未知错误')}"
        self.page.update()

    async def _execute_ai_reply_action(self, core_function, action_name: str, **kwargs):
        cached_analysis = self.analysis_cache.get(self.selected_thread.tid)
        if not cached_analysis or "summary" not in cached_analysis:
            self.log_message(f"错误：未找到当前帖子的分析摘要，无法{action_name}回复。", LogLevel.ERROR); return
        self.current_mode_id = self.mode_selector.value
        if not self.current_mode_id: self.log_message("请先选择一个回复模式！", LogLevel.WARNING); return
        selected_mode_config = core.PROMPTS.get('reply_generator', {}).get('modes', {}).get(self.current_mode_id, {})
        custom_input = None
        if selected_mode_config.get('is_custom', False):
            custom_input = self.custom_view_input.value.strip()
            if not custom_input: self.log_message("使用此自定义模型时，自定义内容不能为空！", LogLevel.WARNING); return
        self.generate_reply_ring.visible = True; self.generate_button.disabled = True; self.optimize_button.disabled = True
        self.copy_button.disabled = True; self.reply_display.value = f"⏳ {action_name}中，请稍候..."; self.page.update()
        generated_reply = await core_function(self.gemini_client, self.discussion_text, cached_analysis["summary"], self.current_mode_id, self.settings["generator_model"], self.log_message, custom_input=custom_input, **kwargs)
        self.reply_display.value = generated_reply; self.generate_reply_ring.visible = False; self.generate_button.disabled = False
        self._update_optimize_button_state(); self.copy_button.disabled = not bool(generated_reply); self.page.update()

    async def generate_reply_click(self, e): await self._execute_ai_reply_action(core.generate_reply, "生成")
    async def optimize_reply_click(self, e):
        reply_draft = self.reply_draft_input.value.strip()
        if not reply_draft and self.reply_display.value:
            self.log_message("优化草稿为空，自动使用已有回复进行优化。"); reply_draft = self.reply_display.value.strip()
            self.reply_draft_input.value = reply_draft; self.page.update()
        if not reply_draft: self.log_message("错误：没有可供优化的内容。", LogLevel.ERROR); return
        await self._execute_ai_reply_action(core.optimize_reply, "优化", reply_draft=reply_draft)

    def _update_optimize_button_state(self):
        has_draft = bool(self.reply_draft_input.value and self.reply_draft_input.value.strip())
        has_existing_reply = bool(self.reply_display.value and self.reply_display.value.strip() and "⏳" not in self.reply_display.value)
        self.optimize_button.disabled = not (has_draft or has_existing_reply)
        if self.page: self.page.update()

    def on_draft_input_change(self, e): self._update_optimize_button_state()

    async def search_tieba(self, e):
        if not self.tieba_name_input.value.strip(): self.log_message("贴吧名称不能为空，请先输入。", level=LogLevel.WARNING); return
        self.current_page_num = 1; query = self.search_query_input.value.strip(); self.current_search_query = query if query else None
        await self._fetch_and_display_threads()

    async def load_next_page(self, e): self.current_page_num += 1; await self._fetch_and_display_threads()
    async def load_prev_page(self, e):
        if self.current_page_num > 1: self.current_page_num -= 1; await self._fetch_and_display_threads()
    
    async def _fetch_and_display_threads(self):
        tieba_name = self.tieba_name_input.value.strip()
        if not tieba_name: self.log_message("错误：贴吧名称不能为空。", LogLevel.ERROR); return
        if not self.gemini_client: self.log_message("Gemini客户端未初始化，请先在设置中配置有效的API Key。", LogLevel.ERROR); return
        self.progress_ring.visible = True; self.search_button.disabled = True; self.prev_page_button.disabled = True; self.next_page_button.disabled = True; self.page.update()
        async with tb.Client() as tieba_client:
            if self.current_search_query: self.threads = await core.search_threads_by_page(tieba_client, tieba_name, self.current_search_query, self.current_page_num, self.log_message)
            else:
                try: sort_type = ThreadSortType(int(self.sort_type_dropdown.value))
                except (ValueError, TypeError): self.log_message(f"警告：无效的排序值。将使用默认排序。", LogLevel.WARNING); sort_type = ThreadSortType.REPLY
                self.threads = await core.fetch_threads_by_page(tieba_client, tieba_name, self.current_page_num, sort_type, self.log_message)
        self._update_thread_list_view(); self.progress_ring.visible = False; self.search_button.disabled = False
        self.page_num_display.value = f"第 {self.current_page_num} 页"
        self.prev_page_button.disabled = self.current_page_num <= 1; self.next_page_button.disabled = not self.threads
        self.main_view_content_area.content = self._build_main_view_content()
        self.page.update()

    def _update_thread_list_view(self):
        self.thread_list_view.controls.clear()
        if not self.threads: self.thread_list_view.controls.append(ft.Text("这一页没有找到帖子。", text_align=ft.TextAlign.CENTER)); return
        for thread in self.threads:
            user_name = "未知用户"
            if hasattr(thread, 'user') and thread.user: user_name = thread.user.user_name
            elif hasattr(thread, 'show_name'): user_name = thread.show_name
            reply_num_text = str(thread.reply_num) if hasattr(thread, 'reply_num') else "N/A"
            list_tile = ft.ListTile(leading=ft.Icon(ft.Icons.ARTICLE_OUTLINED), title=ft.Text(f"{thread.title}", weight=ft.FontWeight.BOLD), subtitle=ft.Text(f"作者: {user_name} | 回复数: {reply_num_text}"), on_click=self.select_thread, data=thread)
            self.thread_list_view.controls.append(list_tile)

    def _update_custom_view_visibility(self):
        current_mode_id = self.mode_selector.value
        if not current_mode_id: self.custom_view_input.visible = False; self.page.update(); return
        modes = core.PROMPTS.get('reply_generator', {}).get('modes', {})
        self.custom_view_input.visible = modes.get(current_mode_id, {}).get('is_custom', False)
        self.page.update()

    def _populate_mode_dropdown(self):
        sorted_modes = core.get_sorted_reply_modes(); default_mode_ids = core.get_default_mode_ids()
        options = []; valid_mode_ids = set()
        def truncate_text(text, max_length=30): return text[:max_length] + "..." if len(text) > max_length else text
        for mode_id, config in sorted_modes:
            prefix = "⚙️ " if mode_id in default_mode_ids else "👤 "
            display_text = f"{prefix}{config.get('name', '未命名')} - {truncate_text(config.get('description', ''))}"
            options.append(ft.dropdown.Option(key=mode_id, text=display_text)); valid_mode_ids.add(mode_id)
        self.mode_selector.options = options
        if self.current_mode_id and self.current_mode_id in valid_mode_ids: self.mode_selector.value = self.current_mode_id
        elif options: self.mode_selector.value = options[0].key; self.current_mode_id = options[0].key
        else: self.mode_selector.value = None; self.current_mode_id = None
        self._update_custom_view_visibility()

    def on_mode_change(self, e):
        self.current_mode_id = e.control.value; self._update_custom_view_visibility()

    def copy_reply_click(self, e): self.page.set_clipboard(self.reply_display.value); self._show_snackbar("回复已复制到剪贴板!","tertiary"); self.page.update()
    def copy_log_click(self, e):
        if not self.status_log.controls: self._show_snackbar("日志为空，无需复制。", "tertiary"); return
        log_texts = [row.controls[1].value for row in self.status_log.controls if isinstance(row, ft.Row) and len(row.controls) > 1 and hasattr(row.controls[1], 'value')]
        self.page.set_clipboard("\n".join(log_texts)); self._show_snackbar("所有日志已成功复制到剪贴板！", "primary"); self.page.update()

    async def back_to_main_view(self, e):
        self.navigation_rail.selected_index = 0
        await self.navigate(None)
        await asyncio.sleep(0.1)
        if self.page.scroll and self.thread_list_view.uid in self.page.scroll: self.page.scroll[self.thread_list_view.uid].scroll_to(offset=self.thread_list_scroll_offset, duration=100)
        self.page.update()

    def _save_mode_and_refresh_ui(self, mode_id: str, config: dict, success_message: str):
        if 'reply_generator' not in core.PROMPTS: core.PROMPTS['reply_generator'] = {}
        if 'modes' not in core.PROMPTS['reply_generator']: core.PROMPTS['reply_generator']['modes'] = {}
        core.PROMPTS['reply_generator']['modes'][mode_id] = config; core.save_prompts(core.PROMPTS)
        if hasattr(self, 'save_prompts_button'): self.save_prompts_button.disabled = True
        self.log_message(f"回复模式 '{config.get('name')}' (ID: {mode_id}) 已更新并保存。")
        self._show_snackbar(success_message, color_role="primary"); self._build_reply_modes_editor_list()
        if self.navigation_rail.selected_index == 1: self._populate_mode_dropdown()
        self.page.update()

    def share_mode_click(self, e):
        mode_id_to_share = e.control.data; modes = core.PROMPTS.get('reply_generator', {}).get('modes', {})
        if mode_id_to_share not in modes: self._show_snackbar(f"错误：找不到模式 '{mode_id_to_share}'", color_role="error"); return
        mode_config = modes[mode_id_to_share]
        share_data = {"tieba_gpt_mode_version": core.PROMPTS.get('prompts_version', 0), "name": mode_config.get("name", ""), "icon": mode_config.get("icon", "settings_suggest"), "description": mode_config.get("description", ""), "is_custom": mode_config.get("is_custom", False), "role": mode_config.get("role", ""), "task": mode_config.get("task", "")}
        try:
            json_string = json.dumps(share_data, indent=2, ensure_ascii=False); self.page.set_clipboard(json_string)
            self._show_snackbar(f"模式 '{mode_config.get('name')}' 已复制到剪贴板！", color_role="primary")
        except Exception as ex:
            self.log_message(f"序列化模式 '{mode_config.get('name')}' 失败: {ex}", LogLevel.ERROR); self._show_snackbar("复制失败，请检查日志。", color_role="error")
        self.page.update()

    def show_overwrite_confirmation(self, existing_mode_id, existing_mode_name, new_config):
        def handle_overwrite(ev):
            self.page.close(confirm_dialog); self._save_mode_and_refresh_ui(existing_mode_id, new_config, success_message=f"模式 '{existing_mode_name}' 已成功覆盖!")
        confirm_dialog = ft.AlertDialog(modal=True, title=ft.Text("模式名称冲突"), content=ft.Text(f"名为“{existing_mode_name}”的模式已存在。您要用导入的新配置覆盖它吗？"), actions=[ft.TextButton("取消", on_click=lambda _: self.page.close(confirm_dialog)), ft.FilledButton("覆盖", on_click=handle_overwrite)])
        self.page.open(confirm_dialog)

    def _create_mode_config_from_inputs(self, name: str, icon: str, description: str, role: str, task: str) -> dict:
        stripped_task = task.strip()
        return {"name": name.strip(), "icon": icon.strip(), "description": description.strip(), "is_custom": "{user_custom_input}" in stripped_task, "role": role.strip(), "task": stripped_task}

    def open_import_dialog(self, e):
        dialog_textfield = ft.TextField(label="模式分享码", hint_text="请在此处粘贴分享的模式JSON代码...", multiline=True, min_lines=10, max_lines=15, text_size=12)
        def do_import(ev):
            try:
                json_text = dialog_textfield.value
                if not json_text: dialog_textfield.error_text = "输入框不能为空！"; import_dialog.update(); return
                data = json.loads(json_text)
                if not isinstance(data, dict) or "tieba_gpt_mode_version" not in data: raise ValueError("这不是一个有效的TiebaGPT模式分享码。")
                imported_name = data.get("name")
                if not imported_name: raise ValueError("导入的模式缺少'name'字段。")
                new_config = self._create_mode_config_from_inputs(name=imported_name, icon=data.get("icon","settings_suggest"), description=data.get("description", ""), role=data.get("role", ""), task=data.get("task", ""))
                self.page.close(import_dialog); modes = core.PROMPTS.get('reply_generator', {}).get('modes', {})
                existing_mode_id = next((mode_id for mode_id, config in modes.items() if config.get("name") == imported_name), None)
                if existing_mode_id: self.show_overwrite_confirmation(existing_mode_id, imported_name, new_config)
                else: self._save_mode_and_refresh_ui(str(uuid.uuid4()), new_config, success_message=f"模式 '{imported_name}' 已成功导入!")
            except json.JSONDecodeError: dialog_textfield.error_text = "无效的JSON格式，请检查代码是否完整。"; import_dialog.update()
            except ValueError as ve: dialog_textfield.error_text = str(ve); import_dialog.update()
            except Exception as ex:
                self.log_message(f"导入模式时发生未知错误: {ex}", LogLevel.ERROR); self.page.close(import_dialog)
                self._show_snackbar("发生未知错误，请检查日志。", color_role="error")
        import_dialog = ft.AlertDialog(modal=True, title=ft.Text("导入回复模式"), content=ft.Container(content=dialog_textfield, width=500), actions=[ft.TextButton("取消", on_click=lambda _: self.page.close(import_dialog)), ft.FilledButton("导入", on_click=do_import)], actions_alignment=ft.MainAxisAlignment.END)
        self.page.open(import_dialog); self.page.update()

    async def open_mode_dialog(self, e):
        mode_id_to_edit = e.control.data if hasattr(e.control, 'data') else None; is_new_mode = mode_id_to_edit is None
        dialog_ai_error_text = ft.Text("请先填写名称和描述", color="error", visible=False); dialog_ai_result_text = ft.Text(color="tertiary", visible=False, weight=ft.FontWeight.BOLD)
        def clear_name_error_on_change(ev):
            if dialog_mode_name_input.error_text: dialog_mode_name_input.error_text = None; mode_editor_dialog.update()
            clear_ai_error_on_change(ev)
        def clear_ai_error_on_change(ev):
            if dialog_ai_error_text.visible: dialog_ai_error_text.visible = False; mode_editor_dialog.update()
        def update_buttons_state(ev):
            ai_optimize_button.disabled = not (dialog_role_input.value and dialog_task_input.value)
            dialog_is_custom_switch.value = "{user_custom_input}" in dialog_task_input.value; mode_editor_dialog.update()
        dialog_mode_name_input = ft.TextField(label="模式名称 (唯一)",on_change=clear_name_error_on_change)
        dialog_mode_icon_input = ft.TextField(label="图标 (可选)"); dialog_mode_desc_input = ft.TextField(label="模式描述",on_change=clear_ai_error_on_change)
        dialog_is_custom_switch = ft.Switch(label="需要自定义输入 (自动检测)", disabled=True)
        dialog_role_input = ft.TextField(label="角色 (Role)", multiline=True, min_lines=3, max_lines=5, on_change=update_buttons_state)
        dialog_task_input = ft.TextField(label="任务 (Task)", multiline=True, min_lines=5, max_lines=10, hint_text="若需用户输入，请用 {user_custom_input} 占位。", on_change=update_buttons_state)
        dialog_ai_progress = ft.ProgressRing(visible=False, width=16, height=16)
        ai_generate_button = ft.ElevatedButton("AI生成", icon=ft.Icons.AUTO_AWESOME, tooltip="根据名称和描述，让AI自动填写", on_click=None)
        ai_optimize_button = ft.ElevatedButton("AI优化", icon=ft.Icons.AUTO_FIX_HIGH, tooltip="让AI优化已填写的Role和Task", on_click=None)
        if not is_new_mode:
            config = core.PROMPTS['reply_generator']['modes'].get(mode_id_to_edit, {})
            dialog_mode_name_input.value = config.get("name", ""); dialog_mode_icon_input.value = config.get("icon","settings_suggest")
            dialog_mode_desc_input.value = config.get("description", ""); dialog_is_custom_switch.value = config.get("is_custom", False)
            dialog_role_input.value = config.get("role", ""); dialog_task_input.value = config.get("task", "")
        async def _handle_ai_mode_action(action_name: str, core_logic_callable, args: tuple):
            dialog_ai_result_text.visible = False; ai_generate_button.disabled = True; ai_optimize_button.disabled = True; dialog_ai_progress.visible = True; mode_editor_dialog.update()
            success, result = await core_logic_callable(self.gemini_client, self.settings["generator_model"], *args, self.log_message)
            if success:
                dialog_role_input.value = result.get("role", ""); dialog_task_input.value = result.get("task", "")
                dialog_ai_result_text.value = f"✓ {action_name}成功!"; dialog_ai_result_text.color = "tertiary"
            else: dialog_ai_result_text.value = f"✗ {action_name}失败: {result}"; dialog_ai_result_text.color = "error"
            dialog_ai_result_text.visible = True; ai_generate_button.disabled = False; dialog_ai_progress.visible = False; update_buttons_state(None); mode_editor_dialog.update()
            await asyncio.sleep(2); dialog_ai_result_text.visible = False; mode_editor_dialog.update()
        async def ai_generate_prompts_click(ev):
            mode_name = dialog_mode_name_input.value.strip(); mode_desc = dialog_mode_desc_input.value.strip()
            if not mode_name or not mode_desc:
                dialog_ai_error_text.value = "生成需要填写名称和描述"; dialog_ai_error_text.visible = True; mode_editor_dialog.update(); return
            await _handle_ai_mode_action("生成", core.generate_mode_prompts, (mode_name, mode_desc))
        async def ai_optimize_prompts_click(ev):
            mode_name = dialog_mode_name_input.value.strip(); mode_desc = dialog_mode_desc_input.value.strip()
            existing_role = dialog_role_input.value.strip(); existing_task = dialog_task_input.value.strip()
            if not all([mode_name, mode_desc, existing_role, existing_task]):
                dialog_ai_error_text.value = "优化需要所有字段都有内容"; dialog_ai_error_text.visible = True; mode_editor_dialog.update(); return
            await _handle_ai_mode_action("优化", core.optimize_mode_prompts, (mode_name, mode_desc, existing_role, existing_task))
        ai_generate_button.on_click = ai_generate_prompts_click; ai_optimize_button.on_click = ai_optimize_prompts_click
        def save_mode(ev):
            mode_name = dialog_mode_name_input.value.strip()
            if not mode_name: dialog_mode_name_input.error_text = "模式名称不能为空"; mode_editor_dialog.update(); return
            modes = core.PROMPTS.get('reply_generator', {}).get('modes', {})
            if any(cfg.get('name') == mode_name and mid != mode_id_to_edit for mid, cfg in modes.items()):
                dialog_mode_name_input.error_text = "已存在同名模式"; mode_editor_dialog.update(); return
            new_config = self._create_mode_config_from_inputs(name=dialog_mode_name_input.value, icon=dialog_mode_icon_input.value or "settings_suggest", description=dialog_mode_desc_input.value, role=dialog_role_input.value, task=dialog_task_input.value)
            mode_id_to_save = mode_id_to_edit if not is_new_mode else str(uuid.uuid4())
            self.page.close(mode_editor_dialog); self._save_mode_and_refresh_ui(mode_id_to_save, new_config, success_message=f"模式 '{mode_name}' 已保存!")
        mode_editor_dialog = ft.AlertDialog(modal=True, title=ft.Text("添加新模式" if is_new_mode else "编辑模式"),
            content=ft.Column(controls=[ft.Container(margin=ft.margin.only(top=5)), dialog_mode_name_input, dialog_mode_icon_input, dialog_mode_desc_input, ft.Row([ai_generate_button, ai_optimize_button, dialog_ai_progress, dialog_ai_error_text,dialog_ai_result_text], vertical_alignment=ft.CrossAxisAlignment.CENTER), ft.Divider(), dialog_is_custom_switch, dialog_role_input, dialog_task_input], scroll=ft.ScrollMode.ADAPTIVE, spacing=15, width=500, height=500),
            actions=[ft.TextButton("取消", on_click=lambda _: self.page.close(mode_editor_dialog)), ft.FilledButton("保存", on_click=save_mode)], actions_alignment=ft.MainAxisAlignment.END)
        self.page.open(mode_editor_dialog); self.page.update()

    def delete_mode_click(self, e):
        mode_id = e.control.data; mode_name = core.PROMPTS.get('reply_generator', {}).get('modes', {}).get(mode_id, {}).get("name", "未知模式")
        confirm_dialog = ft.AlertDialog(modal=True, title=ft.Text("确认删除"), content=ft.Text(f"您确定要永久删除回复模式 “{mode_name}” 吗？"), actions_alignment=ft.MainAxisAlignment.END)
        def confirm_delete(ev):
            if 'reply_generator' in core.PROMPTS and 'modes' in core.PROMPTS['reply_generator'] and mode_id in core.PROMPTS['reply_generator']['modes']:
                del core.PROMPTS['reply_generator']['modes'][mode_id]; core.save_prompts(core.PROMPTS)
                self.log_message(f"回复模式 '{mode_name}' 已删除。"); self._show_snackbar(f"模式 '{mode_name}' 已删除!", color_role="primary")
            self.page.close(confirm_dialog); self._build_reply_modes_editor_list()
        confirm_dialog.actions = [ft.TextButton("取消", on_click=lambda _: self.page.close(confirm_dialog)), ft.FilledButton("确认删除", on_click=confirm_delete, bgcolor=ft.Colors.RED_700)]
        self.page.open(confirm_dialog); self.page.update()

    def _show_prompt_update_dialog(self, user_v, default_v):
        def handle_incremental_merge(e, prefer_user : bool = False):
            self.page.close(update_dialog); self.progress_ring.visible = True; self.page.update()
            success, msg = core.merge_default_prompts(prefer_user); self.log_message(msg, LogLevel.INFO if success else LogLevel.ERROR)
            self.progress_ring.visible = False
            if success:
                self._show_snackbar(msg, color_role="primary")
                if self.navigation_rail.selected_index == 2: self.settings_content_area.content = self._build_advanced_settings_tab()
            else: self._show_snackbar(f"更新失败: {msg}",color_role="error")
            self.page.update()
        def handle_incremental_update(e): handle_incremental_merge(e, prefer_user=True)
        def handle_full_restore(e):
            self.page.close(update_dialog); self.progress_ring.visible = True; self.page.update()
            success, msg = core.restore_default_prompts(); self.log_message(msg, LogLevel.INFO if success else LogLevel.ERROR)
            self.progress_ring.visible = False
            if success:
                 self._show_snackbar("已彻底恢复为默认配置！", color_role="secondary")
                 if self.navigation_rail.selected_index == 2: self.settings_content_area.content = self._build_advanced_settings_tab()
            else: self._show_snackbar(f"恢复失败: {msg}",color_role="error")
            self.page.update()
        update_dialog = ft.AlertDialog(modal=True, title=ft.Text("💡 配置更新提示"), content=ft.Text(f"检测到新的配置可用！\n\n您的配置版本: {user_v}\n最新配置版本: {default_v}\n\n建议进行“增量更新”以获取新功能。"),
            actions=[ft.TextButton("稍后提示", on_click=lambda _: self.page.close(update_dialog)), ft.ElevatedButton("增量更新", on_click=handle_incremental_update, icon=ft.Icons.UPGRADE, tooltip="添加新功能，保留您已修改的。"), ft.ElevatedButton("增量覆盖", on_click=handle_incremental_merge, icon=ft.Icons.MERGE_TYPE, tooltip="用最新默认值覆盖，保留新增的。"), ft.FilledButton("彻底覆盖", on_click=handle_full_restore, icon=ft.Icons.SETTINGS_BACKUP_RESTORE, tooltip="警告：删除所有自定义模式！")],
            actions_alignment=ft.MainAxisAlignment.END)
        self.page.open(update_dialog)

def main(page: ft.Page):
    app = TiebaGPTApp(page)

    main_layout = ft.Row(
        [
            app.navigation_rail,
            ft.VerticalDivider(width=1),
            app.content_view,
        ],
        expand=True,
    )
    
    page.add(main_layout)
    app.initialize_app()

if __name__ == "__main__":
    ft.app(target=main)