import asyncio
import os
import re
import json
import sys
import typing
import shutil
import subprocess
import aiotieba as tb
import httpx
from aiotieba import ThreadSortType
from aiotieba import typing as tb_typing
from google import genai
from google.genai import types

VERSION = "1.5.3"
SETTINGS_FILE = "settings.json"
PROMPTS_FILE = "prompts.json"
DEFAULT_PROMPTS_FILE = "prompts.default.json"
POSTS_PER_PAGE = 30
README_FILE = "README.md"
README_URL = "https://raw.githubusercontent.com/LaplaceDemon29/TiebaGPT/main/README.md"

def get_app_version() -> str:
    base_version = VERSION
    try:
        git_hash_bytes = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL)
        git_hash = git_hash_bytes.decode('utf-8').strip()
        return f"{base_version}.{git_hash}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return base_version
        
def load_settings() -> dict:
    default_settings = {"api_key": "","analyzer_model": "gemini-1.5-flash-latest","generator_model": "gemini-1.5-flash-latest","available_models": [],"color_scheme_seed": "blue","pages_per_api_call": 4}
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            user_settings = json.load(f)
            default_settings.update(user_settings)
            return default_settings
    except (FileNotFoundError, json.JSONDecodeError):
        return default_settings

def save_settings(settings_data: dict):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings_data, f, indent=4)

PROMPTS = {}
def load_prompts():
    global PROMPTS
    if not os.path.exists(PROMPTS_FILE):
        try:
            shutil.copy(DEFAULT_PROMPTS_FILE, PROMPTS_FILE)
        except FileNotFoundError:
            return False, f"错误: 默认Prompt配置文件 '{DEFAULT_PROMPTS_FILE}' 未找到，应用无法运行。"
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            PROMPTS = json.load(f)
        return True, "Prompts 加载成功。"
    except (json.JSONDecodeError, Exception) as e:
        return False, f"加载 Prompts 时发生错误: {e}"

def check_prompts_version() -> tuple[str, int, int]:
    try:
        with open(DEFAULT_PROMPTS_FILE, 'r', encoding='utf-8') as f:
            default_prompts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"严重错误: 无法加载默认配置文件 {DEFAULT_PROMPTS_FILE}. {e}")
        return "ERROR", 0, 0

    default_version = default_prompts.get("prompts_version", 1)
    user_version = PROMPTS.get("prompts_version", 0)

    if user_version < default_version:
        return "NEEDS_UPDATE", user_version, default_version
    else:
        return "UP_TO_DATE", user_version, default_version

def save_prompts(prompts_data: dict):
    with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=4, ensure_ascii=False)
def restore_default_prompts():
    try:
        shutil.copy(DEFAULT_PROMPTS_FILE, PROMPTS_FILE)
        return load_prompts()
    except (FileNotFoundError, Exception) as e:
        return False, f"恢复默认 Prompts 时发生错误: {e}"
def merge_default_prompts(prefer_user: bool = False) -> tuple[bool, str]:
    try:
        with open(DEFAULT_PROMPTS_FILE, 'r', encoding='utf-8') as f:
            default_prompts = json.load(f)
    except FileNotFoundError:
        return False, f"错误: 无法找到默认Prompt配置文件: '{DEFAULT_PROMPTS_FILE}'"
    except json.JSONDecodeError as e:
        return False, f"错误: 加载默认Prompts时发生JSON解析错误: {e}"

    def deep_merge(target: dict, source: dict, preserve_target_values: bool):
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(target.get(key), dict) and isinstance(value, dict):
                deep_merge(target[key], value, preserve_target_values)
            elif not preserve_target_values:
                target[key] = value

    deep_merge(PROMPTS, default_prompts, preserve_target_values=prefer_user)

    if "prompts_version" in default_prompts:
        PROMPTS["prompts_version"] = default_prompts["prompts_version"]

    save_prompts(PROMPTS)

    if prefer_user:
        message = "Prompts更新成功！您的自定义设置已保留。"
    else:
        message = "Prompts更新成功！与默认设置冲突的条目已被覆盖。"

    return True, message

def build_stance_analyzer_prompt(discussion_text: str) -> str:
    prompt_config = PROMPTS['stance_analyzer']
    tasks_text = "\n".join([f"- {task}" for task in prompt_config['tasks']])
    prompt_parts = [
        prompt_config['system_prompt'],
        "\n[分析任务]\n" + tasks_text,
        f"\n[帖子和讨论的结构化文本]\n{discussion_text[:30000]}",
        f"\n[输出要求]\n{prompt_config['output_format_instruction']}"
    ]
    return "\n".join(prompt_parts)

def build_analysis_summarizer_prompt(chunk_summaries: list[dict]) -> str:
    prompt_config = PROMPTS['analysis_summarizer']
    tasks_text = "\n".join([f"- {task}" for task in prompt_config['tasks']])
    summaries_text = ""
    for i, summary in enumerate(chunk_summaries):
        summaries_text += f"\n--- 分块摘要 {i+1} ---\n{summary}"
    prompt_parts = [
        prompt_config['system_prompt'],
        "\n[整合任务]\n" + tasks_text,
        prompt_config['input_format_instruction'],
        summaries_text,
        f"\n[输出要求]\n{prompt_config['output_format_instruction']}"
    ]
    return "\n".join(prompt_parts)

def build_reply_generator_prompt(discussion_text: str, analysis_summary: str, mode_id: str, custom_input: typing.Optional[str] = None) -> str:
    gen_config = PROMPTS['reply_generator']
    mode_config = gen_config['modes'].get(mode_id)
    if not mode_config:
        raise ValueError(f"未找到 ID 为 '{mode_id}' 的回复模式配置。")
    
    mode_name = mode_config.get("name", "未知模式")
    role_prompt = mode_config['role']
    task_description = mode_config['task']
    
    if mode_config.get('is_custom', False):
        if not custom_input:
            raise ValueError(f"使用模式 '{mode_name}' 时，必须提供自定义输入。")
        task_description = task_description.format(user_custom_input=custom_input)

    rules_config = gen_config['common_rules']
    rules_text = rules_config['title'] + "\n" + "\n".join([f"- {rule}" for rule in rules_config['rules']])
    
    return f"""{role_prompt}

[你的任务]
{task_description}

---
[讨论状况摘要]
{analysis_summary}
---
[讨论背景原文]
{discussion_text[:15000]}
---
{rules_text}
""".strip()

def build_reply_optimizer_prompt(discussion_text: str, analysis_summary: str, mode_id: str, reply_draft: str, custom_input: typing.Optional[str] = None) -> str:
    optimizer_template = PROMPTS.get('reply_optimizer', {}).get('system_prompt')
    if not optimizer_template:
        raise ValueError("未找到 'reply_optimizer' 的 prompt 模板配置。")
    gen_config = PROMPTS['reply_generator']
    mode_config = gen_config['modes'].get(mode_id)
    if not mode_config:
        raise ValueError(f"未找到 ID 为 '{mode_id}' 的回复模式配置。")
    
    role_prompt = mode_config['role']
    task_prompt = mode_config['task']

    if mode_config.get('is_custom', False):
        if not custom_input:
            raise ValueError(f"使用模式 '{mode_config.get('name')}' 时，必须提供自定义输入。")
        task_prompt = task_prompt.format(user_custom_input=custom_input)

    return optimizer_template.format(
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        discussion_text=discussion_text[:15000],
        analysis_summary=analysis_summary,
        reply_draft=reply_draft
    )

def build_mode_generator_prompt(mode_name: str, mode_description: str) -> str:
    prompt_template = PROMPTS.get('mode_generator', {}).get('system_prompt')
    if not prompt_template:
        raise ValueError("未找到 'mode_generator' 的 prompt 模板配置。")
    return prompt_template.format(
        mode_name=mode_name,
        mode_description=mode_description
    )

def build_mode_optimizer_prompt(mode_name: str, mode_description: str, existing_role: str, existing_task: str) -> str:
    prompt_template = PROMPTS.get('mode_optimizer', {}).get('system_prompt')
    if not prompt_template:
        raise ValueError("未找到 'mode_optimizer' 的 prompt 模板配置。")
    return prompt_template.format(
        mode_name=mode_name,
        mode_description=mode_description,
        existing_role=existing_role,
        existing_task=existing_task
    )

async def _call_gemini_for_json_mode(
    client: genai.Client, model_name: str, prompt: str, log_callback: typing.Callable
) -> typing.Tuple[bool, typing.Union[dict, str]]:
    generation_config = {"response_mime_type": "application/json"}
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    try:
        response = await asyncio.to_thread(
            client.models.generate_content, model=model_name, contents=contents, config=generation_config
        )
        if response.text:
            result = json.loads(response.text)
            if "role" in result and "task" in result:
                log_callback("AI 操作成功！")
                return True, result
            else:
                log_callback("AI返回的JSON格式不正确，缺少'role'或'task'字段。")
                return False, "AI返回的JSON格式不正确，缺少'role'或'task'字段。"
        else:
            feedback_info = "未知原因"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback_info = str(response.prompt_feedback)
            log_callback(f"AI未返回有效文本。反馈: {feedback_info}")
            return False, f"AI未能生成内容。可能原因：内容安全策略触发。反馈: {feedback_info}"
    except json.JSONDecodeError:
        log_callback("AI返回的内容不是有效的JSON格式。")
        return False, "AI返回的内容不是有效的JSON格式。"
    except Exception as e:
        log_callback(f"调用Gemini时出错: {e}")
        return False, f"调用API时出错: {e}"

async def generate_mode_prompts(
    client: genai.Client, model_name: str, mode_name: str, mode_description: str, log_callback: typing.Callable
) -> typing.Tuple[bool, typing.Union[dict, str]]:
    log_callback(f"正在请求AI为模式“{mode_name}”生成Role和Task...")
    if not mode_name or not mode_description:
        return False, "模式名称和描述不能为空。"
    try:
        prompt = build_mode_generator_prompt(mode_name, mode_description)
    except ValueError as e:
        return False, str(e)
    return await _call_gemini_for_json_mode(client, model_name, prompt, log_callback)

async def optimize_mode_prompts(
    client: genai.Client, model_name: str,mode_name: str, mode_description: str,existing_role: str, existing_task: str,log_callback: typing.Callable
) -> typing.Tuple[bool, typing.Union[dict, str]]:
    log_callback(f"正在请求AI优化模式“{mode_name}”的Role和Task...")
    if not all([mode_name, mode_description, existing_role, existing_task]):
        return False, "所有输入字段（名称、描述、Role、Task）都不能为空。"
    try:
        prompt = build_mode_optimizer_prompt(mode_name, mode_description, existing_role, existing_task)
    except ValueError as e:
        return False, str(e)
    return await _call_gemini_for_json_mode(client, model_name, prompt, log_callback)

async def fetch_gemini_models(api_key: str) -> typing.Tuple[bool, typing.Union[list[str], str]]:
    if not api_key: return False, "API Key 不能为空。"
    try:
        temp_client = genai.Client(api_key=api_key); model_list = await asyncio.to_thread(temp_client.models.list)
        return True, sorted([m.name for m in model_list])
    except Exception as e: return False, f"获取模型列表失败: {e}"

async def fetch_threads_by_page(client: tb.Client, tieba_name: str, page_num: int, sort_type: ThreadSortType, log_callback: typing.Callable) -> list[tb_typing.Thread]:
    try:
        sort_map = {ThreadSortType.REPLY: "回复时间", ThreadSortType.CREATE: "发布时间", ThreadSortType.HOT: "热门"}
        log_callback(f"正在获取“{tieba_name}”吧第 {page_num} 页的帖子 (排序: {sort_map.get(sort_type, '默认')})...")
        return await client.get_threads(tieba_name, pn=page_num, sort=sort_type)
    except Exception as e: log_callback(f"获取第 {page_num} 页帖子失败: {e}"); return []

async def search_threads_by_page(client: tb.Client, tieba_name: str, query: str, page_num: int, log_callback: typing.Callable) -> list[tb_typing.Thread]:
    try:
        log_callback(f"正在“{tieba_name}”吧中搜索关键词“{query}”的第 {page_num} 页...")
        return await client.search_exact(tieba_name, query, pn=page_num, only_thread=True)
    except Exception as e: log_callback(f"搜索关键词“{query}”失败: {e}"); return []

async def fetch_full_thread_data(client: tb.Client, tid: int, log_callback: typing.Callable, page_num: int = 1) -> tuple[typing.Optional[tb_typing.Thread], typing.Optional[tb_typing.Posts], dict[int, list[tb_typing.Comment]]]:
    log_callback(f"正在获取帖子 {tid} 第 {page_num} 页的数据...")
    
    posts_obj: tb_typing.Posts = await client.get_posts(tid, pn=page_num, rn=POSTS_PER_PAGE)
    
    if not posts_obj:
        return None, None, {}
    
    thread_obj = posts_obj.thread
    all_comments: dict[int, list[tb_typing.Comment]] = {}
    
    post_list = posts_obj.objs
    
    results = await asyncio.gather(*[client.get_comments(tid, post.pid) for post in post_list], return_exceptions=True)
    for post, comments_or_exc in zip(post_list, results):
        if isinstance(comments_or_exc, Exception):
            pass
        elif comments_or_exc:
            all_comments[post.pid] = comments_or_exc
            
    return thread_obj, posts_obj, all_comments

def format_contents(contents: tb_typing.contents) -> str:
    if not contents or not contents.objs: return ""
    parts = []
    for frag in contents.objs:
        type_name = type(frag).__name__
        if type_name == 'FragText': parts.append(frag.text)
        elif type_name == 'FragEmoji': parts.append(f"[表情:{frag.desc}]")
        elif type_name in ['FragImage_p', 'FragImage_c', 'FragImage_t']: parts.append("[图片]")
        elif type_name == 'FragAt': parts.append(frag.text)
        elif type_name == 'FragLink': parts.append(f"[链接:{frag.text}]")
        elif type_name in ['FragVoice_p', 'FragVoice_c']: parts.append("[语音]")
    return " ".join(parts).strip()

def _format_user_info(user, lz_user_name: str = "") -> str:
    if not user:
        return "(用户: 未知用户)"

    parts = []
    user_name = getattr(user, 'user_name', '未知用户')
    parts.append(f"用户: {user_name}")

    nick_name = getattr(user, 'nick_name', '无昵称')
    parts.append(f"昵称: {nick_name}")

    if lz_user_name and user_name == lz_user_name: # 没有检查是否是未知用户，api保证user_name唯一性
        parts.append("楼主")

    if getattr(user, 'is_bawu', False):
        parts.append("吧务")
    
    level = getattr(user, 'level', None)
    if level is not None and level > 0:
        parts.append(f"Lv.{level}")

    ip_addr = getattr(user, 'ip', None)
    if ip_addr:
        parts.append(f"IP:{ip_addr}")
    
    return f"({', '.join(parts)})"

def format_main_post_text(thread: tb_typing.Thread) -> str:
    if not thread:
        return ""
    lz_user_name = getattr(thread.user, 'user_name', '未知用户')
    lz_info_str = _format_user_info(thread.user, lz_user_name)
    content_text = format_contents(thread.contents)
    return f"[帖子标题]: {thread.title}\n[主楼] {lz_info_str}\n{content_text}"

def format_discussion_text(thread: tb_typing.Thread, posts: list[tb_typing.Post], all_comments: dict[int, list[tb_typing.Comment]]) -> str:
    formatted_list = []

    lz_user_name = getattr(thread.user, 'user_name', '未知用户')
    if posts:
        formatted_list.append("---")
        formatted_list.append("[讨论区]")

    for post in posts:
        if post.floor == 1:
            continue
        
        post_text = format_contents(post.contents).strip()
        if not post_text:
            continue
            
        user_info_str = _format_user_info(post.user, lz_user_name)
        formatted_list.append(f"\n[回复 {post.floor}楼] {user_info_str}")
        formatted_list.append(post_text)
        
        if post.pid in all_comments:
            for j, comment in enumerate(all_comments[post.pid]):
                comment_text = format_contents(comment.contents).strip()
                if not comment_text:
                    continue
                comment_user_info_str = _format_user_info(comment.user, lz_user_name)
                formatted_list.append(f"  [楼中楼 to {post.floor}楼, #{j+1}] {comment_user_info_str}")
                formatted_list.append(f"  > {comment_text}")
                
    return "\n".join(formatted_list)

async def _analyze_single_chunk(gemini_client: genai.Client, discussion_text: str, model_name: str, log_callback: typing.Callable) -> dict:
    prompt = build_stance_analyzer_prompt(discussion_text)
    generation_config = {"response_mime_type": "text/plain"}
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    try:
        response = await asyncio.to_thread(gemini_client.models.generate_content, model=model_name, contents=contents, config=generation_config)
        if response.text and response.text.strip():
            return {"summary": response.text.strip()}
        else:
            return {"error": "AI未能生成有效的摘要内容。"}
    except Exception as e:
        log_callback(f"Gemini API 分块分析调用失败: {e}")
        return {"error": str(e)}

async def _summarize_analyses(gemini_client: genai.Client, chunk_summaries: list[dict], model_name: str, log_callback: typing.Callable) -> dict:
    log_callback(f"--- 使用模型 {model_name} 整合 {len(chunk_summaries)} 个摘要块 ---")
    prompt = build_analysis_summarizer_prompt(chunk_summaries)
    generation_config = {"response_mime_type": "text/plain"}
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    try:
        log_callback("正在调用 Gemini API 进行最终整合...")
        response = await asyncio.to_thread(gemini_client.models.generate_content, model=model_name, contents=contents, config=generation_config)
        log_callback("Gemini API 整合调用成功。")
        if response.text and response.text.strip():
            return {"summary": response.text.strip()}
        else:
            return {"error": "AI未能生成有效的最终摘要。"}
    except Exception as e:
        log_callback(f"Gemini API 整合调用失败: {e}")
        return {"error": f"整合失败: {e}"}

async def analyze_stance_by_page(tieba_client: tb.Client, gemini_client: genai.Client, tid: int, total_pages: int, model_name: str, log_callback: typing.Callable, progress_callback: typing.Callable, pages_per_call: int) -> dict:
    log_callback(f"--- 开始对TID {tid} 进行分块分析，共 {total_pages} 页，每块 {pages_per_call} 页 ---")
    chunk_results = []
    
    thread_obj, _, _ = await fetch_full_thread_data(tieba_client, tid, log_callback, page_num=1)
    if not thread_obj:
        return {"error": "无法获取帖子主楼信息，分析中止。"}
    
    main_post_text = format_main_post_text(thread_obj)

    total_chunks = (total_pages + pages_per_call - 1) // pages_per_call
    current_chunk = 0

    for i in range(1, total_pages + 1, pages_per_call):
        current_chunk += 1
        page_start = i
        page_end = min(i + pages_per_call - 1, total_pages)
        
        progress_callback(current_chunk, total_chunks, page_start, page_end)
        
        chunk_posts_list = []
        chunk_comments = {}
        for page_num in range(page_start, page_end + 1):
            log_callback(f"  正在获取第 {page_num} 页内容...")
            _, posts_obj, comments = await fetch_full_thread_data(tieba_client, tid, log_callback, page_num=page_num)
            if posts_obj and posts_obj.objs:
                chunk_posts_list.extend(posts_obj.objs)
                chunk_comments.update(comments)
        
        if not chunk_posts_list:
            log_callback(f"警告：块 {current_chunk} (页 {page_start}-{page_end}) 没有获取到内容，跳过。")
            continue
        
        discussion_part_text = format_discussion_text(thread_obj, chunk_posts_list, chunk_comments)
        full_discussion_text = f"{main_post_text}\n{discussion_part_text}"

        chunk_result = await _analyze_single_chunk(gemini_client, full_discussion_text, model_name, log_callback)
        chunk_results.append({"analysis_failed": True, "chunk": current_chunk, **chunk_result})
            
    successful_summaries = [r['summary'] for r in chunk_results if 'summary' in r]
    if not successful_summaries:
        first_error = next((r['error'] for r in chunk_results if 'error' in r), "所有分析块均失败，无法生成最终报告。")
        return {"error": first_error}
    
    if len(successful_summaries) == 1:
        log_callback("只有一个分析块成功，直接返回该块摘要。")
        return {"summary": successful_summaries[0]}
        
    final_analysis_result = await _summarize_analyses(gemini_client, successful_summaries, model_name, log_callback)
    return final_analysis_result

async def generate_reply(client: genai.Client, discussion_text: str, analysis_summary: str, mode_id: str, model_name: str, log_callback: typing.Callable, custom_input: typing.Optional[str] = None) -> str:
    modes = PROMPTS.get('reply_generator', {}).get('modes', {})
    mode_name = modes.get(mode_id, {}).get("name", "未知模式")
    log_callback(f"--- 使用模型 {model_name} 和 “{mode_name}”模式生成回复 ---")
    try:
        prompt = build_reply_generator_prompt(discussion_text, analysis_summary, mode_id, custom_input)
    except Exception as e:
        return f"构建Prompt失败: {e}"

    generation_config = {"response_mime_type": "text/plain"}
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    try:
        log_callback("正在调用 Gemini API 生成回复...")
        response = await asyncio.to_thread(client.models.generate_content, model=model_name, contents=contents, config=generation_config)
        if response.text and response.text.strip():
            log_callback("Gemini API 回复生成成功。")
            return response.text.strip()
        else:
            feedback_info = "未知原因"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback_info = str(response.prompt_feedback)
            log_callback(f"Gemini API 未返回有效文本。可能原因：内容安全策略触发。反馈: {feedback_info}")
            return f"生成回复失败：AI未能生成内容。\n\n这通常是由于安全设置或内容审查策略导致的。\n\n(API反馈: {feedback_info})"
    except Exception as e:
        log_callback(f"Gemini API 回复生成失败: {e}"); return f"生成回复失败: {e}"

async def optimize_reply(client: genai.Client, discussion_text: str, analysis_summary: str, mode_id: str, model_name: str, log_callback: typing.Callable, reply_draft: str, custom_input: typing.Optional[str] = None) -> str:
    modes = PROMPTS.get('reply_generator', {}).get('modes', {})
    mode_name = modes.get(mode_id, {}).get("name", "未知模式")
    log_callback(f"--- 使用模型 {model_name} 和 “{mode_name}”模式优化已有回复 ---")
    
    try:
        prompt = build_reply_optimizer_prompt(discussion_text, analysis_summary, mode_id, reply_draft, custom_input)
    except Exception as e:
        return f"构建优化Prompt失败: {e}"
    generation_config = {"response_mime_type": "text/plain"}
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    try:
        log_callback("正在调用 Gemini API 优化回复...")
        response = await asyncio.to_thread(client.models.generate_content, model=model_name, contents=contents, config=generation_config)
        if response.text and response.text.strip():
            log_callback("Gemini API 回复优化成功。")
            return response.text.strip()
        else:
            feedback_info = "未知原因"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback_info = str(response.prompt_feedback)
            return f"优化回复失败：AI未能生成内容。\n\n(API反馈: {feedback_info})"
    except Exception as e:
        log_callback(f"Gemini API 回复优化失败: {e}")
        return f"优化回复失败: {e}"

_DEFAULT_MODE_IDS = None

def get_default_mode_ids() -> set:
    global _DEFAULT_MODE_IDS
    if _DEFAULT_MODE_IDS is not None:
        return _DEFAULT_MODE_IDS
    try:
        with open(DEFAULT_PROMPTS_FILE, 'r', encoding='utf-8') as f:
            default_prompts = json.load(f)
        mode_ids = set(default_prompts.get('reply_generator', {}).get('modes', {}).keys())
        _DEFAULT_MODE_IDS = mode_ids
        return mode_ids
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def get_sorted_reply_modes() -> list[tuple[str, dict]]:
    modes = PROMPTS.get('reply_generator', {}).get('modes', {})
    if not modes:
        return []
    default_mode_ids = get_default_mode_ids()

    def sort_key(item: tuple[str, dict]) -> tuple[int, str]:
        mode_id, config = item
        priority = 0 if mode_id in default_mode_ids else 1
        name = config.get('name', '')
        return (priority, name)

    return sorted(modes.items(), key=sort_key)

def _process_readme_for_flet(content: str) -> str:

    badge_pattern = re.compile(r"\[!\[([^\]]+)\]\([^)]+\)\]\(([^)]+)\)")
    
    replacement_format = r"- **\1:** [view](\2)"
    
    processed_content = badge_pattern.sub(replacement_format, content)
    
    return processed_content

async def get_readme_content() -> typing.Tuple[bool, str]:

    try:
        if os.path.exists(README_FILE):
            with open(README_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(README_URL, follow_redirects=True)
                response.raise_for_status()
                content = response.text
                with open(README_FILE, 'w', encoding='utf-8') as f:
                    f.write(content)
        return True, _process_readme_for_flet(content)
    except httpx.RequestError as e:
        error_msg = f"网络请求错误: 无法下载 README.md。请检查您的网络连接。\n\n错误详情: {e}"
        return False, f"# 下载失败\n\n{error_msg}"
    except Exception as e:
        error_msg = f"处理 README.md 时发生未知错误。\n\n错误详情: {e}"
        return False, f"# 加载失败\n\n{error_msg}"