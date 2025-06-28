import asyncio
import os
import json
import sys
import typing
import shutil
import subprocess
import aiotieba as tb
from aiotieba import ThreadSortType
from aiotieba import typing as tb_typing
from google import genai
from google.genai import types

VERSION = "1.0.8"
SETTINGS_FILE = "settings.json"
PROMPTS_FILE = "prompts.json"
DEFAULT_PROMPTS_FILE = "prompts.default.json"

def get_app_version() -> str:
    base_version = VERSION
    try:
        git_hash_bytes = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        )
        git_hash = git_hash_bytes.decode('utf-8').strip()
        return f"{base_version}.{git_hash}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return base_version

# --- 1. 配置加载/保存 ---

def load_settings() -> dict:
    """加载 settings.json 配置文件，并确保所有关键字段存在"""
    default_settings = {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "analyzer_model": "gemini-1.5-flash-latest",
        "generator_model": "gemini-1.5-flash-latest",
        "available_models": []
    }
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            user_settings = json.load(f)
            # 合并用户设置和默认设置，以防旧的配置文件缺少字段
            default_settings.update(user_settings)
            return default_settings
    except (FileNotFoundError, json.JSONDecodeError):
        return default_settings

def save_settings(settings_data: dict):
    """保存设置到 settings.json"""
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
    except json.JSONDecodeError as e:
        return False, f"错误: Prompt配置文件 '{PROMPTS_FILE}' 格式无效: {e}"
    except Exception as e:
        return False, f"加载 Prompts 时发生未知错误: {e}"
def save_prompts(prompts_data: dict):
    with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=4, ensure_ascii=False)
def restore_default_prompts():
    try:
        shutil.copy(DEFAULT_PROMPTS_FILE, PROMPTS_FILE)
        return load_prompts()
    except FileNotFoundError:
        return False, f"错误: 默认Prompt配置文件 '{DEFAULT_PROMPTS_FILE}' 未找到，无法恢复。"
    except Exception as e:
        return False, f"恢复默认 Prompts 时发生错误: {e}"
def build_stance_analyzer_prompt(discussion_text: str) -> str:
    prompt_config = PROMPTS['stance_analyzer']; prompt_parts = [prompt_config['system_prompt']]; task_description = []
    for i, task in enumerate(prompt_config['tasks']):
        desc = f"{i+1}. `{task['field']}`: {task['description']}"
        if 'sub_fields' in task:
            sub_desc = [f"    - `{sub['field']}`: {sub['description']}" for sub in task['sub_fields']]
            desc += "\n" + "\n".join(sub_desc)
        task_description.append(desc)
    prompt_parts.extend(task_description); prompt_parts.append(f"\n[帖子和讨论的结构化文本]\n{discussion_text[:20000]}"); prompt_parts.append(f"\n[输出格式要求]\n{prompt_config['output_format_instruction']}")
    return "\n".join(prompt_parts)
def build_reply_generator_prompt(discussion_text: str, analysis_result: dict, mode: str, custom_viewpoint: typing.Optional[str] = None) -> str:
    gen_config = PROMPTS['reply_generator']; mode_config = gen_config['modes'][mode]; role_prompt = mode_config['role']; scenarios = mode_config['scenarios']; task_description = ""
    if mode == "自定义模型":
        if not custom_viewpoint: raise ValueError("使用自定义模型时，必须提供 custom_viewpoint。")
        task_description = scenarios['main_task'].format(user_viewpoint=custom_viewpoint)
    else:
        is_debatable = analysis_result.get('is_debatable', False); dominant_stance_id = analysis_result.get('dominant_stance_id', 'None'); main_stances = analysis_result.get('main_stances', []); format_args = {}
        if is_debatable and main_stances:
            sorted_stances = sorted(main_stances, key=lambda x: x.get('count', 0), reverse=True)
            if dominant_stance_id != 'None' and sorted_stances:
                dominant_stance = next((s for s in sorted_stances if s.get('stance_id') == dominant_stance_id), None)
                if dominant_stance:
                    scenario_key = 'dominant_side_exists'; format_args['dominant_summary'] = dominant_stance.get('summary', '未知观点')
                    if '{minority_summary}' in scenarios[scenario_key]:
                        minority_stance = next((s for s in sorted_stances if s.get('stance_id') != dominant_stance_id), None)
                        format_args['minority_summary'] = minority_stance.get('summary', '另一个未知观点') if minority_stance else "没有其他反对观点"
                    task_description = scenarios[scenario_key].format(**format_args)
                else: task_description = scenarios['multiple_sides_no_dominant']
            else: task_description = scenarios['multiple_sides_no_dominant']
        else: task_description = scenarios['no_debate']
    rules_config = gen_config['common_rules']; rules_text = rules_config['title'] + "\n" + "\n".join([f"- {rule}" for rule in rules_config['rules']])
    return f"""{role_prompt}\n\n现在有一个帖子正在讨论，我已经为你分析好了当前的讨论状况。请根据讨论状况和你的任务，写一个符合你角色的回复。\n\n[讨论背景]\n{discussion_text[:15000]}\n\n[讨论状况分析]\n{json.dumps(analysis_result, ensure_ascii=False, indent=2)}\n\n[你的任务]\n{task_description}\n\n{rules_text}""".strip()


async def fetch_gemini_models(api_key: str) -> typing.Tuple[bool, typing.Union[list[str], str]]:
    """根据提供的API Key，动态获取可用的Gemini模型列表"""
    if not api_key:
        return False, "API Key 不能为空。"
    try:
        # 必须在函数内创建临时客户端以验证Key
        temp_client = genai.Client(api_key=api_key)
        
        model_list = await asyncio.to_thread(temp_client.models.list)
        
        usable_models = [m.name for m in model_list]
        
        return True, sorted(usable_models)
    except Exception as e:
        return False, f"获取模型列表失败: {e}"

async def fetch_threads_by_page(client: tb.Client, tieba_name: str, page_num: int, sort_type: ThreadSortType, log_callback: typing.Callable) -> list[tb_typing.Thread]:
    """
    根据指定的贴吧名、页码和排序方式获取主题帖列表。
    """
    try:
        sort_map = {ThreadSortType.REPLY: "回复时间", ThreadSortType.CREATE: "发布时间", ThreadSortType.HOT: "热门"}
        await log_callback(f"正在获取“{tieba_name}”吧第 {page_num} 页的帖子 (排序: {sort_map.get(sort_type, '默认')})...")
        threads = await client.get_threads(tieba_name, pn=page_num, sort=sort_type)
        return threads
    except Exception as e:
        await log_callback(f"获取第 {page_num} 页帖子失败: {e}")
        return []

async def search_threads_by_page(client: tb.Client, tieba_name: str, query: str, page_num: int, log_callback: typing.Callable) -> list[tb_typing.Thread]:
    """
    根据指定的贴吧名、页码和关键词搜索主题帖列表。
    """
    try:
        await log_callback(f"正在“{tieba_name}”吧中搜索关键词“{query}”的第 {page_num} 页...")
        # only_thread=True is highly recommended to filter out non-thread results
        threads = await client.search_exact(tieba_name, query, pn=page_num, only_thread=True)
        return threads
    except Exception as e:
        await log_callback(f"搜索关键词“{query}”失败: {e}")
        return []

async def get_thread_info(client: tb.Client, tid: int, log_callback: typing.Callable) -> typing.Optional[tb_typing.Thread]:
    try:
        posts = await client.get_posts(tid, rn=1)
        if posts:
            return posts[0].thread
        else:
            await log_callback(f"无法获取TID {tid}的元数据，因为帖子列表为空。")
            return None
    except Exception as e:
        await log_callback(f"获取TID {tid}的元数据失败: {e}")
        return None

async def fetch_full_thread_data(client: tb.Client, tid: int, log_callback: typing.Callable) -> tuple[list[tb_typing.Post], dict[int, list[tb_typing.Comment]]]:
    await log_callback(f"正在获取帖子 {tid} 的完整数据...")
    posts: list[tb_typing.Post] = await client.get_posts(tid)
    all_comments: dict[int, list[tb_typing.Comment]] = {}
    posts_to_process_comments = posts[:50]; await log_callback(f"共获取到 {len(posts)} 条回复，将尝试获取前 {len(posts_to_process_comments)} 条回复的楼中楼。")
    results = await asyncio.gather(*[client.get_comments(tid, post.pid) for post in posts_to_process_comments], return_exceptions=True)
    for i, (post, comments_or_exc) in enumerate(zip(posts_to_process_comments, results)):
        if isinstance(comments_or_exc, Exception): await log_callback(f"  - 获取回复(pid: {post.pid})的楼中楼失败: {comments_or_exc}")
        elif comments_or_exc: all_comments[post.pid] = comments_or_exc; await log_callback(f"  - 成功获取回复(pid: {post.pid})的 {len(comments_or_exc)} 条楼中楼。")
    return posts, all_comments

def format_contents(contents: tb_typing.contents) -> str:
    if not contents or not contents.objs:
        return ""

    parts = []
    for frag in contents.objs:
        type_name = type(frag).__name__

        if type_name == 'FragText':
            parts.append(frag.text)
        elif type_name == 'FragEmoji':
            parts.append(f"[表情:{frag.desc}]")
        elif type_name in ['FragImage_p', 'FragImage_c', 'FragImage_t']:
            parts.append("[图片]")
        elif type_name == 'FragAt':
            parts.append(frag.text)
        elif type_name == 'FragLink':
            parts.append(f"[链接:{frag.text}]")
        elif type_name in ['FragVoice_p', 'FragVoice_c']:
            parts.append("[语音]")
        else:
            print(f"DEBUG: 忽略了未知类型 -> {type_name}")
            pass
    return " ".join(parts).strip()

def format_discussion_text(thread: tb_typing.Thread, posts: list[tb_typing.Post], all_comments: dict[int, list[tb_typing.Comment]]) -> str:
    formatted_list = [f"[帖子标题]: {thread.title}", f"[主楼内容]\n{format_contents(thread.contents)}", "---", "[讨论区]"]
    for post in posts:
        if post.floor == 1: continue
        post_text = format_contents(post.contents).strip();
        if not post_text: continue
        user_name = post.user.user_name if post.user and hasattr(post.user, 'user_name') else '未知用户'; formatted_list.append(f"\n[回复 {post.floor}楼] (用户: {user_name})"); formatted_list.append(post_text)
        if post.pid in all_comments:
            for j, comment in enumerate(all_comments[post.pid]):
                comment_text = format_contents(comment.contents).strip();
                if not comment_text: continue
                comment_user_name = comment.user.user_name if comment.user and hasattr(comment.user, 'user_name') else '未知用户'; formatted_list.append(f"  [楼中楼 to {post.floor}楼, #{j+1}] (用户: {comment_user_name})"); formatted_list.append(f"  > {comment_text}")
    return "\n".join(formatted_list)

async def analyze_stance(client: genai.Client, discussion_text: str, model_name: str, log_callback: typing.Callable) -> dict:
    await log_callback(f"--- 使用模型 {model_name} 分析讨论状况 ---")
    prompt = build_stance_analyzer_prompt(discussion_text)
    generation_config = {"response_mime_type": "application/json"}
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    try:
        await log_callback("正在调用 Gemini API 进行分析...")
        response = await asyncio.to_thread(client.models.generate_content, model=model_name, contents=contents, config=generation_config)
        await log_callback("Gemini API 分析调用成功。")
        return json.loads(response.text)
    except Exception as e:
        await log_callback(f"Gemini API 分析调用失败: {e}"); return {"error": str(e), "is_debatable": False, "main_stances": [], "neutral_or_off_topic_count": 0, "dominant_stance_id": "None"}

async def generate_reply(client: genai.Client, discussion_text: str, analysis_result: dict, mode: str, model_name: str, log_callback: typing.Callable, custom_viewpoint: typing.Optional[str] = None) -> str:
    await log_callback(f"--- 使用模型 {model_name} 和 “{mode}”模式生成回复 ---")
    prompt = build_reply_generator_prompt(discussion_text, analysis_result, mode, custom_viewpoint)
    generation_config = {"response_mime_type": "text/plain"}
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    try:
        await log_callback("正在调用 Gemini API 生成回复...")
        response = await asyncio.to_thread(client.models.generate_content, model=model_name, contents=contents, config=generation_config)
        if response.text and response.text.strip():
            await log_callback("Gemini API 回复生成成功。")
            return response.text.strip()
        else:
            feedback_info = "未知原因"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                feedback_info = str(response.prompt_feedback)
            await log_callback(f"Gemini API 未返回有效文本。可能原因：内容安全策略触发。反馈: {feedback_info}")
            return f"生成回复失败：AI未能生成内容。\n\n这通常是由于安全设置或内容审查策略导致的。请尝试使用不同的模式或调整您的自定义观点。\n\n(API反馈: {feedback_info})"
        await log_callback("Gemini API 回复生成成功。"); return response.text.strip()
    except Exception as e: await log_callback(f"Gemini API 回复生成失败: {e}"); return f"生成回复失败: {e}"